"""Calibration decay (plan 4.5): fit on data up to month M, check calibration on months M+1..M+3.

The windows reuse the rolling-origin rules (``emva.eval.rolling.RollingSpec``, ADR 0013) with the split
at the first day of month M+1: under the headline rule the model is trained on leads created before that
date whose horizon label is mature by then, and each evaluation month holds the labelled leads created in
it whose label is mature at ``AS_OF``. Evaluation months with no such lead are listed as empty.

Per evaluation month: Brier score, mean p vs observed win rate, the calibration slope (logistic
regression of y on logit p; 1 = well spread) and the calibration intercept (calibration-in-the-large:
the intercept of y on an offset of logit p; 0 = right level, negative = over-predicting). Plus a decile
table per M, pooled over its evaluation months.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from emva.constants import AS_OF
from emva.eval.metrics import calibration_by_decile
from emva.eval.rolling import FitPredict, RollingSpec, lr_fit_predict

CALIBRATION_MONTHS: tuple[str, ...] = ("2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06")
DECAY_MONTHS: int = 3
P_CLIP: tuple[float, float] = (1e-6, 1 - 1e-6)
NEWTON_MAX_ITER: int = 100
NEWTON_TOL: float = 1e-10


def month_start(month: str) -> pd.Timestamp:
    """First instant (UTC) of ``YYYY-MM``."""
    return pd.Timestamp(f"{month}-01", tz="UTC")


def decay_windows(spec: RollingSpec, created_at: pd.Series, y: pd.Series, month: str, k: int = DECAY_MONTHS,
                  as_of: pd.Timestamp = AS_OF) -> tuple[pd.Series, list[tuple[str, pd.Series]]]:
    """``(train, [(eval month, mask), ...])`` for a model fitted at the end of ``month``.

    ``spec`` must have a one-month test window (each evaluation month is that window moved on).
    """
    if spec.test_window is None:
        raise ValueError("calibration decay needs a spec with a one-month test window")
    cutoff = month_start(month) + pd.DateOffset(months=1)
    train, _ = spec.masks(created_at, y, cutoff, as_of)
    evals = []
    for j in range(k):
        start = cutoff + pd.DateOffset(months=j)
        evals.append((start.strftime("%Y-%m"), spec.masks(created_at, y, start, as_of)[1]))
    return train, evals


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, *P_CLIP)
    return np.log(q / (1 - q))


def _newton_logistic(Z: np.ndarray, y: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Unpenalised logistic regression of ``y`` on columns ``Z`` with ``offset`` (Newton-Raphson)."""
    beta = np.zeros(Z.shape[1])
    for _ in range(NEWTON_MAX_ITER):
        mu = 1 / (1 + np.exp(-(Z @ beta + offset)))
        grad = Z.T @ (y - mu)
        hess = (Z * (mu * (1 - mu))[:, None]).T @ Z
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.max(np.abs(step)) < NEWTON_TOL:
            return beta
    raise RuntimeError("calibration fit did not converge (separated classes?)")


@dataclass(frozen=True)
class CalibrationStats:
    """Calibration of ``p`` against ``y`` on one set of rows (NaN fields when a class is missing)."""

    n: int
    wins: int
    mean_p: float
    observed: float
    brier: float
    slope: float
    intercept: float


def calibration_stats(y: np.ndarray, p: np.ndarray) -> CalibrationStats:
    """Brier, mean p, observed rate, calibration slope and calibration-in-the-large intercept."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    n, wins = len(y), int(y.sum())
    if n == 0:
        return CalibrationStats(0, 0, np.nan, np.nan, np.nan, np.nan, np.nan)
    brier = float(brier_score_loss(y, p)) if 0 < wins < n else np.nan
    slope = intercept = np.nan
    if 0 < wins < n:
        lp = _logit(p)
        slope = float(_newton_logistic(np.column_stack([np.ones(n), lp]), y, np.zeros(n))[1])
        intercept = float(_newton_logistic(np.ones((n, 1)), y, lp)[0])
    return CalibrationStats(n, wins, float(p.mean()), float(y.mean()), brier, slope, intercept)


@dataclass
class DecayResult:
    """One fit month: training size, per-evaluation-month stats and the pooled evaluation rows."""

    month: str
    n_train: int
    wins_train: int
    evals: list[tuple[str, CalibrationStats]]
    y_pooled: np.ndarray
    p_pooled: np.ndarray


def calibration_decay(spec: RollingSpec, D: pd.DataFrame, y: pd.Series, created_at: pd.Series,
                      months: Sequence[str] = CALIBRATION_MONTHS, k: int = DECAY_MONTHS,
                      fit_predict: FitPredict = lr_fit_predict, as_of: pd.Timestamp = AS_OF) -> list[DecayResult]:
    """Fit at the end of each month in ``months`` and compute calibration on the next ``k`` months.

    A month whose training set lacks a class is returned with empty evaluations.
    """
    out = []
    for m in months:
        train, evals = decay_windows(spec, created_at, y, m, k, as_of)
        ytr = y[train]
        n_tr, w_tr = int(train.sum()), int((ytr == 1).sum())
        stats, ys, ps = [], [], []
        if 0 < w_tr < n_tr:
            p_all = fit_predict(D, y, train)
            for label_, mask in evals:
                yy, pp = y[mask].to_numpy(dtype=float), p_all[mask.to_numpy()]
                stats.append((label_, calibration_stats(yy, pp)))
                ys.append(yy)
                ps.append(pp)
        out.append(DecayResult(m, n_tr, w_tr, stats,
                               np.concatenate(ys) if ys else np.array([]), np.concatenate(ps) if ps else np.array([])))
    return out


def _f(x: float, dp: int = 3) -> str:
    return "n/a" if np.isnan(x) else f"{x:.{dp}f}"


def decay_table(results: Sequence[DecayResult]) -> pd.DataFrame:
    """One row per (fit month, evaluation month): sizes, mean p, observed, Brier, slope, intercept.

    A fit whose evaluation months are all empty gets a single row saying so.
    """
    rows = []
    for r in results:
        if not r.evals:
            rows.append({"fit through": r.month, "train (wins)": f"{r.n_train} ({r.wins_train})",
                         "eval month": "no fit: training set lacks a class"})
            continue
        if not any(s.n for _, s in r.evals):
            rows.append({"fit through": r.month, "train (wins)": f"{r.n_train} ({r.wins_train})",
                         "eval month": f"{r.evals[0][0]} to {r.evals[-1][0]}", "eval (wins)": "0: no labelled leads"})
            continue
        for k, (label_, s) in enumerate(r.evals, start=1):
            rows.append({"fit through": r.month, "train (wins)": f"{r.n_train} ({r.wins_train})",
                         "eval month": f"{label_} (M+{k})", "eval (wins)": f"{s.n} ({s.wins})" if s.n else "0: no labelled leads",
                         "mean p": _f(s.mean_p), "observed": _f(s.observed), "Brier": _f(s.brier, 4),
                         "slope": _f(s.slope, 2), "intercept": _f(s.intercept, 2)})
    return pd.DataFrame(rows).fillna("")


def decile_table(results: Sequence[DecayResult]) -> pd.DataFrame:
    """Decile of p (1 = lowest) × fit month: ``mean p / observed`` over the pooled evaluation rows."""
    out: pd.DataFrame | None = None
    for r in results:
        if len(r.y_pooled) < 10 or r.y_pooled.min() == r.y_pooled.max():
            continue
        d = calibration_by_decile(pd.Series(r.y_pooled), r.p_pooled)
        col = pd.DataFrame({"decile": d.decile,
                            f"fit through {r.month} (n={len(r.y_pooled)})": [f"{a:.3f} / {b:.3f}" for a, b in zip(d.mean_p, d.observed)]})
        out = col if out is None else out.merge(col, on="decile")
    return pd.DataFrame() if out is None else out
