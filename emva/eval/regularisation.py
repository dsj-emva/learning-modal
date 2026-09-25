"""Regularisation sweep (plan 4.6), a standing check: does AUC depend on the L2 strength C?

For every C in ``C_GRID`` (log grid around the pipeline's ``LR_C``) the formula model is refitted with
``LogisticRegression(C=C, max_iter=LR_MAX_ITER)`` (otherwise ``emva.model.make_lr``) and scored on:

- (a) the frozen legacy test set, legacy labels, legacy features (the baseline's own sweep, which
  was flat at 0.8107 to 0.8138 on v1) and v2 features;
- (b) the frozen mature test set, horizon labels, v2 features, with its Brier score;
- the rolling-origin headline (ADR 0013): mean AUC over the sufficient splits, v2 features.

"Flat" is judged on the AUC range over ``FLAT_RANGE`` (C from 0.1 to 10): the check passes when that
range is below ``FLAT_TOLERANCE``.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from emva.constants import LR_C, LR_MAX_ITER
from emva.eval.rolling import FitPredict, RollingSpec, mean_auc, rolling_origin

C_GRID: tuple[float, ...] = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, LR_C, 1.0, 3.0, 10.0, 30.0, 100.0)
FLAT_RANGE: tuple[float, float] = (0.1, 10.0)
FLAT_TOLERANCE: float = 0.005


def make_lr_c(c: float) -> LogisticRegression:
    """The formula model with L2 strength ``c`` (every other setting as ``emva.model.make_lr``)."""
    return LogisticRegression(C=c, max_iter=LR_MAX_ITER)


def fit_predict_c(c: float) -> FitPredict:
    """A rolling ``FitPredict`` for the formula model with strength ``c``."""
    def fit_predict(D: pd.DataFrame, y: pd.Series, train: pd.Series) -> np.ndarray:
        return make_lr_c(c).fit(D[train], y[train]).predict_proba(D)[:, 1]
    return fit_predict


def frozen_auc(D: pd.DataFrame, y: pd.Series, train: pd.Series, test: pd.Series, c: float) -> tuple[float, float]:
    """``(AUC, Brier)`` on ``test`` of the model with strength ``c`` fitted on ``train``."""
    m = make_lr_c(c).fit(D[train], y[train])
    p = m.predict_proba(D[test])[:, 1]
    return float(roc_auc_score(y[test], p)), float(brier_score_loss(y[test], p))


def sweep(cases: dict[str, tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]],
          rolling: tuple[RollingSpec, pd.DataFrame, pd.Series, pd.Series] | None,
          grid: Sequence[float] = C_GRID, brier_for: Sequence[str] = ()) -> pd.DataFrame:
    """One row per C: AUC for each frozen case (name -> (D, y, train, test)), Brier for ``brier_for``, rolling mean.

    ``rolling`` = (spec, D, y, created_at) for the rolling-origin headline column, or None.
    """
    rows = []
    for c in grid:
        row: dict[str, float] = {"C": c}
        for name, (D, y, tr, te) in cases.items():
            auc, brier = frozen_auc(D, y, tr, te, c)
            row[name] = auc
            if name in brier_for:
                row[f"Brier {name}"] = brier
        if rolling is not None:
            spec, D, y, created = rolling
            m = mean_auc(rolling_origin(spec, D, y, created, fit_predict_c(c), with_ci=False))
            row["rolling-origin mean"] = np.nan if m is None else m
        rows.append(row)
    return pd.DataFrame(rows)


def flatness(table: pd.DataFrame, column: str, lo: float = FLAT_RANGE[0], hi: float = FLAT_RANGE[1]) -> tuple[float, float, bool]:
    """``(min, max, flat)`` of ``column`` over ``lo <= C <= hi``; flat when max - min < ``FLAT_TOLERANCE``."""
    v = table.loc[table.C.between(lo, hi), column]
    return float(v.min()), float(v.max()), bool(v.max() - v.min() < FLAT_TOLERANCE)
