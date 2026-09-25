"""Point metrics shared by the pipeline summary and the standard report."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from emva.constants import TOP_FRACTION


def _cut_size(n_rows: int, frac: float) -> int:
    """Number of rows in the top ``frac``: ``int(n_rows * frac)``; raises if that is 0."""
    n = int(n_rows * frac)
    if n < 1:
        raise ValueError(f"top {frac:.0%} of {n_rows} rows is empty; need at least {int(np.ceil(1 / frac))} rows")
    return n


def top_indices(score: np.ndarray, frac: float = TOP_FRACTION) -> np.ndarray:
    """Positions of the top ``int(len * frac)`` scores, via numpy's default (unstable) argsort.

    This matches the baseline. With tied scores at the cut the selection depends on the
    sort implementation; see ``tie_averaged_top_share``. Raises ``ValueError`` if the cut is empty.
    """
    s = np.asarray(score)
    return np.argsort(-s)[:_cut_size(len(s), frac)]


def top_share(target: pd.Series, score: np.ndarray, frac: float = TOP_FRACTION) -> float:
    """Share of ``target``'s total held by the top ``frac`` of rows ranked by ``score``."""
    return float(target.iloc[top_indices(score, frac)].sum() / target.sum())


def _cut_threshold(s: np.ndarray, frac: float) -> tuple[int, float]:
    """``(n, threshold)``: cut size and the score of the n-th highest row."""
    n = _cut_size(len(s), frac)
    return n, np.sort(s)[::-1][n - 1]


def ties_at_cut(score: np.ndarray, frac: float = TOP_FRACTION) -> int:
    """Number of rows whose score equals the score at the top-``frac`` cut (1 = no tie)."""
    s = np.asarray(score)
    _, thr = _cut_threshold(s, frac)
    return int((s == thr).sum())


def tie_averaged_top_share(target: pd.Series, score: np.ndarray, frac: float = TOP_FRACTION) -> float:
    """Top-``frac`` share where rows tied at the cut get equal fractional credit.

    This is the expected ``top_share`` over all tie orders, so it does not depend on the sort.
    """
    s = np.asarray(score)
    t = np.asarray(target, dtype=float)
    n, thr = _cut_threshold(s, frac)
    above, at = s > thr, s == thr
    frac_at = (n - above.sum()) / at.sum()
    return float((t[above].sum() + frac_at * t[at].sum()) / t.sum())


def bottom_share(flag: pd.Series, score: np.ndarray, frac: float) -> float:
    """Mean of boolean ``flag`` among the ``int(len * frac)`` rows with the lowest ``score``.

    Uses a stable sort, so rows tied at the cut are taken in their original order.
    Raises ``ValueError`` if the cut is empty or the lengths differ.
    """
    s = np.asarray(score, dtype=float)
    if len(s) != len(flag):
        raise ValueError(f"score length {len(s)} != flag length {len(flag)}")
    idx = np.argsort(s, kind="stable")[:_cut_size(len(s), frac)]
    return float(np.asarray(flag, dtype=float)[idx].mean())


def summary(name: str, y: pd.Series, p: np.ndarray, rev: pd.Series, value: np.ndarray | None = None) -> dict:
    """The baseline's summary row (``baseline/emva_score.py::report``, unchanged).

    Wins are ranked by ``p``; revenue by ``value`` when given, else by ``p``.
    """
    o = top_indices(p)
    ov = o if value is None else top_indices(value)
    return {"model": name, "auc": round(roc_auc_score(y, p), 3), "brier": round(brier_score_loss(y, np.clip(p, 0, 1)), 4),
            "top20_wins": round(y.iloc[o].sum() / y.sum(), 3), "top20_revenue": round(rev.iloc[ov].sum() / rev.sum(), 3)}


def calibration_by_decile(y: pd.Series, p: np.ndarray) -> pd.DataFrame:
    """Mean predicted vs observed win rate in each decile of ``p`` (decile 10 = highest p)."""
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    df["decile"] = pd.qcut(df.p.rank(method="first"), 10, labels=range(1, 11)).astype(int)
    out = df.groupby("decile").agg(n=("y", "size"), mean_p=("p", "mean"), observed=("y", "mean"))
    return out.reset_index()


def auc_by_month(y: pd.Series, score: np.ndarray, created_at: pd.Series) -> pd.DataFrame:
    """AUC per calendar month of ``created_at``; NaN for a month with a single class."""
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "s": np.asarray(score, dtype=float),
                       "month": created_at.dt.strftime("%Y-%m").values})
    rows = []
    for month, g in df.groupby("month"):
        auc = roc_auc_score(g.y, g.s) if g.y.nunique() == 2 else np.nan
        rows.append({"month": month, "n": len(g), "wins": int(g.y.sum()), "auc": auc})
    return pd.DataFrame(rows)


def value_scale(value: np.ndarray) -> dict:
    """Scale stats of a value vector: p50, p99, max, max/median and the top-1% share of total value."""
    v = np.sort(np.asarray(value, dtype=float))[::-1]
    k = max(1, int(len(v) * 0.01))
    med = float(np.median(v))
    return {"p50": med, "p99": float(np.percentile(v, 99)), "max": float(v[0]),
            "max_over_median": float(v[0] / med), "top1pct_share": float(v[:k].sum() / v.sum())}
