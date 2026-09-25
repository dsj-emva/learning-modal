"""Bootstrap confidence intervals for AUC and paired AUC comparisons.

Percentile bootstrap over test rows (resampled with replacement). Resamples that contain a
single class have no AUC and are redrawn, so every result has exactly ``n_resamples`` values.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

N_RESAMPLES: int = 1000
CI_LEVEL: float = 0.95
SEED: int = 0


@dataclass(frozen=True)
class BootstrapCI:
    """Point estimate on the full sample and a percentile CI from the bootstrap distribution."""

    point: float
    lo: float
    hi: float
    samples: np.ndarray

    @property
    def width(self) -> float:
        """``hi - lo``."""
        return self.hi - self.lo


@dataclass(frozen=True)
class PairedComparison:
    """Bootstrap of ``AUC(b) - AUC(a)`` on shared resamples.

    ``p_value`` is the two-sided bootstrap p-value for "no difference":
    ``2 * min(P(diff <= 0), P(diff >= 0))``, capped at 1.
    """

    auc_a: float
    auc_b: float
    diff: BootstrapCI
    p_value: float


def _resample_indices(y: np.ndarray, n_resamples: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Draw ``n_resamples`` index vectors of length ``len(y)``, each containing both classes."""
    n = len(y)
    out: list[np.ndarray] = []
    while len(out) < n_resamples:
        idx = rng.integers(0, n, n)
        if y[idx].min() != y[idx].max():
            out.append(idx)
    return out


def _percentile_ci(point: float, samples: np.ndarray, level: float) -> BootstrapCI:
    """Wrap ``point`` and the equal-tailed ``level`` percentile interval of ``samples``."""
    alpha = (1 - level) / 2
    lo, hi = np.quantile(samples, [alpha, 1 - alpha])
    return BootstrapCI(point=point, lo=float(lo), hi=float(hi), samples=samples)


def _check(y: np.ndarray, *scores: np.ndarray) -> None:
    """Raise ``ValueError`` unless ``y`` is binary with both classes and each score matches it in length, NaN-free."""
    if set(np.unique(y)) != {0.0, 1.0}:
        raise ValueError("y must contain both classes and only 0/1")
    for s in scores:
        if len(s) != len(y):
            raise ValueError(f"score length {len(s)} != label length {len(y)}")
        if np.isnan(s).any():
            raise ValueError("scores contain NaN")


def auc_ci(y: np.ndarray, score: np.ndarray, n_resamples: int = N_RESAMPLES,
           level: float = CI_LEVEL, seed: int = SEED) -> BootstrapCI:
    """AUC of ``score`` against binary ``y`` with a percentile bootstrap CI."""
    y, score = np.asarray(y, dtype=float), np.asarray(score, dtype=float)
    _check(y, score)
    rng = np.random.default_rng(seed)
    samples = np.array([roc_auc_score(y[i], score[i]) for i in _resample_indices(y, n_resamples, rng)])
    return _percentile_ci(float(roc_auc_score(y, score)), samples, level)


def paired_auc(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, n_resamples: int = N_RESAMPLES,
               level: float = CI_LEVEL, seed: int = SEED) -> PairedComparison:
    """Compare two score vectors on the same rows: CI of ``AUC(b) - AUC(a)`` and a bootstrap p-value."""
    y = np.asarray(y, dtype=float)
    a, b = np.asarray(score_a, dtype=float), np.asarray(score_b, dtype=float)
    _check(y, a, b)
    rng = np.random.default_rng(seed)
    diffs = np.array([roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i])
                      for i in _resample_indices(y, n_resamples, rng)])
    auc_a, auc_b = float(roc_auc_score(y, a)), float(roc_auc_score(y, b))
    p = min(1.0, 2 * min(float((diffs <= 0).mean()), float((diffs >= 0).mean())))
    return PairedComparison(auc_a=auc_a, auc_b=auc_b, diff=_percentile_ci(auc_b - auc_a, diffs, level), p_value=p)
