"""Subsampling curve (plan 4.2) and the customer-sized test set (plan 4.3).

Subsampling: draw N training rows (without replacement, ``numpy.random.default_rng(seed)``) from a
fixed training pool, fit, and score a fixed test set; repeat over seeds. For each seed the same rows
go to every model, so the models are compared on identical data. ``full`` is the whole pool (every
seed is then the same fit).

Models: the pipeline's L2 logistic regression on the v2 design, and
``HistGradientBoostingClassifier`` on a compact ordinal design (``gbdt_design``): one column per v2
feature. Ordered features (``ORDERED_LEVELS``: band, spend, time_on_page, seniority) are integer
ranks with a monotone increasing constraint (``monotone_constraints``); an absent value (band
``missing``, spend without enrichment, time on page without a session, seniority not asked) is NaN,
which the GBDT routes on its own. Every other feature is categorical. The planted time-on-page
effect is not monotone (60-300s +0.4, >600s −0.4), nor is size (201-1000 +1.3, 1000+ +1.0), so an
unconstrained GBDT is reported next to the constrained one.

Customer-sized test set: draw ``CUSTOMER_N`` labelled leads, hold out the latest ``CUSTOMER_HOLDOUT``
share by ``created_at`` (a customer validates on their most recent leads), fit on the rest and
bootstrap the AUC on the holdout; repeat over draws and report the mean CI width.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from emva.constants import MISSING, V2_LEVELS
from emva.eval.bootstrap import N_RESAMPLES, SEED, auc_ci
from emva.features import CATS_V2
from emva.model import make_lr

SIZES: tuple[int | None, ...] = (300, 500, 1000, 2000, 3000, None)   # None = the full pool
N_SEEDS: int = 20
CUSTOMER_N: int = 2000
CUSTOMER_HOLDOUT: float = 0.2
CUSTOMER_DRAWS: int = 20

# Ordered v2 features for the GBDT, low to high. Values not listed (the absent levels) become NaN.
ORDERED_LEVELS: dict[str, tuple[str, ...]] = {
    "band": ("1-10", "11-50", "51-200", "201-1000", "1000+"),
    "spend": ("none", "under £5k", "£5k-£25k", "£25k-£100k", "£100k+"),
    "time_on_page": ("15-60s", "60-300s", "300-600s", ">600s"),
    "seniority": ("student", "junior/ic", "mid", "senior"),
}
# v2 puts these features at their reference level when an indicator says the input is absent; the GBDT gets NaN.
ABSENT_WHEN: dict[str, str] = {"spend": "enrichment_missing", "time_on_page": "session_missing"}
GBDT_MAX_ITER: int = 100
GBDT_LEARNING_RATE: float = 0.1


@dataclass(frozen=True)
class GbdtDesign:
    """The GBDT's input matrix, its monotone constraint vector (+1 / 0 per column) and categorical mask."""

    matrix: pd.DataFrame
    monotone: tuple[int, ...]
    categorical: tuple[bool, ...]


def gbdt_design(X: pd.DataFrame, features: Sequence[str] = tuple(CATS_V2)) -> GbdtDesign:
    """Ordinal design over ``features`` (default: the v2 model's features, in design order).

    Ordered features become ranks 0..k-1 (NaN when absent, see ``ABSENT_WHEN`` and ``ORDERED_LEVELS``)
    with constraint +1; every other feature becomes its index in ``V2_LEVELS`` (categorical, constraint 0).
    Raises ``ValueError`` on a value outside ``V2_LEVELS`` (as ``emva.design.fixed_design`` does).
    """
    cols: dict[str, pd.Series] = {}
    categorical = []
    for f in features:
        values = X[f].astype(str)
        unknown = sorted(set(values) - set(V2_LEVELS[f]))
        if unknown:
            raise ValueError(f"feature {f!r} has values outside its declared levels: {unknown[:5]}")
        if f in ORDERED_LEVELS:
            rank = values.map({lvl: i for i, lvl in enumerate(ORDERED_LEVELS[f])}).astype(float)
            if f in ABSENT_WHEN:
                rank = rank.mask(X[ABSENT_WHEN[f]].eq("yes"))
            cols[f] = rank
            categorical.append(False)
        else:
            cols[f] = values.map({lvl: i for i, lvl in enumerate(V2_LEVELS[f])}).astype(float)
            categorical.append(True)
    return GbdtDesign(pd.DataFrame(cols, index=X.index), constraint_vector(features), tuple(categorical))


def make_gbdt(design: GbdtDesign, constrained: bool, seed: int = SEED) -> HistGradientBoostingClassifier:
    """Unfitted GBDT for ``design``; with ``constrained`` the monotone vector applies, else none."""
    return HistGradientBoostingClassifier(
        max_iter=GBDT_MAX_ITER, learning_rate=GBDT_LEARNING_RATE, random_state=seed,
        categorical_features=list(design.categorical),
        monotonic_cst=list(design.monotone) if constrained else None)


# fit_score(train_positions, test_positions) -> scores for the test positions
FitScore = Callable[[np.ndarray, np.ndarray], np.ndarray]


def matrix_model(matrix: pd.DataFrame, y: pd.Series, factory: Callable[[], object]) -> FitScore:
    """A ``FitScore`` fitting ``factory()`` on rows of ``matrix`` given by position."""
    M, yv = matrix.to_numpy(dtype=float), y.to_numpy(dtype=float)

    def fit_score(tr: np.ndarray, te: np.ndarray) -> np.ndarray:
        return factory().fit(M[tr], yv[tr]).predict_proba(M[te])[:, 1]
    return fit_score


def subsample_indices(pool: np.ndarray, n: int | None, seed: int) -> np.ndarray:
    """``n`` positions drawn without replacement from ``pool`` with ``default_rng(seed)``; all of ``pool`` for None.

    Raises ``ValueError`` if ``n`` exceeds the pool.
    """
    if n is None:
        return np.asarray(pool)
    if n > len(pool):
        raise ValueError(f"cannot draw {n} rows from a pool of {len(pool)}")
    return np.sort(np.random.default_rng(seed).choice(pool, size=n, replace=False))


def subsample_curve(y: pd.Series, train: pd.Series, test: pd.Series, models: dict[str, FitScore],
                    sizes: Sequence[int | None] = SIZES, n_seeds: int = N_SEEDS) -> pd.DataFrame:
    """AUC on ``test`` for every model, size and seed (seeds 0..n_seeds-1): long frame model, N, full, seed, auc.

    ``train``/``test`` are boolean masks over ``y``. The full pool (size None) is fitted once, since every
    seed would draw the same rows. A size larger than the pool is skipped (listed in no row), and a draw
    with a single class raises (it cannot be fitted).
    """
    pool = np.flatnonzero(train.to_numpy())
    te = np.flatnonzero(test.to_numpy())
    yv = y.to_numpy(dtype=float)
    rows = []
    for n in sizes:
        if n is not None and n > len(pool):
            continue
        for seed in range(1 if n is None else n_seeds):   # the full pool is the same fit for every seed
            idx = subsample_indices(pool, n, seed)
            if yv[idx].min() == yv[idx].max():
                raise ValueError(f"subsample N={n} seed={seed} has a single class")
            for name, fit_score in models.items():
                rows.append({"model": name, "N": len(pool) if n is None else n, "full": n is None, "seed": seed,
                             "auc": float(roc_auc_score(yv[te], fit_score(idx, te)))})
    return pd.DataFrame(rows)


def curve_table(long: pd.DataFrame) -> pd.DataFrame:
    """Mean ± sd AUC per model and N over seeds (the single full-pool fit has no sd), one column per model."""
    g = long.groupby(["N", "full", "model"], sort=True).auc.agg(["mean", "std", "count"]).reset_index()
    g["cell"] = [f"{m:.3f}" if k == 1 else f"{m:.3f} ± {s:.3f}" for m, s, k in zip(g["mean"], g["std"], g["count"])]
    t = g.pivot(index=["N", "full"], columns="model", values="cell").reset_index()
    t["N"] = [f"{n} (full)" if f else str(n) for n, f in zip(t.N, t.full)]
    return t.drop(columns="full")[["N", *long.model.drop_duplicates()]]


@dataclass(frozen=True)
class CustomerDraw:
    """One customer-sized draw: holdout size and wins, AUC and its bootstrap CI width."""

    seed: int
    n_test: int
    wins_test: int
    auc: float
    width: float


def customer_sized(D: pd.DataFrame, y: pd.Series, eligible: pd.Series, created_at: pd.Series,
                   n: int = CUSTOMER_N, holdout: float = CUSTOMER_HOLDOUT, draws: int = CUSTOMER_DRAWS,
                   n_resamples: int = N_RESAMPLES) -> list[CustomerDraw]:
    """For seeds 0..draws-1: sample ``n`` labelled ``eligible`` leads, hold out the latest ``holdout`` share.

    Fits the formula model on the earlier rows and bootstraps the holdout AUC (``n_resamples``, seed 0,
    the report convention). A draw whose holdout or training part lacks a class raises.
    """
    pool = np.flatnonzero((eligible & y.notna()).to_numpy())
    Dv, yv, t = D.to_numpy(dtype=float), y.to_numpy(dtype=float), created_at.to_numpy()
    n_test = int(round(n * holdout))
    out = []
    for seed in range(draws):
        idx = subsample_indices(pool, n, seed)
        idx = idx[np.argsort(t[idx], kind="stable")]
        tr, te = idx[:n - n_test], idx[n - n_test:]
        if yv[tr].min() == yv[tr].max() or yv[te].min() == yv[te].max():
            raise ValueError(f"customer draw seed={seed} has a single class in train or holdout")
        score = make_lr().fit(Dv[tr], yv[tr]).predict_proba(Dv[te])[:, 1]
        ci = auc_ci(yv[te], score, n_resamples, seed=SEED)
        out.append(CustomerDraw(seed, len(te), int(yv[te].sum()), ci.point, ci.width))
    return out


def lr_model(D: pd.DataFrame, y: pd.Series) -> FitScore:
    """The pipeline's formula model on design ``D`` as a ``FitScore``."""
    return matrix_model(D, y, make_lr)


def gbdt_models(X: pd.DataFrame, y: pd.Series) -> dict[str, FitScore]:
    """Constrained and unconstrained GBDT ``FitScore``s on ``gbdt_design(X)``."""
    g = gbdt_design(X)
    return {"GBDT monotone": matrix_model(g.matrix, y, lambda: make_gbdt(g, True)),
            "GBDT unconstrained": matrix_model(g.matrix, y, lambda: make_gbdt(g, False))}


def constraint_vector(features: Sequence[str] = tuple(CATS_V2)) -> tuple[int, ...]:
    """The GBDT's monotone constraint per column of ``gbdt_design`` over ``features``: +1 ordered, 0 otherwise."""
    return tuple(1 if f in ORDERED_LEVELS else 0 for f in features)


def constraint_table(features: Sequence[str] = tuple(CATS_V2)) -> pd.DataFrame:
    """The constraint vector as a table: column, kind, constraint, level order, absent handling (for the report)."""
    absent = {"band": f"NaN for {MISSING}", "seniority": "NaN for not_asked/blank",
              **{f: f"NaN when {ind}=yes" for f, ind in ABSENT_WHEN.items()}}
    rows = [{"column": f, "kind": "ordered" if m else "categorical", "constraint": f"{m:+d}" if m else "0",
             "order (low to high)": " < ".join(ORDERED_LEVELS.get(f, ())), "absent": absent.get(f, "")}
            for f, m in zip(features, constraint_vector(features))]
    return pd.DataFrame(rows)
