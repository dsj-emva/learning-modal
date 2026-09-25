"""Interaction detectability (Phase 5 hand-off): can the formula model see the two planted v2 interactions?

Generator v2 plants LinkedIn × company size 51+ (+0.8) and senior title × form D (−0.8) on close
log-odds (``data/v2/ground_truth.md``, 5.5). This check adds the two products of the pipeline's *own*
bucketed features to the v2 design (no ground truth: ``channel == linkedin`` × ``band`` in 51-200,
201-1000, 1000+; ``seniority == senior`` × ``form_variant == D``) and asks whether

1. their coefficients' 95% bootstrap CIs (1,000 refits of the training rows, seed 0) exclude zero, and
2. held-out AUC improves (paired bootstrap on the frozen test sets and on the rolling-origin headline).

On v1 the terms are not planted, so v1 is the null control.
"""
from __future__ import annotations

import pandas as pd

from emva.eval.bootstrap import N_RESAMPLES, SEED, BootstrapCI, coef_bootstrap
from emva.model import make_lr

INTERACTION_BANDS: tuple[str, ...] = ("51-200", "201-1000", "1000+")
INTERACTION_COLUMNS: tuple[str, ...] = ("linkedin_x_51plus", "senior_x_formD")
PLANTED_V2: dict[str, float] = {"linkedin_x_51plus": 0.8, "senior_x_formD": -0.8}


def interaction_columns(X: pd.DataFrame) -> pd.DataFrame:
    """The two interaction columns (0/1 floats) from the pipeline's v2 features ``channel``, ``band``, ``seniority``, ``form_variant``."""
    return pd.DataFrame({
        "linkedin_x_51plus": (X.channel.eq("linkedin") & X.band.isin(INTERACTION_BANDS)).astype(float),
        "senior_x_formD": (X.seniority.eq("senior") & X.form_variant.astype(str).eq("D")).astype(float),
    }, index=X.index)


def with_interactions(D: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """``D`` with ``interaction_columns(X)`` appended."""
    return D.join(interaction_columns(X))


def interaction_coef_cis(D: pd.DataFrame, y: pd.Series, train: pd.Series, n_resamples: int = N_RESAMPLES,
                         seed: int = SEED) -> dict[str, BootstrapCI]:
    """Bootstrap CIs of the interaction coefficients of the formula model fitted on ``train`` rows of ``D``.

    ``D`` must already contain ``INTERACTION_COLUMNS`` (``with_interactions``).
    """
    pos = [D.columns.get_loc(c) for c in INTERACTION_COLUMNS]
    cis = coef_bootstrap(D[train].to_numpy(dtype=float), y[train].to_numpy(dtype=float),
                         lambda A, b: make_lr().fit(A, b).coef_[0][pos], n_resamples, seed=seed)
    return dict(zip(INTERACTION_COLUMNS, cis))
