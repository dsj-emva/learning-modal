"""Everything that differs between ``--feature-set legacy`` and ``v2``, in one object (like ``labels.LabelConfig``).

``feature_spec(FeatureSet.LEGACY)`` is the baseline: its features, its data-driven ``design`` and its
data-driven deal-value design and fixed deal-value residual sd, so ``--label-mode legacy --feature-set legacy``
reproduces ``baseline/`` byte for byte. ``feature_spec(FeatureSet.V2)`` is plan Phase 2: v2 features, the fixed
39-column design from ``constants.V2_LEVELS`` (ADR 0009), a residual sd estimated from training and (plan 3.1)
the fixed deal-value design from ``constants.DEAL_VALUE_LEVELS``. ``feature_spec(FeatureSet.GENERIC)`` (Phase 10,
ADR 0024) is the v2 spec with ``extras = True``: the pipeline and ``emva.scoring`` append the declared extras' design
columns (``emva.generic``, an encoder fitted on the training leads) to the v2 design; the deal-value model is v2's.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import pandas as pd

from emva.constants import CATS
from emva.design import design, fixed_design
from emva.eval.regression import BASELINE_DEAL_LOG_RESIDUAL_SD  # legacy feature set only: baseline byte identity
from emva.features import CATS_V2, FeatureSet, add_features, add_features_v2
from emva.value import deal_value_design, fixed_deal_value_design


@dataclass(frozen=True)
class FeatureSpec:
    """One feature set: its reference levels, featuriser, design functions and deal-value residual sd.

    ``featurise`` adds the feature columns to a cleaned, labelled frame in place and returns it;
    ``design`` turns that frame into the model matrix and ``value_design`` into the deal-value matrix;
    ``fixed_log_residual_sd`` is passed to ``value.fit_deal_value`` (None = estimate it from training residuals);
    ``extras`` is True when the design also takes the dataset's declared extra columns (``emva.generic``).
    """

    feature_set: FeatureSet
    cats: Mapping[str, str]
    featurise: Callable[[pd.DataFrame], pd.DataFrame]
    design: Callable[[pd.DataFrame], pd.DataFrame]
    value_design: Callable[[pd.DataFrame], pd.DataFrame]
    fixed_log_residual_sd: float | None
    extras: bool = False


def _v2_design(X: pd.DataFrame) -> pd.DataFrame:
    """The v2 model design: ``fixed_design`` over ``CATS_V2``."""
    return fixed_design(X, CATS_V2)


LEGACY_SPEC = FeatureSpec(FeatureSet.LEGACY, CATS, add_features, design, deal_value_design,
                          BASELINE_DEAL_LOG_RESIDUAL_SD)
V2_SPEC = FeatureSpec(FeatureSet.V2, CATS_V2, add_features_v2, _v2_design, fixed_deal_value_design, None)
GENERIC_SPEC = FeatureSpec(FeatureSet.GENERIC, CATS_V2, add_features_v2, _v2_design, fixed_deal_value_design, None,
                           extras=True)
_SPECS: dict[FeatureSet, FeatureSpec] = {s.feature_set: s for s in (LEGACY_SPEC, V2_SPEC, GENERIC_SPEC)}


def feature_spec(feature_set: FeatureSet | str) -> FeatureSpec:
    """The ``FeatureSpec`` for ``feature_set`` (a ``FeatureSet`` or its string value; ValueError otherwise)."""
    return _SPECS[FeatureSet(feature_set)]
