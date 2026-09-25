"""Dummy-coded design matrix (``baseline/emva_score.py::design``)."""
from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from emva.constants import CATS
from emva.features import FeatureSet, feature_cats


def design(X: pd.DataFrame, extra: pd.Series | pd.DataFrame | None = None,
           cats: Mapping[str, str] = CATS) -> pd.DataFrame:
    """One-hot encode every feature in ``cats`` (default: the legacy set), dropping each feature's reference level.

    Column names are ``<feature>=<level>``. Levels absent from ``X`` produce no column,
    and a reference level absent from ``X`` is silently not dropped (baseline behaviour).
    ``extra`` columns are left-joined on the index after the dummies.
    """
    parts = []
    for c, ref in cats.items():
        d = pd.get_dummies(X[c].astype(str), prefix=c, prefix_sep="=")
        parts.append(d.drop(columns=[f"{c}={ref}"], errors="ignore"))
    D = pd.concat(parts, axis=1).astype(float)
    if extra is not None:
        D = D.join(extra)
    return D


def feature_design(X: pd.DataFrame, feature_set: FeatureSet) -> pd.DataFrame:
    """The model design for ``feature_set``: ``design(X)`` for legacy (exactly the baseline), ``CATS_V2`` for v2."""
    return design(X, cats=feature_cats(feature_set))
