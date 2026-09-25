"""Dummy-coded design matrix (``baseline/emva_score.py::design``) and exact-alias pruning (v2 only)."""
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


def drop_aliased_columns(D: pd.DataFrame, rows: pd.Series) -> tuple[pd.DataFrame, dict[str, str]]:
    """Drop every column that is identical, on ``rows``, to an earlier column of ``D``.

    Returns the pruned matrix and ``{dropped column: the earlier column it equals}``. Identical
    columns cannot be told apart by any fit on those rows (the L2 penalty would just split one
    weight equally between them), so the earlier column carries the shared effect. On v1 this is
    what happens to the behavioural ``missing`` levels (present exactly for Meta lead-ads leads,
    so equal to ``channel=meta_leadads``) and to ``spend``/``crm``/``hiring=missing`` (equal to
    ``enrichment_missing=yes`` by construction).
    """
    sub = D.loc[rows]
    first: dict[bytes, str] = {}
    aliases: dict[str, str] = {}
    for col in D.columns:
        key = sub[col].to_numpy().tobytes()
        if key in first:
            aliases[col] = first[key]
        else:
            first[key] = col
    return D.drop(columns=list(aliases)), aliases


def feature_design(X: pd.DataFrame, feature_set: FeatureSet, rows: pd.Series) -> tuple[pd.DataFrame, dict[str, str]]:
    """The model design for ``feature_set`` and its dropped aliases.

    Legacy: ``design(X)`` exactly as the baseline, no pruning (so its collinear pair stays).
    v2: ``design(X, cats=CATS_V2)`` with columns identical on ``rows`` (the training rows) pruned.
    """
    if FeatureSet(feature_set) is FeatureSet.LEGACY:
        return design(X), {}
    return drop_aliased_columns(design(X, cats=feature_cats(feature_set)), rows)
