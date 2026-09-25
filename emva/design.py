"""Dummy-coded design matrix: data-driven for the legacy set (``baseline/emva_score.py::design``), fixed for v2."""
from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from emva.constants import CATS, V2_LEVELS


def design(X: pd.DataFrame, extra: pd.Series | pd.DataFrame | None = None,
           cats: Mapping[str, str] = CATS) -> pd.DataFrame:
    """Legacy (data-driven) design: one-hot encode every feature in ``cats``, dropping each reference level.

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


def fixed_columns(cats: Mapping[str, str | None], levels: Mapping[str, tuple[str, ...]] = V2_LEVELS) -> list[str]:
    """The design columns ``fixed_design`` emits: ``<feature>=<level>`` for every non-reference level, in spec order.

    A reference of None keeps every level (e.g. the ridge deal-value design). Raises ``ValueError`` if a feature
    has no level list or its (non-None) reference level is not in it.
    """
    cols = []
    for c, ref in cats.items():
        if c not in levels or (ref is not None and ref not in levels[c]):
            raise ValueError(f"feature {c!r}: no level list containing its reference level {ref!r}")
        cols += [f"{c}={lvl}" for lvl in levels[c] if lvl != ref]
    return cols


def fixed_design(X: pd.DataFrame, cats: Mapping[str, str | None],
                 levels: Mapping[str, tuple[str, ...]] = V2_LEVELS) -> pd.DataFrame:
    """Fixed-schema design: exactly ``fixed_columns(cats, levels)``, whatever levels occur in ``X``.

    A level absent from ``X`` gives an all-zero column, so one row gets the full column set. A value
    not in the feature's level list raises ``ValueError`` naming the feature and the values. A reference
    of None keeps every level of that feature.
    """
    cols = fixed_columns(cats, levels)
    data: dict[str, pd.Series] = {}
    for c, ref in cats.items():
        values = X[c].astype(str)
        unknown = sorted(set(values) - set(levels[c]))
        if unknown:
            raise ValueError(f"feature {c!r} has values outside its declared levels {list(levels[c])}: {unknown[:5]}")
        for lvl in levels[c]:
            if lvl != ref:
                data[f"{c}={lvl}"] = values.eq(lvl).astype(float)
    return pd.DataFrame(data, index=X.index, columns=cols)
