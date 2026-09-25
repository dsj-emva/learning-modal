"""Dummy-coded design matrix (``baseline/emva_score.py::design``, unchanged)."""
from __future__ import annotations

import pandas as pd

from emva.constants import CATS


def design(X: pd.DataFrame, extra: pd.Series | pd.DataFrame | None = None) -> pd.DataFrame:
    """One-hot encode every feature in ``CATS``, dropping each feature's reference level.

    Column names are ``<feature>=<level>``. Levels absent from ``X`` produce no column,
    and a reference level absent from ``X`` is silently not dropped (baseline behaviour).
    ``extra`` columns are left-joined on the index after the dummies.
    """
    parts = []
    for c, ref in CATS.items():
        d = pd.get_dummies(X[c].astype(str), prefix=c, prefix_sep="=")
        parts.append(d.drop(columns=[f"{c}={ref}"], errors="ignore"))
    D = pd.concat(parts, axis=1).astype(float)
    if extra is not None:
        D = D.join(extra)
    return D
