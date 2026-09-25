"""Label rules and the frozen train/test split.

Unchanged from the "labels (decision log)" block of ``baseline/emva_score.py::build``:
Won = 1; Lost = 0; open deals idle ``STALLED_DAYS``+ = 0; never-contacted (stage New)
leads ``GHOSTED_DAYS``+ old = 0; everything else unlabelled (NaN). Phase 1 replaces this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from emva.constants import AS_OF, GHOSTED_DAYS, OPEN_STAGES, STALLED_DAYS, TEST_FROM


def label(X: pd.DataFrame, as_of: pd.Timestamp = AS_OF) -> pd.Series:
    """Return the baseline label (1.0, 0.0 or NaN) for each row.

    Needs ``created_at``, ``last_change`` and ``final_stage``. Later rules override earlier
    ones in this order: Won, Lost, stalled, ghosted (the rules are disjoint on stage anyway).
    """
    age = (as_of - X.created_at).dt.days
    idle = (as_of - X.last_change).dt.days
    open_ = X.final_stage.isin(list(OPEN_STAGES))
    y = pd.Series(np.nan, index=X.index, name="y")
    y.loc[X.final_stage == "Won"] = 1
    y.loc[X.final_stage == "Lost"] = 0
    y.loc[open_ & (idle >= STALLED_DAYS)] = 0          # stalled 90+ days = Lost
    y.loc[(X.final_stage == "New") & (age >= GHOSTED_DAYS)] = 0  # ghosted 90+ days = Lost
    return y


def split_masks(X: pd.DataFrame, test_from: str = TEST_FROM) -> tuple[pd.Series, pd.Series]:
    """Return ``(train, test)`` boolean masks: labelled rows created before / on-or-after ``test_from``."""
    lab = X.y.notna()
    return lab & (X.created_at < test_from), lab & (X.created_at >= test_from)
