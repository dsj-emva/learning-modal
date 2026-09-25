"""Turn context-agent output into model features (Phase 6 replaces this with per-judgment buckets).

Current behaviour, from ``baseline/emva_score.py::main``: the single 0-1 ``context_score`` is
clipped to ``CONTEXT_SCORE_CLIP`` and converted to a logit, giving one numeric column
``context_logit`` that is appended to the formula's design matrix.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from emva.constants import CONTEXT_SCORE_CLIP


def context_logit(path: str | Path, index: pd.Index) -> pd.Series:
    """Read an agent output CSV and return ``context_logit`` reindexed to ``index`` (NaN = no score)."""
    cs = pd.read_csv(path).set_index("lead_id")["context_score"].clip(*CONTEXT_SCORE_CLIP)
    return np.log(cs / (1 - cs)).reindex(index).rename("context_logit")
