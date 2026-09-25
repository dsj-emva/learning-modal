"""Turn context-agent output into model features (plan 6.3).

Two input formats, told apart by their columns (``context_features``):

- **Judgments** (context agent v2, ``emva.context.contract.OUTPUT_COLUMNS``): each judgment becomes a
  bucketed categorical ``ctx_<judgment>`` with the declared levels ``CONTEXT_LEVELS`` and reference
  level ``CONTEXT_CATS`` (fixed schema, ADR 0009: one column per non-reference level, whatever the data
  holds; an unknown value raises). Rows whose status is not ``ok``, and leads with no row, get NaN in
  every column (no context). A file must carry a single ``(brief_hash, prompt_version, model_id)``
  stamp: judgments from two briefs, prompts or models cannot share one set of weights.
- **Legacy score** (the baseline's ``lead_id,context_score``): one numeric ``context_logit`` column, the
  0-1 score clipped to ``CONTEXT_SCORE_CLIP`` and converted to a logit (``baseline/emva_score.py::main``).

Reference levels (weights read relative to them): a real business (``yes``), a ``buyer``, a ``specific``
problem, no timing signal (``none``) and a ``partial`` brief fit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from emva.constants import CONTEXT_SCORE_CLIP
from emva.context.contract import JUDGMENTS, STAMP_COLUMNS, STATUS_OK
from emva.design import fixed_columns, fixed_design

PREFIX = "ctx_"
CONTEXT_REFERENCE: dict[str, str] = {"is_real_business": "yes", "persona": "buyer", "problem_specificity": "specific",
                                     "urgency": "none", "brief_fit": "partial"}
CONTEXT_CATS: dict[str, str] = {PREFIX + k: CONTEXT_REFERENCE[k] for k in JUDGMENTS}
CONTEXT_LEVELS: dict[str, tuple[str, ...]] = {PREFIX + k: v for k, v in JUDGMENTS.items()}
CONTEXT_COLUMNS: list[str] = fixed_columns(CONTEXT_CATS, CONTEXT_LEVELS)


def context_logit(path: str | Path, index: pd.Index) -> pd.Series:
    """Read a legacy agent CSV and return ``context_logit`` reindexed to ``index`` (NaN = no score)."""
    cs = pd.read_csv(path).set_index("lead_id")["context_score"].clip(*CONTEXT_SCORE_CLIP)
    return np.log(cs / (1 - cs)).reindex(index).rename("context_logit")


def stamp(J: pd.DataFrame) -> tuple[str, str, str]:
    """The single ``(brief_hash, prompt_version, model_id)`` of a judgments frame; ``ValueError`` if there are several."""
    stamps = J[list(STAMP_COLUMNS)].drop_duplicates()
    if len(stamps) != 1:
        raise ValueError(f"context file mixes {len(stamps)} (brief_hash, prompt_version, model_id) stamps; "
                         "judgments from different briefs, prompts or models need separate runs and refits")
    return tuple(stamps.iloc[0])  # type: ignore[return-value]


def context_design(J: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """Fixed-schema dummies (``CONTEXT_COLUMNS``) for a judgments frame, reindexed to ``index``.

    Rows with ``status != "ok"`` and leads absent from ``J`` are all-NaN. Raises ``ValueError`` for a
    duplicated ``lead_id``, mixed stamps, or a judgment value outside its declared levels.
    """
    if J.lead_id.duplicated().any():
        raise ValueError(f"duplicated lead_id in context file: {J.lead_id[J.lead_id.duplicated()].tolist()[:5]}")
    stamp(J)
    ok = J[J.status == STATUS_OK].set_index("lead_id")
    F = pd.DataFrame({PREFIX + k: ok[k] for k in JUDGMENTS}, index=ok.index)
    return fixed_design(F, CONTEXT_CATS, CONTEXT_LEVELS).reindex(index)


def context_features(path: str | Path, index: pd.Index) -> pd.DataFrame:
    """Context feature columns for ``index`` from an agent CSV in either format (see the module docstring).

    Returns ``context_logit`` (legacy score file) or ``CONTEXT_COLUMNS`` (judgments file); a lead has
    context when all its columns are non-NaN. Raises ``ValueError`` for a file in neither format.
    """
    J = pd.read_csv(path)
    if "context_score" in J:
        return context_logit(path, index).to_frame()
    if {"lead_id", "status", *JUDGMENTS, *STAMP_COLUMNS} <= set(J):
        return context_design(J, index)
    raise ValueError(f"{path}: neither a legacy context_score file nor a judgments file (columns {list(J.columns)})")
