"""Turn context-agent output into model features (plan 6.3).

Two input formats, told apart by their columns (``context_features``):

- **Judgments** (context agent v2, ``emva.context.contract.OUTPUT_COLUMNS``): each judgment becomes a
  bucketed categorical with the declared levels ``CONTEXT_LEVELS`` and reference level ``CONTEXT_CATS``
  (fixed schema, ADR 0009: one column per non-reference level, whatever the data holds; an unknown value
  raises). ``is_real_business``, ``problem_specificity``, ``urgency`` and ``brief_fit`` map one to one to
  ``ctx_<judgment>``. ``persona`` is pooled (ruling R13, ADR 0016) into ``ctx_persona_group`` with levels
  buyer / non_buyer / unclear via ``PERSONA_GROUPS``: the contract keeps its eight values, the model gets
  three. The pooling rule was chosen after seeing the per-level results of the first run (Phase 6) and is
  now fixed here, so every later run is pre-registered. Rows whose status is not ``ok``, and leads with no row, get NaN in
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
PERSONA_GROUP_LEVELS: tuple[str, ...] = ("buyer", "non_buyer", "unclear")
# Every contract persona -> exactly one group (R13). A new contract value must be added here or import fails.
PERSONA_GROUPS: dict[str, str] = {"buyer": "buyer", "unclear": "unclear",
                                  **dict.fromkeys(("vendor", "student", "job_seeker", "competitor", "nonprofit",
                                                   "agency_pitching"), "non_buyer")}
if set(PERSONA_GROUPS) != set(JUDGMENTS["persona"]) or not set(PERSONA_GROUPS.values()) <= set(PERSONA_GROUP_LEVELS):
    raise RuntimeError("PERSONA_GROUPS must map every contract persona to one of PERSONA_GROUP_LEVELS")

# feature -> (levels, reference), in design order: the judgments' order with persona replaced by its group
CONTEXT_LEVELS: dict[str, tuple[str, ...]] = {
    PREFIX + ("persona_group" if k == "persona" else k): (PERSONA_GROUP_LEVELS if k == "persona" else v)
    for k, v in JUDGMENTS.items()}
CONTEXT_CATS: dict[str, str] = {"ctx_is_real_business": "yes", "ctx_persona_group": "buyer",
                                "ctx_problem_specificity": "specific", "ctx_urgency": "none", "ctx_brief_fit": "partial"}
CONTEXT_COLUMNS: list[str] = fixed_columns(CONTEXT_CATS, CONTEXT_LEVELS)
PERSONA_ACCEPTANCE_COLUMN = "ctx_persona_group=non_buyer"   # the weight judged under R12 (ADR 0016)


def persona_group(persona: pd.Series) -> pd.Series:
    """Map contract persona values to ``PERSONA_GROUP_LEVELS``; ``ValueError`` for any value not in the contract."""
    unknown = sorted(set(persona.astype(str)) - set(PERSONA_GROUPS))
    if unknown:
        raise ValueError(f"persona values outside the contract {list(PERSONA_GROUPS)}: {unknown[:5]}")
    return persona.astype(str).map(PERSONA_GROUPS)


def judgment_frame(ok: pd.DataFrame, pool_persona: bool = True) -> pd.DataFrame:
    """The ``CONTEXT_CATS`` feature columns for ok judgment rows (indexed by lead): judgments as is, persona grouped.

    ``pool_persona=False`` keeps the eight contract values in a ``ctx_persona`` column instead (the harness's
    secondary, unpooled weights).
    """
    if not pool_persona:
        return pd.DataFrame({PREFIX + k: ok[k] for k in JUDGMENTS}, index=ok.index)
    return pd.DataFrame({f: persona_group(ok.persona) if f == "ctx_persona_group" else ok[f.removeprefix(PREFIX)]
                         for f in CONTEXT_CATS}, index=ok.index)


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
    duplicated ``lead_id``, mixed stamps, or a judgment value outside its declared levels (persona:
    outside the contract).
    """
    if J.lead_id.duplicated().any():
        raise ValueError(f"duplicated lead_id in context file: {J.lead_id[J.lead_id.duplicated()].tolist()[:5]}")
    stamp(J)
    ok = J[J.status == STATUS_OK].set_index("lead_id")
    return fixed_design(judgment_frame(ok), CONTEXT_CATS, CONTEXT_LEVELS).reindex(index)


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
