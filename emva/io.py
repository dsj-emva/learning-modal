"""Ingest: load the CSVs, normalise CRM stages, join enrichment, drop bots and repeats.

``load`` is ``baseline/emva_score.py::load`` and ``clean`` is the cleaning block at the top
of ``baseline/emva_score.py::build``, both unchanged in behaviour.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from emva.constants import ANSWER_KEYS, BOT_MIN_TIME_ON_PAGE_S, ENRICHMENT_COLUMNS, STAGE


def load(data: str | Path) -> pd.DataFrame:
    """Load leads indexed by ``lead_id`` with final CRM stage, deal value, answers and enrichment.

    Adds ``final_stage``, ``last_change``, ``deal_value``, ``a_<answer>`` for each of
    ``ANSWER_KEYS``, ``dom`` (normalised company domain) and ``co_<column>`` for each of
    ``ENRICHMENT_COLUMNS`` (NaN when the domain is not in companies.csv).
    """
    L = pd.read_csv(f"{data}/historical_leads.csv", parse_dates=["created_at"]).set_index("lead_id")
    C = pd.read_csv(f"{data}/crm_history.csv", parse_dates=["changed_at"])
    CO = pd.read_csv(f"{data}/companies.csv")
    C["stage_n"] = C.stage.str.lower().map(STAGE)
    C = C.sort_values(["lead_id", "changed_at"])
    last = C.groupby("lead_id").tail(1).set_index("lead_id")
    L["final_stage"] = last.stage_n
    L["last_change"] = last.changed_at
    L["deal_value"] = C[C.stage_n == "Won"].set_index("lead_id").deal_value
    ans = L.answers.apply(json.loads)
    for k in ANSWER_KEYS:
        L["a_" + k] = ans.apply(lambda d: d.get(k))
    CO["dom"] = CO.domain.str.lower()
    L["dom"] = L.company_domain.str.lower().str.strip()
    L = L.join(CO.set_index("dom")[list(ENRICHMENT_COLUMNS)].add_prefix("co_"), on="dom")
    return L


def flag_bots_and_duplicates(L: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``L`` with ``email_l``, ``bot`` and ``dup`` columns added.

    ``bot``: headless user agent or under ``BOT_MIN_TIME_ON_PAGE_S`` seconds on the page.
    ``dup``: not the first submission (by ``created_at``) for its normalised email.
    """
    X = L.copy()
    X["email_l"] = X.email.str.strip().str.lower()
    X["bot"] = X.user_agent.fillna("").str.contains("Headless", case=False) | (X.time_on_page_s < BOT_MIN_TIME_ON_PAGE_S)
    X["dup"] = X.groupby("email_l").created_at.rank(method="first") > 1
    return X


def clean(L: pd.DataFrame) -> pd.DataFrame:
    """Drop bots and repeat submissions (keeps the first submission per email)."""
    X = flag_bots_and_duplicates(L)
    return X[~X.bot & ~X.dup].copy()
