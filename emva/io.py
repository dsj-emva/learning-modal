"""Ingest: load the CSVs, normalise CRM stages, join enrichment, drop bots and repeats.

``load`` (= ``read_leads`` + ``add_crm_outcomes`` + ``enrich``) is ``baseline/emva_score.py::load`` plus the two CRM timestamps the Phase 1 labels
need (``won_at``, ``first_contact_at``) and the Phase 2 company-name enrichment (``en_<column>``,
``enrichment_source``); ``clean`` is the cleaning block at the top of
``baseline/emva_score.py::build``. Neither changes any column the baseline uses: the legacy
feature set reads the domain-only ``co_<column>`` join, exactly as the baseline. ``enrich`` is the part
that needs no CRM history, shared with single-lead scoring (``emva.scoring``).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from emva.constants import ANSWER_KEYS, BOT_MIN_TIME_ON_PAGE_S, ENRICHMENT_COLUMNS, LEGAL_SUFFIXES, STAGE


def normalise_company_name(name: object) -> str | None:
    """Normalise a company name for exact matching (plan 2.3); None when nothing is left.

    Lower-case; drop dots and apostrophes (so "B.V." becomes "bv"); turn any other
    non-alphanumeric character into a space; collapse whitespace; then strip trailing
    ``LEGAL_SUFFIXES`` tokens repeatedly ("Acme Holdings Co Ltd" becomes "acme holdings").
    A name that is only a suffix ("Ltd") normalises to None. Case folding is ``str.lower()`` only, with
    no unicode normalisation or accent folding, so "Müller GmbH" and "Muller GmbH" do not match: this
    fails safe (a missed match, never a wrong one).
    """
    if not isinstance(name, str):
        return None
    s = re.sub(r"[.'’]", "", name.lower())
    tokens = re.sub(r"[^\w]+|_", " ", s).split()
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens) or None


def company_name_index(companies: pd.DataFrame) -> pd.Series:
    """Map normalised company name -> row label of ``companies``, keeping only unambiguous names.

    Names that normalise to the same key for two or more companies are left out: an exact match
    to one of them could be the wrong company.
    """
    keys = companies.company_name.map(normalise_company_name)
    counts = keys.value_counts()
    unique = keys[keys.notna() & keys.map(counts).eq(1)]
    return pd.Series(unique.index, index=unique.values)


def match_company_names(names: list[pd.Series], companies: pd.DataFrame) -> pd.Series:
    """Row label of ``companies`` matched by exact normalised name, trying each series of ``names`` in turn.

    ``names`` are aligned series of typed company names (e.g. the ``company_name`` column, then the
    ``company`` form answer); the first one whose normalised value is in ``companies`` wins. No fuzzy
    matching: a typo or a missing word means no match. NaN where nothing matches.
    """
    index = company_name_index(companies)
    out = pd.Series(pd.NA, index=names[0].index, dtype="object")
    for s in names:
        hit = s.map(normalise_company_name).map(index)
        out = out.fillna(hit)
    return out


def read_leads(path: str | Path) -> pd.DataFrame:
    """Read a ``historical_leads.csv`` file: indexed by ``lead_id``, ``created_at`` parsed as UTC timestamps."""
    return pd.read_csv(path, parse_dates=["created_at"]).set_index("lead_id")


def read_companies(data: str | Path) -> pd.DataFrame:
    """Read ``companies.csv`` from the dataset directory ``data``.

    Raises ``ValueError`` if it has no ``company_name`` column (needed for name enrichment).
    """
    CO = pd.read_csv(f"{data}/companies.csv")
    if "company_name" not in CO:
        raise ValueError(f"{data}/companies.csv has no company_name column (needed for name enrichment)")
    return CO


def add_crm_outcomes(L: pd.DataFrame, C: pd.DataFrame) -> pd.DataFrame:
    """Add the CRM columns of ``load`` (``final_stage``, ``last_change``, ``deal_value``, ``won_at``,
    ``first_contact_at``) to ``L`` in place from the raw ``crm_history.csv`` frame ``C``; returns ``L``.

    Raises ``ValueError`` if a lead has more than one Won row.
    """
    C = C.copy()
    C["stage_n"] = C.stage.str.lower().map(STAGE)
    C = C.sort_values(["lead_id", "changed_at"])
    multi_won = C[C.stage_n == "Won"].lead_id.duplicated()
    if multi_won.any():
        raise ValueError(f"leads with more than one Won row: {sorted(C[C.stage_n == 'Won'].lead_id[multi_won].unique())[:5]}")
    last = C.groupby("lead_id").tail(1).set_index("lead_id")
    L["final_stage"] = last.stage_n
    L["last_change"] = last.changed_at
    won = C[C.stage_n == "Won"].set_index("lead_id")
    L["deal_value"] = won.deal_value
    L["won_at"] = won.changed_at
    L["first_contact_at"] = C[C.stage_n.notna() & (C.stage_n != "New")].groupby("lead_id").changed_at.min()
    return L


def enrich(L: pd.DataFrame, CO: pd.DataFrame) -> pd.DataFrame:
    """Add the submit-time join columns of ``load`` to ``L`` in place and return it; ``CO`` is not modified.

    Needs only what exists when a lead is submitted (its ``historical_leads.csv`` row and ``companies.csv``),
    no CRM history, so training (``load``) and single-lead scoring (``emva.scoring``) share it. Adds
    ``a_<answer>`` for each of ``ANSWER_KEYS``, ``dom``, ``co_<column>``, ``enrichment_source`` and
    ``en_<column>`` (definitions in ``load``).
    """
    ans = L.answers.apply(json.loads)
    for k in ANSWER_KEYS:
        L["a_" + k] = ans.apply(lambda d: d.get(k))
    CO = CO.assign(dom=CO.domain.str.lower())
    L["dom"] = L.company_domain.str.lower().str.strip()
    L = L.join(CO.set_index("dom")[list(ENRICHMENT_COLUMNS)].add_prefix("co_"), on="dom")
    return _add_name_enrichment(L, CO)


def load(data: str | Path) -> pd.DataFrame:
    """Load leads indexed by ``lead_id`` with final CRM stage, deal value, answers and enrichment.

    ``read_leads`` + ``add_crm_outcomes`` + ``enrich``. Adds ``final_stage``, ``last_change``,
    ``deal_value``, ``won_at`` (time of the Won stage change, NaT if never Won), ``first_contact_at``
    (time of the first change to a recognised stage other than New, NaT if the lead never left New),
    ``a_<answer>`` for each of ``ANSWER_KEYS``, ``dom`` (normalised company domain), ``co_<column>``
    for each of ``ENRICHMENT_COLUMNS`` (domain join only, NaN when the domain is not in
    companies.csv; this is what the baseline uses), ``enrichment_source`` (``"domain"``, ``"name"``
    when the domain misses but the typed ``company_name`` (if the column exists) or ``company``
    answer matches a companies.csv name exactly after ``normalise_company_name``, NaN otherwise)
    and ``en_<column>`` (the domain row, else the name-matched row).

    Raises ``ValueError`` if a lead has more than one Won row (``deal_value`` and ``won_at``
    would be ambiguous) or companies.csv has no ``company_name`` column.
    """
    L = read_leads(f"{data}/historical_leads.csv")
    C = pd.read_csv(f"{data}/crm_history.csv", parse_dates=["changed_at"])
    CO = read_companies(data)
    return enrich(add_crm_outcomes(L, C), CO)


def _add_name_enrichment(L: pd.DataFrame, CO: pd.DataFrame) -> pd.DataFrame:
    """Add ``enrichment_source`` and ``en_<column>`` (see ``load``) to ``L`` and return it."""
    by_domain = L.co_employee_band.notna()
    typed = [L[c] for c in ("company_name", "a_company") if c in L]  # historical_leads.company_name is optional
    name_row = match_company_names(typed, CO).where(~by_domain)
    by_name = name_row.notna()
    L["enrichment_source"] = pd.Series(pd.NA, index=L.index, dtype="object").mask(by_domain, "domain").mask(by_name, "name")
    for c in ENRICHMENT_COLUMNS:
        named = pd.Series(CO[c].reindex(name_row[by_name]).values, index=name_row.index[by_name])
        L["en_" + c] = L["co_" + c].where(by_domain, named.reindex(L.index))
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
