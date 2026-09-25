"""Bucketed model features: the legacy set (the baseline's) and the v2 set (plan Phase 2).

``--feature-set legacy`` (``add_features``) is the "features" block of
``baseline/emva_score.py::build`` unchanged: missing values fall into reference levels (for
example a missing ``time_on_page`` becomes "15-60s") and enrichment comes from the email-domain
join only.

``--feature-set v2`` (``add_features_v2``, the default):

Absent inputs use an explicit indicator encoding (orchestrator ruling on Phase 2): the design's
column set is fixed by this spec, never by the data.

- 2.1: ``session_missing`` = yes when the on-site session is absent (``session_absent``: no
  landing page visit, or any of the behavioural inputs blank). Then the seven
  ``BEHAVIOURAL_FEATURES`` take their reference level and the indicator carries the effect.
  ``business_hours`` has no channel special case; Meta lead-ads leads (no session) and
  consent-declined website leads (v2 data) both get ``session_missing=yes``.
- 2.2: ``enrichment_missing`` (no companies.csv row by domain or by name) replaces
  ``no_company``; ``spend``, ``crm``, ``hiring`` take their reference level without enrichment and
  the indicator carries the effect. ``band`` is enrichment, else the typed ``company_size``
  answer, else its own ``missing`` level (distinct from the indicator because of the typed
  fallback). ``sector`` (deal-value model only) is ``missing`` without enrichment.
- 2.3: enrichment is ``en_<column>`` from ``emva.io.load``: the domain row, else an exact
  normalised company-name match.
- 2.5: ``text`` detects boilerplate by similarity (``emva.boilerplate``), not by prefix.
- 2.6: the features in ``V2_DROPPED`` are computed but not used by the model.
"""
from __future__ import annotations

import re
from enum import Enum

import numpy as np
import pandas as pd

from emva.boilerplate import is_boilerplate
from emva.constants import (
    CATS_V2_CANDIDATES,
    COPY_PASTE_PREFIXES,
    FREE,
    MID_TITLE_PATTERN,
    MISSING,
    SENIOR_TITLE_PATTERN,
    SPECIFIC_TEXT_PATTERN,
    VAGUE,
    WEEKDAYS,
)


class FeatureSet(str, Enum):
    """``--feature-set``: ``legacy`` = the baseline's features, ``v2`` = plan Phase 2 (default)."""

    LEGACY = "legacy"
    V2 = "v2"


# Plan 2.6: v2 candidates dropped because no level's 95% coefficient-bootstrap CI excludes zero on v1
# (python -m emva.eval.feature_selection, 1000 refits, horizon labels; same decision with legacy labels).
# Evidence: reports/phase2.md, section "Plan 2.6: feature selection by coefficient bootstrap". Re-run the
# evaluation on new data.
V2_DROPPED: tuple[str, ...] = ("c_budget", "c_timeline", "ip_type", "edits_1_4")
CATS_V2: dict[str, str] = {k: v for k, v in CATS_V2_CANDIDATES.items() if k not in V2_DROPPED}

# ``business_hours`` counts these ``submitted_weekday`` values as outside business hours.
WEEKEND: tuple[str, ...] = WEEKDAYS[5:]

# Behavioural (on-site session) features whose absence session_missing carries in v2 (plan 2.1).
BEHAVIOURAL_FEATURES: tuple[str, ...] = ("time_on_page", "hesitation_90s", "sessions_3plus", "viewed_pricing",
                                         "ip_country", "ip_type", "business_hours")
# Raw inputs of those features; any of them blank (or no landing_url) means the session is absent.
SESSION_INPUTS: tuple[str, ...] = ("time_on_page_s", "hesitation_ms", "sessions_before_convert", "viewed_pricing",
                                   "ip_country", "is_datacenter_ip", "submitted_weekday", "local_submit_hour")


# ``channel`` rules: this utm_medium means Meta lead ads; otherwise channel <- utm_source values, first match wins.
LEAD_ADS_MEDIUM: str = "lead_form"
CHANNEL_SOURCES: dict[str, tuple[str, ...]] = {"meta": ("facebook", "instagram"), "google": ("google",),
                                               "linkedin": ("linkedin",), "chatgpt": ("chatgpt",)}


def _normalise_text(s: object) -> str:
    """Stripped, lower-cased text; ``""`` for anything that is not a string (None, NaN, ``pd.NA``)."""
    return s.strip().lower() if isinstance(s, str) else ""


def _vague_specific_or_neutral(s: str) -> str:
    """The rules after the copy-paste test, shared by both text classifiers; ``s`` is normalised."""
    if s in VAGUE:
        return "vague"
    if re.search(SPECIFIC_TEXT_PATTERN, s):
        return "specific"
    return "neutral"


def text_cat(s: object) -> str:
    """Legacy: classify the free-text answer as ``copy_paste``, ``vague``, ``specific`` or ``neutral``."""
    s = _normalise_text(s)
    if s.startswith(COPY_PASTE_PREFIXES):
        return "copy_paste"
    return _vague_specific_or_neutral(s)


def text_cat_v2(s: object) -> str:
    """v2: like ``text_cat`` but ``copy_paste`` means similar to a boilerplate snippet (``emva.boilerplate``)."""
    if is_boilerplate(s):
        return "copy_paste"
    return _vague_specific_or_neutral(_normalise_text(s))


def seniority(t: object) -> str:
    """Bucket a job title into ``student``, ``senior``, ``mid``, ``junior/ic`` or ``not_asked/blank``.

    Rules are applied in that order, so "Student Director" is ``student``.
    """
    if not isinstance(t, str) or not t.strip():
        return "not_asked/blank"
    t = t.lower()
    if "student" in t or "intern" in t:
        return "student"
    if re.search(SENIOR_TITLE_PATTERN, t):
        return "senior"
    if re.search(MID_TITLE_PATTERN, t):
        return "mid"
    return "junior/ic"


def channel(utm_medium: pd.Series, utm_source: pd.Series) -> np.ndarray:
    """Map UTM medium/source to a channel; first matching rule wins.

    lead_form medium -> ``meta_leadads``; facebook/instagram -> ``meta``; google -> ``google``;
    linkedin -> ``linkedin``; chatgpt -> ``chatgpt``; anything else (incl. missing) -> ``organic_direct``.
    """
    conditions = [utm_medium == LEAD_ADS_MEDIUM] + [utm_source.isin(src) for src in CHANNEL_SOURCES.values()]
    return np.select(conditions, ["meta_leadads", *CHANNEL_SOURCES], "organic_direct")


def email_type(email_l: pd.Series) -> np.ndarray:
    """``free`` when the normalised email's domain is in ``FREE``, else ``business``."""
    return np.where(email_l.str.split("@").str[1].isin(FREE), "free", "business")


def search_term(ch: pd.Series, utm_term: pd.Series) -> np.ndarray:
    """``not_google`` off Google; on Google ``brand`` if the term mentions northstar, else ``generic``."""
    return np.where(ch != "google", "not_google",
                    np.where(utm_term.fillna("").str.lower().str.contains("northstar"), "brand", "generic"))


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Legacy feature set: add every bucketed feature column to ``X`` in place and return it.

    Needs the columns produced by ``emva.io.load`` and ``emva.io.clean`` (``email_l``).
    Note: ``viewed_pricing`` and ``ip_country`` are overwritten with their bucketed values,
    exactly as the baseline does.
    """
    X["channel"] = channel(X.utm_medium, X.utm_source)
    lead_ads = X.channel == "meta_leadads"
    has_co = X.co_employee_band.notna()
    X["no_company"] = np.where(has_co, "no", "yes")
    X["band"] = X.co_employee_band.fillna(X.a_company_size.replace("", np.nan)).fillna("1-10")
    X["email"] = email_type(X.email_l)
    X["text"] = X.a_what_to_solve.apply(text_cat)
    X["seniority"] = X.a_job_title.apply(seniority)
    X["spend"] = X.co_monthly_ad_spend_band.fillna("under £5k")
    X["crm"] = np.where(X.co_crm_platform.isin(["HubSpot", "Salesforce"]), "hubspot_sf", "other")
    X["hiring"] = np.where(X.co_is_hiring == True, "hiring", "not_hiring")  # noqa: E712 (NaN-safe on object dtype)
    top = X.time_on_page_s
    X["time_on_page"] = np.select([top < 60, top < 300, top < 600, top >= 600],
                                  ["15-60s", "60-300s", "300-600s", ">600s"], "15-60s")
    X["hesitation_90s"] = np.where(X.hesitation_ms > 90000, "yes", "no")
    X["sessions_3plus"] = np.where(X.sessions_before_convert >= 3, "yes", "no")
    X["viewed_pricing"] = np.where(X.viewed_pricing == True, "yes", "no")  # noqa: E712
    X["search_term"] = search_term(X.channel, X.utm_term)
    wk = ~X.submitted_weekday.isin(WEEKEND) & X.local_submit_hour.between(9, 17)
    X["business_hours"] = np.where(wk & ~lead_ads, "wkday_9-18", "outside")
    X["ip_country"] = np.where(X.ip_country.notna() & (X.ip_country != X.a_country), "mismatch", "match")
    X["ip_type"] = np.where(X.is_datacenter_ip == True, "dc", "residential")  # noqa: E712
    X["c_budget"] = X.a_budget.fillna("not_asked")
    X["c_timeline"] = X.a_timeline.fillna("not_asked")
    X["edits_1_4"] = np.where(X.field_edit_count.between(1, 4), "yes", "no")
    X["sector"] = X.co_sector.fillna("none")
    return X


def session_absent(X: pd.DataFrame) -> pd.Series:
    """True where the lead has no usable on-site session: blank ``landing_url`` or any ``SESSION_INPUTS`` blank.

    Meta lead-ads leads never visit the page; consent-declined website leads visit it but send no
    telemetry. Either way the behavioural features cannot be bucketed.
    """
    return X.landing_url.isna() | X[list(SESSION_INPUTS)].isna().any(axis=1)


def _or_reference(absent: pd.Series, feature: str, values: np.ndarray | pd.Series) -> np.ndarray:
    """``values`` with ``feature``'s v2 reference level wherever ``absent`` is True (the indicator carries it)."""
    return np.where(absent, CATS_V2_CANDIDATES[feature], values)


def add_features_v2(X: pd.DataFrame) -> pd.DataFrame:
    """v2 feature set (module docstring): add every bucketed feature column to ``X`` in place and return it.

    Needs the columns produced by ``emva.io.load`` (including ``en_<column>`` and
    ``enrichment_source``) and ``emva.io.clean``. ``viewed_pricing`` and ``ip_country`` are
    overwritten with their bucketed values (read from the raw columns first), as in the legacy set.
    ``ip_country`` is ``mismatch`` only when both the IP country and the typed country are present
    and differ.
    """
    X["channel"] = channel(X.utm_medium, X.utm_source)
    no_session = session_absent(X)
    X["session_missing"] = np.where(no_session, "yes", "no")
    no_enrichment = X.enrichment_source.isna()
    X["enrichment_missing"] = np.where(no_enrichment, "yes", "no")
    X["band"] = X.en_employee_band.fillna(X.a_company_size.replace("", np.nan)).fillna(MISSING)
    X["email"] = email_type(X.email_l)
    X["text"] = X.a_what_to_solve.apply(text_cat_v2)
    X["seniority"] = X.a_job_title.apply(seniority)
    X["spend"] = _or_reference(no_enrichment, "spend", X.en_monthly_ad_spend_band)
    X["crm"] = _or_reference(no_enrichment, "crm",
                             np.where(X.en_crm_platform.isin(["HubSpot", "Salesforce"]), "hubspot_sf", "other"))
    X["hiring"] = _or_reference(no_enrichment, "hiring",
                                np.where(X.en_is_hiring == True, "hiring", "not_hiring"))  # noqa: E712
    top = X.time_on_page_s
    X["time_on_page"] = _or_reference(no_session, "time_on_page", np.select(
        [top < 60, top < 300, top < 600], ["15-60s", "60-300s", "300-600s"], ">600s"))
    X["hesitation_90s"] = _or_reference(no_session, "hesitation_90s", np.where(X.hesitation_ms > 90000, "yes", "no"))
    X["sessions_3plus"] = _or_reference(no_session, "sessions_3plus",
                                        np.where(X.sessions_before_convert >= 3, "yes", "no"))
    X["viewed_pricing"] = _or_reference(no_session, "viewed_pricing",
                                        np.where(X.viewed_pricing == True, "yes", "no"))  # noqa: E712
    X["search_term"] = search_term(X.channel, X.utm_term)
    wk = ~X.submitted_weekday.isin(WEEKEND) & X.local_submit_hour.between(9, 17)
    X["business_hours"] = _or_reference(no_session, "business_hours", np.where(wk, "wkday_9-18", "outside"))
    X["ip_country"] = _or_reference(no_session, "ip_country",
                                    np.where(X.ip_country.notna() & X.a_country.notna()
                                             & (X.ip_country != X.a_country), "mismatch", "match"))
    X["ip_type"] = _or_reference(no_session, "ip_type",
                                 np.where(X.is_datacenter_ip == True, "dc", "residential"))  # noqa: E712
    X["c_budget"] = X.a_budget.fillna("not_asked")
    X["c_timeline"] = X.a_timeline.fillna("not_asked")
    X["edits_1_4"] = np.where(X.field_edit_count.between(1, 4), "yes", "no")
    X["sector"] = X.en_sector.fillna(MISSING)
    return X
