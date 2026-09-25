"""Bucketed model features (the "features" block of ``baseline/emva_score.py::build``).

No behaviour change in Phase 0: missing values still fall into reference levels
(for example a missing ``time_on_page`` becomes "15-60s"). Phase 2 adds explicit
``missing`` levels.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from emva.constants import (
    COPY_PASTE_PREFIXES,
    FREE,
    MID_TITLE_PATTERN,
    SENIOR_TITLE_PATTERN,
    SPECIFIC_TEXT_PATTERN,
    VAGUE,
)


def text_cat(s: str | None) -> str:
    """Classify the free-text answer as ``copy_paste``, ``vague``, ``specific`` or ``neutral``."""
    s = (s or "").strip().lower()
    if s.startswith(COPY_PASTE_PREFIXES):
        return "copy_paste"
    if s in VAGUE:
        return "vague"
    if re.search(SPECIFIC_TEXT_PATTERN, s):
        return "specific"
    return "neutral"


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
    med, src = utm_medium, utm_source
    return np.select(
        [med == "lead_form", src.isin(["facebook", "instagram"]), src == "google", src == "linkedin", src == "chatgpt"],
        ["meta_leadads", "meta", "google", "linkedin", "chatgpt"],
        "organic_direct",
    )


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add every bucketed feature column to ``X`` in place and return it.

    Needs the columns produced by ``emva.io.load`` and ``emva.io.clean`` (``email_l``).
    Note: ``viewed_pricing`` and ``ip_country`` are overwritten with their bucketed values,
    exactly as the baseline does.
    """
    X["channel"] = channel(X.utm_medium, X.utm_source)
    lead_ads = X.channel == "meta_leadads"
    has_co = X.co_employee_band.notna()
    X["no_company"] = np.where(has_co, "no", "yes")
    X["band"] = X.co_employee_band.fillna(X.a_company_size.replace("", np.nan)).fillna("1-10")
    X["email"] = np.where(X.email_l.str.split("@").str[1].isin(FREE), "free", "business")
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
    X["search_term"] = np.where(X.channel != "google", "not_google",
                                np.where(X.utm_term.fillna("").str.lower().str.contains("northstar"), "brand", "generic"))
    wk = ~X.submitted_weekday.isin(["Saturday", "Sunday"]) & X.local_submit_hour.between(9, 17)
    X["business_hours"] = np.where(wk & ~lead_ads, "wkday_9-18", "outside")
    X["ip_country"] = np.where(X.ip_country.notna() & (X.ip_country != X.a_country), "mismatch", "match")
    X["ip_type"] = np.where(X.is_datacenter_ip == True, "dc", "residential")  # noqa: E712
    X["c_budget"] = X.a_budget.fillna("not_asked")
    X["c_timeline"] = X.a_timeline.fillna("not_asked")
    X["edits_1_4"] = np.where(X.field_edit_count.between(1, 4), "yes", "no")
    X["sector"] = X.co_sector.fillna("none")
    return X
