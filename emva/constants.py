"""Every constant the pipeline uses, in one place.

Phase 0 values are copied verbatim from ``baseline/emva_score.py``; Phase 1 added the label
horizon and label modes; Phase 2 added the v2 feature set (``CATS_V2_CANDIDATES``, the missing level,
company-name matching, the boilerplate threshold and the collinearity limit). Changing any of them
changes model behaviour, so each change belongs to a numbered plan task with its own report.
"""
from __future__ import annotations

import pandas as pd

# Snapshot date of the data and the frozen train/test boundary (ground rule 4).
AS_OF: pd.Timestamp = pd.Timestamp("2026-09-24", tz="UTC")
TEST_FROM: str = "2026-05-01"

# Label rules (baseline decision log): open deals idle this long, and never-contacted
# leads this old, are labelled Lost. STALLED_DAYS also defines ``label_source == "stalled"``.
STALLED_DAYS: int = 90
GHOSTED_DAYS: int = 90
OPEN_STAGES: tuple[str, ...] = ("Contacted", "Qualified", "Demo booked", "Proposal")

# Fixed-horizon label (plan 1.2): won_within_H with the same H for every lead. A lead is
# mature once created_at + H <= AS_OF. CLI: --horizon-days.
HORIZON_DAYS: int = 120
# The label modes themselves are emva.labels.LabelMode (default horizon).
# Every row gets exactly one label_source (plan 1.1); definitions in emva/labels.py.
LABEL_SOURCES: tuple[str, ...] = ("won", "crm_lost", "stalled", "ghosted", "open")

# Bot rule: headless user agent or less than this many seconds on the page.
BOT_MIN_TIME_ON_PAGE_S: int = 15

FREE: frozenset[str] = frozenset(
    {"gmail.example", "yahoo.example", "hotmail.example", "outlook.example", "icloud.example"}
)

# CRM stage spellings (lower-cased) -> canonical stage.
STAGE: dict[str, str] = {
    "closed won": "Won", "won": "Won", "closed lost": "Lost", "lost": "Lost", "qualified": "Qualified",
    "demo booked": "Demo booked", "proposal": "Proposal", "new": "New", "contacted": "Contacted",
}

# Free-text answers treated as vague (after strip + lower).
VAGUE: frozenset[str] = frozenset(
    {"", "?", "n/a", "hi", "test", "info", "interested", "tell me more", "more info pls", "pricing"}
)
COPY_PASTE_PREFIXES: tuple[str, ...] = (
    "lorem ipsum", "we are a leading provider", "our mission is to empower",
)
SPECIFIC_TEXT_PATTERN: str = (
    r"£\s?\d|\d+\s*(seats|users|sdrs|marketers)|team of \d|budget|within \d+ weeks|"
    r"by (jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)|before (q\d|end|the new)|contract ends"
)
SENIOR_TITLE_PATTERN: str = r"ceo|founder|chief|vp|director|head|owner|partner|president|cmo|cto|cfo"
MID_TITLE_PATTERN: str = r"manager|lead"

# ``submitted_weekday`` values, Monday first (the last two are the weekend for ``business_hours``).
WEEKDAYS: tuple[str, ...] = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
# ``business_hours`` counts these ``submitted_weekday`` values as outside business hours.
WEEKEND: tuple[str, ...] = WEEKDAYS[5:]

# ``channel`` rules: this utm_medium means Meta lead ads; otherwise channel <- utm_source values, first match wins.
LEAD_ADS_MEDIUM: str = "lead_form"
CHANNEL_SOURCES: dict[str, tuple[str, ...]] = {"meta": ("facebook", "instagram"), "google": ("google",),
                                               "linkedin": ("linkedin",), "chatgpt": ("chatgpt",)}

# Form answers extracted from the ``answers`` JSON into ``a_<key>`` columns.
ANSWER_KEYS: tuple[str, ...] = ("country", "what_to_solve", "job_title", "company_size", "budget", "timeline",
                                 "company")
# companies.csv columns joined onto leads as ``co_<column>``.
ENRICHMENT_COLUMNS: tuple[str, ...] = ("sector", "employee_band", "monthly_ad_spend_band", "crm_platform", "is_hiring")

# Legacy feature set (``--feature-set legacy``): feature -> reference level (weights are relative to
# this level). Order defines design columns. Missing inputs fall into reference levels (baseline behaviour).
CATS: dict[str, str] = {
    "channel": "google", "form_variant": "A", "band": "1-10", "email": "business", "text": "neutral",
    "seniority": "junior/ic", "spend": "under £5k", "crm": "other", "hiring": "not_hiring",
    "time_on_page": "15-60s", "hesitation_90s": "no", "sessions_3plus": "no", "viewed_pricing": "no",
    "search_term": "generic", "business_hours": "outside", "ip_country": "match", "ip_type": "residential",
    "c_budget": "not_asked", "c_timeline": "not_asked", "edits_1_4": "no", "no_company": "no",
}

# Level of ``band`` when neither enrichment nor the typed company size is available (v2, plan 2.2), and of
# ``sector`` in the deal-value model. Other absent inputs are carried by the two indicators
# ``session_missing`` and ``enrichment_missing`` (orchestrator ruling on Phase 2, reports/phase2.md).
MISSING: str = "missing"

# v2 feature set before the plan 2.6 selection (``emva.features.V2_DROPPED`` removes some of them).
# Order defines the design columns. The column set is fixed by this spec: nothing is dropped per dataset.
CATS_V2_CANDIDATES: dict[str, str] = {
    "channel": "google", "form_variant": "A", "session_missing": "no", "enrichment_missing": "no", "band": "1-10",
    "email": "business",
    "text": "neutral", "seniority": "junior/ic", "spend": "under £5k", "crm": "other", "hiring": "not_hiring",
    "time_on_page": "15-60s", "hesitation_90s": "no", "sessions_3plus": "no", "viewed_pricing": "no",
    "search_term": "generic", "business_hours": "outside", "ip_country": "match", "ip_type": "residential",
    "c_budget": "not_asked", "c_timeline": "not_asked", "edits_1_4": "no",
}

# Complete level list of every v2 candidate feature, reference level first. The v2 design emits exactly one
# column per non-reference level listed here, whatever occurs in the data (absent level = all-zero column),
# and refuses a value not listed (ADR 0009). Budget and timeline levels are the form's answer options.
V2_LEVELS: dict[str, tuple[str, ...]] = {
    "channel": ("google", "meta", "meta_leadads", "linkedin", "chatgpt", "organic_direct"),
    "form_variant": ("A", "B", "C", "D"),
    "session_missing": ("no", "yes"),
    "enrichment_missing": ("no", "yes"),
    "band": ("1-10", "11-50", "51-200", "201-1000", "1000+", MISSING),
    "email": ("business", "free"),
    "text": ("neutral", "copy_paste", "vague", "specific"),
    "seniority": ("junior/ic", "student", "senior", "mid", "not_asked/blank"),
    "spend": ("under £5k", "none", "£5k-£25k", "£25k-£100k", "£100k+"),
    "crm": ("other", "hubspot_sf"),
    "hiring": ("not_hiring", "hiring"),
    "time_on_page": ("15-60s", "60-300s", "300-600s", ">600s"),
    "hesitation_90s": ("no", "yes"),
    "sessions_3plus": ("no", "yes"),
    "viewed_pricing": ("no", "yes"),
    "search_term": ("generic", "brand", "not_google"),
    "business_hours": ("outside", "wkday_9-18"),
    "ip_country": ("match", "mismatch"),
    "ip_type": ("residential", "dc"),
    "c_budget": ("not_asked", "Under £5k", "£5k-£20k", "£20k-£50k", "£50k+"),
    "c_timeline": ("not_asked", "This month", "This quarter", "Next 6 months", "Just researching"),
    "edits_1_4": ("no", "yes"),
}

# Company-name enrichment (plan 2.3): trailing tokens stripped after normalisation. The plan's list plus
# "sas" (a legal form companies.csv uses) and the long forms incorporated / corporation / corp / company
# (typed on data/v2; review of Phase 2).
LEGAL_SUFFIXES: frozenset[str] = frozenset({"ltd", "limited", "inc", "llc", "gmbh", "bv", "sa", "plc", "co", "sas",
                                            "incorporated", "corporation", "corp", "company"})

# Boilerplate detector (plan 2.5): token-set Jaccard similarity to any snippet in ``emva/boilerplate.py``
# at or above this makes the free text "copy_paste".
BOILERPLATE_SIMILARITY_THRESHOLD: float = 0.6

# Generic feature set (Phase 10, ADR 0024): extra source columns declared in a mapping's ``[[features]]``, encoded on
# the training leads only. A categorical level is kept when at least GENERIC_MIN_LEVEL_COUNT training leads have it
# (rarer and unseen values fall into ``other``); a numeric extra gets GENERIC_NUMERIC_BINS quantile bins.
GENERIC_MIN_LEVEL_COUNT: int = 30
GENERIC_NUMERIC_BINS: int = 5

# Collinearity check (plan 2.7): the largest |correlation| allowed between two design columns.
MAX_ABS_DESIGN_CORR: float = 0.95

# Formula model.
LR_C: float = 0.5
LR_MAX_ITER: int = 5000

# Deal-value model. Its lognormal mean correction uses the residual sd estimated on training (plan 2.4).
DEAL_VALUE_FEATURES: tuple[str, ...] = ("band", "sector", "channel", "spend")
DEAL_VALUE_RIDGE_ALPHA: float = 3.0
# Complete level list of every deal-value feature for the v2 feature set (plan 3.1, the ADR 0009 rule applied
# to the value model): one ridge column per level listed here, whatever occurs in the data; a value not listed
# raises. Sectors are the companies.csv sectors; ``sector`` is ``MISSING`` without enrichment.
DEAL_VALUE_LEVELS: dict[str, tuple[str, ...]] = {
    "band": V2_LEVELS["band"],
    "sector": ("Education", "Financial services", "Healthcare", "Logistics", "Manufacturing",
               "Professional services", "Retail", "Software", MISSING),
    "channel": V2_LEVELS["channel"],
    "spend": V2_LEVELS["spend"],
}

# Value transform defaults (plan 3.1; ADR 0012): cap the expected value at this percentile of the
# training leads' values, compress it (``emva.value_transform.Compression``: "none", "log" or "sqrt") and floor it
# at this many GBP. CLI: --value-cap-percentile, --value-compression, --value-floor.
VALUE_CAP_PERCENTILE: float = 97.0
VALUE_COMPRESSION: str = "log"
VALUE_FLOOR_GBP: float = 25.0

# Scorecard: +20 points = odds of closing double.
POINTS_TO_DOUBLE_ODDS: float = 20.0

# Share of the ranked test set used for top-k capture metrics.
TOP_FRACTION: float = 0.2

# Context layer: context scores are clipped to this range before the logit.
CONTEXT_SCORE_CLIP: tuple[float, float] = (0.001, 0.999)
