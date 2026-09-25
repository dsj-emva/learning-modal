"""Every constant the Phase 0 pipeline uses, in one place.

Values are copied verbatim from ``baseline/emva_score.py``. Changing any of them changes
model behaviour, so each change belongs to a numbered plan task with its own report.
"""
from __future__ import annotations

import pandas as pd

# Snapshot date of the data and the frozen train/test boundary (ground rule 4).
AS_OF: pd.Timestamp = pd.Timestamp("2026-09-24", tz="UTC")
TEST_FROM: str = "2026-05-01"

# Label rules (baseline decision log): open deals idle this long, and never-contacted
# leads this old, are labelled Lost.
STALLED_DAYS: int = 90
GHOSTED_DAYS: int = 90
OPEN_STAGES: tuple[str, ...] = ("Contacted", "Qualified", "Demo booked", "Proposal")

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

# Form answers extracted from the ``answers`` JSON into ``a_<key>`` columns.
ANSWER_KEYS: tuple[str, ...] = ("country", "what_to_solve", "job_title", "company_size", "budget", "timeline")
# companies.csv columns joined onto leads as ``co_<column>``.
ENRICHMENT_COLUMNS: tuple[str, ...] = ("sector", "employee_band", "monthly_ad_spend_band", "crm_platform", "is_hiring")

# feature -> reference level (weights are relative to this level). Order defines design columns.
CATS: dict[str, str] = {
    "channel": "google", "form_variant": "A", "band": "1-10", "email": "business", "text": "neutral",
    "seniority": "junior/ic", "spend": "under £5k", "crm": "other", "hiring": "not_hiring",
    "time_on_page": "15-60s", "hesitation_90s": "no", "sessions_3plus": "no", "viewed_pricing": "no",
    "search_term": "generic", "business_hours": "outside", "ip_country": "match", "ip_type": "residential",
    "c_budget": "not_asked", "c_timeline": "not_asked", "edits_1_4": "no", "no_company": "no",
}

# Formula model.
LR_C: float = 0.5
LR_MAX_ITER: int = 5000

# Deal-value model (the residual-variance constant lives in emva/value.py, see plan 2.4).
DEAL_VALUE_FEATURES: tuple[str, ...] = ("band", "sector", "channel", "spend")
DEAL_VALUE_RIDGE_ALPHA: float = 3.0

# Scorecard: +20 points = odds of closing double.
POINTS_TO_DOUBLE_ODDS: float = 20.0

# Share of the ranked test set used for top-k capture metrics.
TOP_FRACTION: float = 0.2

# Context layer: context scores are clipped to this range before the logit.
CONTEXT_SCORE_CLIP: tuple[float, float] = (0.001, 0.999)
