"""Oracle-feature ceiling (plan 4.4). Reads ground truth: evaluation only (ground rule 2).

The ceiling is the AUC of the formula model (``emva.model.make_lr``, same C) fitted on the generator's
own inputs instead of the pipeline's bucketed guesses, on the same rows and the same train/test
splits as the pipeline. Nothing here is imported by the pipeline.

Oracle feature sets (``ORACLE_SETS``):

``truth file only``
    The categorical columns of ``ground_truth_labels.csv`` that carry a planted effect
    (``TRUTH_COLUMNS``): true company size (``none`` = no company), free email, true text category,
    channel, lead-ads flag, true seniority. The literal reading of the plan item.
``formula``
    Every planted main effect of ``ground_truth.md`` except the noise term and the sales reply time:
    the truth-file columns above, the true company's ad spend, CRM and hiring (``ground_truth_companies.csv``
    on v2, ``companies.csv`` on v1, where it is complete), IP country vs the *true* country, and the
    session inputs the generator scored as recorded (``SESSION_COLUMNS``, bucketed at the planted
    cut points). A session with no telemetry (Meta lead ads; consent declined on v2) gets a
    ``missing`` level: the generator scored the hidden true behaviour, which no model can see.
``formula + interactions``
    Adds the two v2 terms (LinkedIn × true size 51+, true senior × form D; planted 0 on v1).
``formula + interactions + persona`` (v2 only)
    Adds the hidden ``context_persona`` (5.7, planted −1.0). The difference from the row above is
    the planted context-only gap Phase 6 is trying to recover.

``pipeline + true persona`` (v2 only) is the pipeline's own design plus the true persona: the most a
perfect persona detector could add on top of today's features.

``p_close_true`` itself is reported as a reference: it includes the noise term and the reply
time, so it is above anything reachable from lead-time features.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from emva.eval.rolling import EvalFrame

TRUTH_LABELS_FILE: str = "ground_truth_labels.csv"
TRUE_COMPANIES_FILE: str = "ground_truth_companies.csv"   # v2; v1's companies.csv is already complete
TRUTH_COLUMNS: tuple[str, ...] = ("employee_band", "is_free_email", "text_category", "channel", "is_lead_ads",
                                  "seniority")
TRUTH_EXTRA_COLUMNS: tuple[str, ...] = ("company_domain", "country")
PERSONA_COLUMN: str = "context_persona"
COMPANY_COLUMNS: tuple[str, ...] = ("domain", "monthly_ad_spend_band", "crm_platform", "is_hiring")
# Recorded session inputs the generator's close log-odds reads (scripts/generate_data_v1.py::close_log_odds).
SESSION_COLUMNS: tuple[str, ...] = ("time_on_page_s", "hesitation_ms", "field_edit_count", "form_variant",
                                    "viewed_pricing", "sessions_before_convert", "utm_term", "submitted_weekday",
                                    "local_submit_hour", "ip_country")
NO_COMPANY: str = "no_company"
MISSING_SESSION: str = "missing"
LARGE_BANDS: tuple[str, ...] = ("51-200", "201-1000", "1000+")

ORACLE_TRUTH_ONLY = "truth file only"
ORACLE_FORMULA = "formula"
ORACLE_INTERACTIONS = "formula + interactions"
ORACLE_PERSONA = "formula + interactions + persona"
ORACLE_SETS: tuple[str, ...] = (ORACLE_TRUTH_ONLY, ORACLE_FORMULA, ORACLE_INTERACTIONS, ORACLE_PERSONA)


def read_truth(data: str | Path) -> pd.DataFrame:
    """``ground_truth_labels.csv`` indexed by ``lead_id``: the planted columns, domain, country, persona if present."""
    cols = ["lead_id", "p_close_true", *TRUTH_COLUMNS, *TRUTH_EXTRA_COLUMNS]
    header = pd.read_csv(Path(data) / TRUTH_LABELS_FILE, nrows=0).columns
    if PERSONA_COLUMN in header:
        cols.append(PERSONA_COLUMN)
    return pd.read_csv(Path(data) / TRUTH_LABELS_FILE, usecols=cols).set_index("lead_id")


def read_true_companies(data: str | Path) -> pd.DataFrame:
    """True firmographics indexed by lower-cased domain: ``ground_truth_companies.csv`` if present, else companies.csv."""
    path = Path(data) / TRUE_COMPANIES_FILE
    if not path.exists():
        path = Path(data) / "companies.csv"
    co = pd.read_csv(path, usecols=list(COMPANY_COLUMNS))
    co["domain"] = co.domain.str.lower()
    if co.domain.duplicated().any():
        raise ValueError(f"{path} has duplicate domains; the true company would be ambiguous")
    return co.set_index("domain")


def _bool(s: pd.Series) -> pd.Series:
    """NaN-safe boolean (object columns with gaps compare ``== True``)."""
    return s == True  # noqa: E712


def _time_bucket(t: pd.Series) -> np.ndarray:
    """Planted time-on-page buckets (<15s, 15-60s, 60-300s, 300-600s, >600s); ``missing`` without telemetry."""
    return np.select([t.isna(), t < 15, t < 60, t < 300, t <= 600],
                     [MISSING_SESSION, "<15s", "15-60s", "60-300s", "300-600s"], ">600s")


def oracle_features(truth: pd.DataFrame, companies: pd.DataFrame, session: pd.DataFrame) -> pd.DataFrame:
    """Categorical oracle features for every row of ``truth`` (see the module docstring).

    ``truth`` = ``read_truth`` rows; ``companies`` = ``read_true_companies``; ``session`` = the leads'
    ``SESSION_COLUMNS`` only (a frame with any other column is refused, so no pipeline feature can leak in).
    Returns one string column per planted effect, plus ``persona`` when ``truth`` has it.
    """
    extra = set(session.columns) - set(SESSION_COLUMNS)
    if extra:
        raise ValueError(f"session frame may only hold {SESSION_COLUMNS}; got extra columns {sorted(extra)}")
    s = session.reindex(truth.index)
    co = companies.reindex(truth.company_domain.str.lower())
    co.index = truth.index
    has_co = truth.company_domain.notna() & co.monthly_ad_spend_band.notna()
    telemetry = s.time_on_page_s.notna()
    F = pd.DataFrame(index=truth.index)
    F["band"] = truth.employee_band.astype(str)
    F["free_email"] = np.where(_bool(truth.is_free_email), "yes", "no")
    F["text"] = truth.text_category.astype(str)
    F["channel"] = truth.channel.astype(str)
    F["lead_ads"] = np.where(_bool(truth.is_lead_ads), "yes", "no")
    F["seniority"] = truth.seniority.astype(str)
    F["spend"] = np.where(has_co, co.monthly_ad_spend_band, NO_COMPANY)
    F["crm"] = np.where(~has_co, NO_COMPANY, np.where(co.crm_platform.isin(["HubSpot", "Salesforce"]), "hubspot_sf", "other"))
    F["hiring"] = np.where(~has_co, NO_COMPANY, np.where(_bool(co.is_hiring), "yes", "no"))
    F["time_on_page"] = _time_bucket(s.time_on_page_s)
    F["hesitation_90s"] = np.where(~telemetry, MISSING_SESSION, np.where(s.hesitation_ms > 90000, "yes", "no"))
    F["edits_1_4"] = np.where(~telemetry, MISSING_SESSION, np.where(s.field_edit_count.between(1, 4), "yes", "no"))
    F["form_variant"] = s.form_variant.astype(str)
    F["viewed_pricing"] = np.where(~telemetry, MISSING_SESSION, np.where(_bool(s.viewed_pricing), "yes", "no"))
    F["sessions_3plus"] = np.where(~telemetry, MISSING_SESSION, np.where(s.sessions_before_convert >= 3, "yes", "no"))
    F["brand_term"] = np.where(s.utm_term.fillna("").str.contains("northstar"), "yes", "no")
    biz = ~s.submitted_weekday.isin(["Saturday", "Sunday"]) & s.local_submit_hour.between(9, 17)
    F["business_hours"] = np.where(biz, "yes", "no")
    F["ip_mismatch"] = np.where(s.ip_country.notna() & (s.ip_country != truth.country), "yes", "no")
    F["linkedin_x_51plus"] = np.where((truth.channel == "linkedin") & truth.employee_band.isin(LARGE_BANDS), "yes", "no")
    F["senior_x_formD"] = np.where((truth.seniority == "senior") & (s.form_variant == "D"), "yes", "no")
    if PERSONA_COLUMN in truth:
        F["persona"] = np.where(truth[PERSONA_COLUMN].notna(), "yes", "no")
    return F


# Which oracle feature columns each set uses.
_TRUTH_ONLY_FEATURES = ("band", "free_email", "text", "channel", "lead_ads", "seniority")
_FORMULA_FEATURES = (*_TRUTH_ONLY_FEATURES, "spend", "crm", "hiring", "time_on_page", "hesitation_90s", "edits_1_4",
                     "form_variant", "viewed_pricing", "sessions_3plus", "brand_term", "business_hours", "ip_mismatch")
SET_FEATURES: dict[str, tuple[str, ...]] = {
    ORACLE_TRUTH_ONLY: _TRUTH_ONLY_FEATURES,
    ORACLE_FORMULA: _FORMULA_FEATURES,
    ORACLE_INTERACTIONS: (*_FORMULA_FEATURES, "linkedin_x_51plus", "senior_x_formD"),
    ORACLE_PERSONA: (*_FORMULA_FEATURES, "linkedin_x_51plus", "senior_x_formD", "persona"),
}


def oracle_design(F: pd.DataFrame, oracle_set: str) -> pd.DataFrame:
    """One-hot design (first level alphabetically dropped per feature) of ``oracle_set``'s columns of ``F``.

    Raises ``KeyError`` for an unknown set or when ``F`` lacks one of its columns (e.g. persona on v1).
    """
    cols = list(SET_FEATURES[oracle_set])
    missing = [c for c in cols if c not in F]
    if missing:
        raise KeyError(f"oracle set {oracle_set!r} needs columns {missing} that this data has no truth for")
    return pd.get_dummies(F[cols], prefix_sep="=", drop_first=True).astype(float)


def oracle_frame(frame: EvalFrame) -> tuple[pd.DataFrame, pd.Series]:
    """``(F, p_close_true)`` for the frame's rows: oracle features from ground truth plus the recorded session."""
    truth = read_truth(frame.data)
    session = pd.read_csv(frame.data / "historical_leads.csv", usecols=["lead_id", *SESSION_COLUMNS]).set_index("lead_id")
    t = truth.reindex(frame.X.index)
    if t.channel.isna().any():
        raise ValueError(f"{int(t.channel.isna().sum())} cleaned leads have no ground-truth row")
    return oracle_features(t, read_true_companies(frame.data), session), t.p_close_true


def persona_indicator(F: pd.DataFrame) -> pd.Series:
    """The true persona as a 0/1 design column ``persona=yes`` (KeyError when the data has no persona truth)."""
    return F["persona"].eq("yes").astype(float).rename("persona=yes")


def available_sets(F: pd.DataFrame) -> list[str]:
    """The oracle sets this data supports (the persona set needs ``context_persona``)."""
    return [s for s in ORACLE_SETS if all(c in F for c in SET_FEATURES[s])]
