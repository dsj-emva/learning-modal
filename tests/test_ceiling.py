"""Oracle-feature ceiling (plan 4.4): reads only declared ground-truth columns, never pipeline features or reply time."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from emva.eval import ceiling as ce


def _truth(persona: bool = False) -> pd.DataFrame:
    t = pd.DataFrame({
        "p_close_true": [0.1, 0.2, 0.3, 0.4],
        "employee_band": ["none", "51-200", "1000+", "1-10"],
        "is_free_email": [True, False, False, np.nan],
        "text_category": ["vague", "specific", "copy_paste", "neutral"],
        "channel": ["meta", "linkedin", "linkedin", "google"],
        "is_lead_ads": [True, False, False, False],
        "seniority": ["student", "senior", "mid", "senior"],
        "company_domain": [np.nan, "Acme.example", "big.example", "unknown.example"],
        "country": ["UK", "DE", "US", "FR"],
    }, index=["L1", "L2", "L3", "L4"])
    if persona:
        t["context_persona"] = [np.nan, "competitor", np.nan, np.nan]
    return t


def _companies() -> pd.DataFrame:
    return pd.DataFrame({"monthly_ad_spend_band": ["£25k-£100k", "none"], "crm_platform": ["Salesforce", "No CRM"],
                         "is_hiring": [True, False]}, index=pd.Index(["acme.example", "big.example"], name="domain"))


def _session() -> pd.DataFrame:
    return pd.DataFrame({
        "time_on_page_s": [np.nan, 14.9, 600.0, 600.1], "hesitation_ms": [np.nan, 90001, 90000, 0],
        "field_edit_count": [np.nan, 4, 5, 1], "form_variant": ["A", "D", "D", "C"],
        "viewed_pricing": [np.nan, True, False, True], "sessions_before_convert": [np.nan, 3, 2, 5],
        "utm_term": [np.nan, "northstar pricing", "crm", np.nan],
        "submitted_weekday": ["Monday", "Saturday", "Tuesday", "Friday"], "local_submit_hour": [9, 12, 18, 17],
        "ip_country": [np.nan, "DE", "UK", "FR"],
    }, index=["L1", "L2", "L3", "L4"])


def test_oracle_features_follow_the_planted_definitions():
    F = ce.oracle_features(_truth(), _companies(), _session())
    assert F.time_on_page.tolist() == ["missing", "<15s", "300-600s", ">600s"]
    assert F.hesitation_90s.tolist() == ["missing", "yes", "no", "no"]
    assert F.edits_1_4.tolist() == ["missing", "yes", "no", "yes"]
    assert F.spend.tolist() == ["no_company", "£25k-£100k", "none", "no_company"]   # domain case-insensitive; unknown domain
    assert F.crm.tolist() == ["no_company", "hubspot_sf", "other", "no_company"]
    assert F.hiring.tolist() == ["no_company", "yes", "no", "no_company"]
    assert F.ip_mismatch.tolist() == ["no", "no", "yes", "no"]                         # against the TRUE country
    assert F.business_hours.tolist() == ["yes", "no", "no", "yes"]
    assert F.brand_term.tolist() == ["no", "yes", "no", "no"]
    assert F.free_email.tolist() == ["yes", "no", "no", "no"]                          # NaN-safe
    assert F.text.tolist() == _truth().text_category.tolist()                          # true category, not the regex
    assert F.linkedin_x_51plus.tolist() == ["no", "yes", "yes", "no"]
    assert F.senior_x_formD.tolist() == ["no", "yes", "no", "no"]
    assert "persona" not in F


def test_session_frame_may_not_carry_pipeline_columns():
    s = _session().assign(text="specific")
    with pytest.raises(ValueError, match="text"):
        ce.oracle_features(_truth(), _companies(), s)


def test_read_truth_reads_only_declared_columns(tmp_path):
    t = _truth(persona=True).rename_axis("lead_id").reset_index()
    t["reply_hours"] = [1.0, 2.0, 3.0, 4.0]     # the reply time and the outcome must never reach the oracle
    t["outcome"] = ["won", "lost", "open", "won"]
    t["duplicate_of"] = np.nan
    t.to_csv(tmp_path / ce.TRUTH_LABELS_FILE, index=False)
    got = ce.read_truth(tmp_path)
    assert set(got.columns) == {"p_close_true", *ce.TRUTH_COLUMNS, *ce.TRUTH_EXTRA_COLUMNS, ce.PERSONA_COLUMN}
    assert "reply_hours" not in got and "outcome" not in got


def test_true_companies_prefer_the_ground_truth_file(tmp_path):
    base = pd.DataFrame({"domain": ["A.example"], "monthly_ad_spend_band": ["none"], "crm_platform": ["HubSpot"],
                         "is_hiring": [True], "sector": ["Retail"]})
    base.to_csv(tmp_path / "companies.csv", index=False)
    assert ce.read_true_companies(tmp_path).loc["a.example"].monthly_ad_spend_band == "none"
    base.assign(monthly_ad_spend_band="£100k+").to_csv(tmp_path / ce.TRUE_COMPANIES_FILE, index=False)
    assert ce.read_true_companies(tmp_path).loc["a.example"].monthly_ad_spend_band == "£100k+"
    pd.concat([base, base]).to_csv(tmp_path / ce.TRUE_COMPANIES_FILE, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        ce.read_true_companies(tmp_path)


def test_oracle_sets_and_designs():
    F = ce.oracle_features(_truth(), _companies(), _session())
    assert ce.available_sets(F) == [ce.ORACLE_TRUTH_ONLY, ce.ORACLE_FORMULA, ce.ORACLE_INTERACTIONS]
    with pytest.raises(KeyError, match="persona"):
        ce.oracle_design(F, ce.ORACLE_PERSONA)
    D = ce.oracle_design(F, ce.ORACLE_TRUTH_ONLY)
    assert all(c.split("=")[0] in ce.SET_FEATURES[ce.ORACLE_TRUTH_ONLY] for c in D.columns)
    Fp = ce.oracle_features(_truth(persona=True), _companies(), _session())
    assert ce.available_sets(Fp)[-1] == ce.ORACLE_PERSONA
    assert ce.persona_indicator(Fp).tolist() == [0.0, 1.0, 0.0, 0.0]
    assert "persona=yes" in ce.oracle_design(Fp, ce.ORACLE_PERSONA)
    # no set uses the reply time or the noise, and the formula set holds every truth-only feature
    assert set(ce.SET_FEATURES[ce.ORACLE_TRUTH_ONLY]) < set(ce.SET_FEATURES[ce.ORACLE_FORMULA])
