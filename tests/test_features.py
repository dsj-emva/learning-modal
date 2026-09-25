import numpy as np
import pandas as pd
import pytest

from emva.features import add_features, channel, seniority, text_cat


@pytest.mark.parametrize("text,expected", [
    ("Lorem ipsum dolor sit amet", "copy_paste"),
    ("  We are a leading provider of stuff", "copy_paste"),
    ("Our mission is to empower budget holders", "copy_paste"),  # copy-paste beats specific
    ("", "vague"), (None, "vague"), ("  Tell me more ", "vague"), ("PRICING", "vague"), ("?", "vague"),
    ("Our budget is £10k", "specific"), ("we have 12 SDRs", "specific"), ("team of 5 marketers", "specific"),
    ("need it within 6 weeks", "specific"), ("live by March please", "specific"),
    ("before Q3 ideally", "specific"), ("our contract ends soon", "specific"), ("£ 5000/month", "specific"),
    ("We want better attribution for ads", "neutral"), ("pricing details please", "neutral"),
])
def test_text_cat(text, expected):
    assert text_cat(text) == expected


@pytest.mark.parametrize("title,expected", [
    (None, "not_asked/blank"), (np.nan, "not_asked/blank"), ("", "not_asked/blank"), ("   ", "not_asked/blank"),
    ("Student", "student"), ("Marketing Intern", "student"), ("Student Director", "student"),
    ("CEO", "senior"), ("Head of Growth", "senior"), ("VP Marketing", "senior"), ("Co-Founder", "senior"),
    ("Marketing Manager", "mid"), ("Team Lead", "mid"),
    ("Analyst", "junior/ic"), ("Sales Rep", "junior/ic"),
    # substring matching is part of the baseline behaviour: "Account Executive" contains no keyword,
    # but "Leadership coach" contains "lead"
    ("Account Executive", "junior/ic"), ("Leadership coach", "mid"),
])
def test_seniority(title, expected):
    assert seniority(title) == expected


def test_channel_mapping_first_rule_wins():
    med = pd.Series(["lead_form", "lead_form", "paid_social", "paid_social", "cpc", "paid_social", "cpc", None, "email"])
    src = pd.Series(["facebook", "linkedin", "instagram", "facebook", "google", "linkedin", "chatgpt", None, "newsletter"])
    assert channel(med, src).tolist() == ["meta_leadads", "meta_leadads", "meta", "meta", "google", "linkedin",
                                          "chatgpt", "organic_direct", "organic_direct"]


def _lead(**kw) -> dict:
    base = dict(utm_medium="cpc", utm_source="google", co_employee_band=np.nan, a_company_size=None,
                email_l="a@acme.example", a_what_to_solve="x", a_job_title=None, co_monthly_ad_spend_band=np.nan,
                co_crm_platform=np.nan, co_is_hiring=np.nan, time_on_page_s=np.nan, hesitation_ms=np.nan,
                sessions_before_convert=np.nan, viewed_pricing=np.nan, utm_term=None, submitted_weekday="Monday",
                local_submit_hour=10, ip_country=np.nan, a_country="UK", is_datacenter_ip=np.nan, a_budget=None,
                a_timeline=None, field_edit_count=0, co_sector=np.nan)
    base.update(kw)
    return base


def test_missing_values_fall_into_reference_levels():
    X = add_features(pd.DataFrame([_lead()]))
    r = X.iloc[0]
    assert r.no_company == "yes" and r.band == "1-10" and r.spend == "under £5k" and r.crm == "other"
    assert r.hiring == "not_hiring" and r.time_on_page == "15-60s" and r.hesitation_90s == "no"
    assert r.sessions_3plus == "no" and r.viewed_pricing == "no" and r.ip_country == "match"
    assert r.ip_type == "residential" and r.c_budget == "not_asked" and r.sector == "none"


def test_band_prefers_enrichment_then_typed_answer():
    X = add_features(pd.DataFrame([_lead(co_employee_band="51-200", a_company_size="1-10"),
                                   _lead(a_company_size="201-1000"), _lead(a_company_size="")]))
    assert X.band.tolist() == ["51-200", "201-1000", "1-10"]
    assert X.no_company.tolist() == ["no", "yes", "yes"]


def test_email_search_term_and_business_hours():
    X = add_features(pd.DataFrame([
        _lead(email_l="x@gmail.example", utm_term="Northstar pricing"),
        _lead(utm_medium="lead_form", utm_source="facebook"),
        _lead(local_submit_hour=18),
        _lead(submitted_weekday="Sunday"),
    ]))
    assert X.email.tolist() == ["free", "business", "business", "business"]
    assert X.search_term.tolist() == ["brand", "not_google", "generic", "generic"]
    # lead ads are always "outside" business hours (baseline special case, removed in plan 2.1)
    assert X.business_hours.tolist() == ["wkday_9-18", "outside", "outside", "outside"]


@pytest.mark.parametrize("seconds,bucket", [(15, "15-60s"), (59.9, "15-60s"), (60, "60-300s"), (300, "300-600s"),
                                            (600, ">600s")])
def test_time_on_page_buckets(seconds, bucket):
    assert add_features(pd.DataFrame([_lead(time_on_page_s=seconds)])).time_on_page.iloc[0] == bucket


def test_binary_thresholds():
    X = add_features(pd.DataFrame([
        _lead(hesitation_ms=90000, sessions_before_convert=2, field_edit_count=5, ip_country="UK",
              is_datacenter_ip=False, co_is_hiring=False, viewed_pricing=False),
        _lead(hesitation_ms=90001, sessions_before_convert=3, field_edit_count=1, ip_country="US",
              is_datacenter_ip=True, co_is_hiring=True, viewed_pricing=True),
    ]))
    assert X.hesitation_90s.tolist() == ["no", "yes"]
    assert X.sessions_3plus.tolist() == ["no", "yes"]
    assert X.edits_1_4.tolist() == ["no", "yes"]
    assert X.ip_country.tolist() == ["match", "mismatch"]
    assert X.ip_type.tolist() == ["residential", "dc"]
    assert X.hiring.tolist() == ["not_hiring", "hiring"]
    assert X.viewed_pricing.tolist() == ["no", "yes"]
