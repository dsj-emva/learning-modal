"""The v2 feature set (plan 2.1, 2.2, 2.5, 2.6) and its design on v1."""
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from emva.constants import CATS_V2_CANDIDATES, MISSING
from emva.features import (
    BEHAVIOURAL_FEATURES,
    CATS_V2,
    V2_DROPPED,
    FeatureSet,
    add_features_v2,
    feature_cats,
    text_cat_v2,
)


def _lead(**kw) -> dict:
    """A website lead with every behavioural input present and domain enrichment."""
    base = dict(utm_medium="cpc", utm_source="google", email_l="a@acme.example", a_what_to_solve="x",
                a_job_title=None, a_company_size=None, enrichment_source="domain", en_employee_band="51-200",
                en_monthly_ad_spend_band="£5k-£25k", en_crm_platform="HubSpot", en_is_hiring=True,
                en_sector="Software", time_on_page_s=120.0, hesitation_ms=1000.0, sessions_before_convert=1.0,
                viewed_pricing=False, utm_term=None, landing_url="https://lp.example/", submitted_weekday="Monday",
                local_submit_hour=10, ip_country="UK", a_country="UK", is_datacenter_ip=False, a_budget=None,
                a_timeline=None, field_edit_count=0)
    base.update(kw)
    return base


def _features(*rows: dict) -> pd.DataFrame:
    return add_features_v2(pd.DataFrame(list(rows)))


def test_present_inputs_get_their_buckets():
    r = _features(_lead()).iloc[0]
    assert r.time_on_page == "60-300s" and r.hesitation_90s == "no" and r.sessions_3plus == "no"
    assert r.viewed_pricing == "no" and r.ip_country == "match" and r.ip_type == "residential"
    assert r.business_hours == "wkday_9-18" and r.enrichment_missing == "no" and r.band == "51-200"
    assert r.spend == "£5k-£25k" and r.crm == "hubspot_sf" and r.hiring == "hiring" and r.sector == "Software"


@pytest.mark.parametrize("raw,feature,reference", [
    ("time_on_page_s", "time_on_page", "15-60s"),
    ("hesitation_ms", "hesitation_90s", "no"),
    ("sessions_before_convert", "sessions_3plus", "no"),
    ("viewed_pricing", "viewed_pricing", "no"),
    ("ip_country", "ip_country", "match"),
    ("is_datacenter_ip", "ip_type", "residential"),
    ("local_submit_hour", "business_hours", "outside"),
    ("submitted_weekday", "business_hours", "outside"),
    ("landing_url", "business_hours", "outside"),
])
def test_missing_telemetry_lands_in_missing_not_the_reference(raw, feature, reference):
    got = _features(_lead(**{raw: np.nan})).iloc[0][feature]
    assert got == MISSING and got != reference and CATS_V2_CANDIDATES[feature] == reference


@pytest.mark.parametrize("seconds,bucket", [(15, "15-60s"), (59.9, "15-60s"), (60, "60-300s"), (300, "300-600s"),
                                            (600, ">600s"), (np.nan, MISSING)])
def test_time_on_page_buckets(seconds, bucket):
    assert _features(_lead(time_on_page_s=seconds)).time_on_page.iloc[0] == bucket


def test_binary_telemetry_thresholds():
    X = _features(_lead(hesitation_ms=90000, sessions_before_convert=2, viewed_pricing=False, ip_country="UK",
                        is_datacenter_ip=False),
                  _lead(hesitation_ms=90001, sessions_before_convert=3, viewed_pricing=True, ip_country="US",
                        is_datacenter_ip=True))
    assert X.hesitation_90s.tolist() == ["no", "yes"] and X.sessions_3plus.tolist() == ["no", "yes"]
    assert X.viewed_pricing.tolist() == ["no", "yes"] and X.ip_country.tolist() == ["match", "mismatch"]
    assert X.ip_type.tolist() == ["residential", "dc"]


def test_lead_ads_business_hours_is_missing_not_outside():
    """Lead-ads leads have no on-site session: business_hours is missing from the data, not from the channel."""
    no_session = dict(landing_url=np.nan, time_on_page_s=np.nan, hesitation_ms=np.nan, sessions_before_convert=np.nan,
                      viewed_pricing=np.nan, ip_country=np.nan, is_datacenter_ip=np.nan, field_edit_count=np.nan)
    X = _features(
        _lead(utm_medium="lead_form", utm_source="facebook", **no_session),        # lead ads, weekday 10:00
        _lead(utm_medium="lead_form", utm_source="facebook", submitted_weekday="Sunday", **no_session),
        _lead(utm_medium="paid_social", utm_source="facebook"),                     # website lead, weekday 10:00
        _lead(utm_medium="paid_social", utm_source="facebook", landing_url=np.nan),  # website lead, no session
    )
    assert X.channel.tolist() == ["meta_leadads", "meta_leadads", "meta", "meta"]
    assert X.business_hours.tolist() == [MISSING, MISSING, "wkday_9-18", MISSING]
    for feat in BEHAVIOURAL_FEATURES:
        assert (X[feat].iloc[:2] == MISSING).all(), feat


@pytest.mark.parametrize("hour,weekday,expected", [(9, "Monday", "wkday_9-18"), (17, "Friday", "wkday_9-18"),
                                                   (18, "Monday", "outside"), (8, "Monday", "outside"),
                                                   (10, "Saturday", "outside")])
def test_business_hours_window(hour, weekday, expected):
    assert _features(_lead(local_submit_hour=hour, submitted_weekday=weekday)).business_hours.iloc[0] == expected


def test_enrichment_missing_and_band_fallback_order():
    no_en = dict(enrichment_source=np.nan, en_employee_band=np.nan, en_monthly_ad_spend_band=np.nan,
                 en_crm_platform=np.nan, en_is_hiring=np.nan, en_sector=np.nan)
    X = _features(
        _lead(a_company_size="1-10"),                                       # enrichment beats the typed size
        _lead(enrichment_source="name", en_employee_band="201-1000"),       # name-matched enrichment counts
        _lead(a_company_size="1000+", **no_en),                             # typed size when no enrichment
        _lead(a_company_size="", **no_en),                                  # blank answer = no answer
        _lead(**no_en),
    )
    assert X.band.tolist() == ["51-200", "201-1000", "1000+", MISSING, MISSING]
    assert X.enrichment_missing.tolist() == ["no", "no", "yes", "yes", "yes"]
    for feat in ("spend", "crm", "hiring", "sector"):
        assert X[feat].iloc[2:].eq(MISSING).all() and not X[feat].iloc[:2].eq(MISSING).any(), feat
    assert "no_company" not in X


def test_crm_and_hiring_levels_with_enrichment():
    X = _features(_lead(en_crm_platform="Salesforce", en_is_hiring=False), _lead(en_crm_platform="Pipedrive"))
    assert X.crm.tolist() == ["hubspot_sf", "other"] and X.hiring.tolist() == ["not_hiring", "hiring"]


@pytest.mark.parametrize("text,expected", [
    ("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore.", "copy_paste"),
    ("our mission is to empower businesses worldwide with scalable end to end solutions that drive growth, "
     "efficiency and customer delight", "copy_paste"),
    ("Our mission is to empower budget holders", "specific"),   # legacy prefix rule said copy_paste
    ("We are a leading provider", "neutral"),                  # legacy prefix rule said copy_paste
    ("tell me more", "vague"), ("", "vague"), (None, "vague"),
    ("Our budget is £10k", "specific"), ("We want better attribution for ads", "neutral"),
])
def test_text_cat_v2(text, expected):
    assert text_cat_v2(text) == expected


def test_feature_sets_and_2_6_drop():
    assert set(V2_DROPPED) == {"c_budget", "c_timeline", "ip_type", "edits_1_4"}
    assert not set(V2_DROPPED) & set(CATS_V2) and set(CATS_V2) | set(V2_DROPPED) == set(CATS_V2_CANDIDATES)
    assert feature_cats(FeatureSet.V2) is CATS_V2 and "no_company" in feature_cats(FeatureSet.LEGACY)
    assert FeatureSet("legacy") is FeatureSet.LEGACY
    with pytest.raises(ValueError):
        FeatureSet("v3")


def test_v2_design_on_v1(v1_horizon):
    X, D, tr = v1_horizon.X, v1_horizon.design, v1_horizon.train
    assert v1_horizon.features is FeatureSet.V2
    assert not any(c.split("=", 1)[0] in V2_DROPPED or c.startswith("no_company") for c in D.columns)
    assert "enrichment_missing=yes" in D and "band=missing" in D
    # on v1 the behavioural missing levels are exactly the lead-ads leads, so they alias the channel column
    for feat in BEHAVIOURAL_FEATURES:
        if feat in CATS_V2:
            assert v1_horizon.aliases[f"{feat}=missing"] == "channel=meta_leadads"
    for feat in ("spend", "crm", "hiring"):
        assert v1_horizon.aliases[f"{feat}=missing"] == "enrichment_missing=yes"
    lead_ads = X.channel == "meta_leadads"
    assert lead_ads.any() and (X.loc[lead_ads, list(BEHAVIOURAL_FEATURES)] == MISSING).all().all()
    assert D[tr].shape[1] == len(set(map(tuple, D[tr].T.values)))  # no two identical training columns left


def test_v2_default_model_has_no_equal_coefficients(v1_horizon):
    w = v1_horizon.weights.log_odds
    assert [(a, b) for a, b in combinations(w.index, 2) if w[a] == w[b]] == []


def test_v2_residual_sd_is_estimated(v1_horizon, v1_result):
    assert v1_horizon.deal_value.estimated and not v1_result.deal_value.estimated
    assert 0.3 < v1_horizon.deal_value.log_residual_sd < 0.7
