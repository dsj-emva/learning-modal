"""The v2 feature set (plan 2.1, 2.2, 2.5, 2.6, orchestrator indicator ruling) and its design on v1."""
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from emva.constants import CATS, CATS_V2_CANDIDATES, MISSING, V2_LEVELS
from emva.design import design, feature_design, fixed_columns, fixed_design
from emva.pipeline import run

from conftest import REPO
from emva.features import (
    BEHAVIOURAL_FEATURES,
    CATS_V2,
    SESSION_INPUTS,
    V2_DROPPED,
    FeatureSet,
    add_features_v2,
    feature_cats,
    session_absent,
    text_cat_v2,
)

REFERENCE = {f: CATS_V2_CANDIDATES[f] for f in BEHAVIOURAL_FEATURES}
V2_DESIGN_COLUMNS = 39  # sum over CATS_V2 of (declared levels - 1)


def _lead(**kw) -> dict:
    """A website lead with every behavioural input present and domain enrichment."""
    base = dict(form_variant="B", utm_medium="cpc", utm_source="google", email_l="a@acme.example", a_what_to_solve="x",
                a_job_title=None, a_company_size=None, enrichment_source="domain", en_employee_band="51-200",
                en_monthly_ad_spend_band="£5k-£25k", en_crm_platform="HubSpot", en_is_hiring=True,
                en_sector="Software", time_on_page_s=120.0, hesitation_ms=1000.0, sessions_before_convert=1.0,
                viewed_pricing=True, utm_term=None, landing_url="https://lp.example/", submitted_weekday="Monday",
                local_submit_hour=10, ip_country="US", a_country="UK", is_datacenter_ip=True, a_budget=None,
                a_timeline=None, field_edit_count=0)
    base.update(kw)
    return base


def _features(*rows: dict) -> pd.DataFrame:
    return add_features_v2(pd.DataFrame(list(rows)))


def test_present_inputs_get_their_buckets():
    r = _features(_lead()).iloc[0]
    assert r.session_missing == "no" and r.enrichment_missing == "no"
    assert r.time_on_page == "60-300s" and r.hesitation_90s == "no" and r.sessions_3plus == "no"
    assert r.viewed_pricing == "yes" and r.ip_country == "mismatch" and r.ip_type == "dc"
    assert r.business_hours == "wkday_9-18" and r.band == "51-200"
    assert r.spend == "£5k-£25k" and r.crm == "hubspot_sf" and r.hiring == "hiring" and r.sector == "Software"


@pytest.mark.parametrize("raw", [*SESSION_INPUTS, "landing_url"])
def test_absent_session_input_sets_the_indicator_and_reference_levels(raw):
    """A NaN session input never lands silently in a reference bucket: session_missing=yes carries it."""
    r = _features(_lead(**{raw: np.nan})).iloc[0]
    assert r.session_missing == "yes"
    assert {f: r[f] for f in BEHAVIOURAL_FEATURES} == REFERENCE  # 4 of the 7 are non-reference when present
    assert not any(r[f] == MISSING for f in BEHAVIOURAL_FEATURES)


def test_session_absent():
    X = pd.DataFrame([_lead(), _lead(time_on_page_s=np.nan), _lead(landing_url=np.nan)])
    assert session_absent(X).tolist() == [False, True, True]


@pytest.mark.parametrize("seconds,bucket", [(15, "15-60s"), (59.9, "15-60s"), (60, "60-300s"), (300, "300-600s"),
                                            (600, ">600s")])
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


def test_lead_ads_have_no_session_and_business_hours_has_no_channel_rule():
    no_session = dict(landing_url=np.nan, time_on_page_s=np.nan, hesitation_ms=np.nan, sessions_before_convert=np.nan,
                      viewed_pricing=np.nan, ip_country=np.nan, is_datacenter_ip=np.nan, field_edit_count=np.nan)
    consent_declined = {k: v for k, v in no_session.items() if k != "landing_url"}
    X = _features(
        _lead(utm_medium="lead_form", utm_source="facebook", **no_session),       # lead ads, weekday 10:00
        _lead(utm_medium="paid_social", utm_source="facebook", **consent_declined),  # website, no telemetry
        _lead(utm_medium="paid_social", utm_source="facebook"),                    # website lead, weekday 10:00
    )
    assert X.channel.tolist() == ["meta_leadads", "meta", "meta"]
    assert X.session_missing.tolist() == ["yes", "yes", "no"]
    assert X.business_hours.tolist() == ["outside", "outside", "wkday_9-18"]
    for feat in BEHAVIOURAL_FEATURES:
        assert (X[feat].iloc[:2] == REFERENCE[feat]).all(), feat


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
    for feat in ("spend", "crm", "hiring"):  # reference level; the indicator carries the effect
        assert X[feat].iloc[2:].eq(CATS_V2_CANDIDATES[feat]).all() and not X[feat].eq(MISSING).any(), feat
    assert X.sector.iloc[2:].eq(MISSING).all()  # deal-value model level
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
    assert {"session_missing", "enrichment_missing"} <= set(CATS_V2)
    assert FeatureSet("legacy") is FeatureSet.LEGACY
    with pytest.raises(ValueError):
        FeatureSet("v3")


def test_v2_design_columns_come_from_the_spec_not_the_data(v1_horizon):
    X, D, tr = v1_horizon.X, v1_horizon.design, v1_horizon.train
    assert v1_horizon.features is FeatureSet.V2
    assert list(D.columns) == fixed_columns(CATS_V2) and D.shape[1] == V2_DESIGN_COLUMNS  # from the spec
    assert [c for c in D.columns if c.endswith("=missing")] == ["band=missing"]
    assert "session_missing=yes" in D and "enrichment_missing=yes" in D
    assert not any(c.split("=", 1)[0] in V2_DROPPED or c.startswith("no_company") for c in D.columns)
    # a property of v1: every session-less lead is a lead-ads lead, so the two columns coincide
    assert D.loc[tr, "session_missing=yes"].equals(D.loc[tr, "channel=meta_leadads"])
    lead_ads = X.channel == "meta_leadads"
    assert (X.loc[lead_ads, "session_missing"] == "yes").all()
    assert (X.loc[lead_ads, list(BEHAVIOURAL_FEATURES)] == pd.Series(REFERENCE)).all().all()


def test_equal_coefficients_on_v1(v1_horizon):
    """Acceptance 1 on v1: the identical columns session_missing / meta_leadads get equal L2 weights (not hidden)."""
    w, D = v1_horizon.weights.log_odds, v1_horizon.design[v1_horizon.train]
    ties = {(a, b) for a, b in combinations(w.index, 2) if w[a] == w[b]}
    structural = {t for t in ties if abs(D[t[0]].corr(D[t[1]])) > 0.95}
    assert {frozenset(t) for t in structural} == {frozenset({"channel=meta_leadads", "session_missing=yes"})}
    assert "email=free" not in {c for t in ties for c in t}  # the baseline's duplicated pair is gone


def test_v2_residual_sd_is_estimated(v1_horizon, v1_result):
    assert v1_horizon.deal_value.estimated and not v1_result.deal_value.estimated
    assert 0.3 < v1_horizon.deal_value.log_residual_sd < 0.7


def test_level_lists_cover_every_candidate_with_its_reference_first():
    assert set(V2_LEVELS) == set(CATS_V2_CANDIDATES)
    assert all(V2_LEVELS[f][0] == ref and len(set(V2_LEVELS[f])) == len(V2_LEVELS[f])
               for f, ref in CATS_V2_CANDIDATES.items())
    assert len(fixed_columns(CATS_V2)) == V2_DESIGN_COLUMNS


def test_one_row_frame_produces_the_full_column_set():
    X = _features(_lead())
    D = fixed_design(X, CATS_V2)
    assert list(D.columns) == fixed_columns(CATS_V2) and D.shape == (1, V2_DESIGN_COLUMNS)
    assert D.loc[0, "channel=meta"] == 0.0 and D.loc[0, "time_on_page=60-300s"] == 1.0  # absent level -> zero column
    assert D.loc[0, "session_missing=yes"] == 0.0 and set(D.values.ravel()) <= {0.0, 1.0}
    assert list(feature_design(X, FeatureSet.V2).columns) == list(D.columns)


def test_unseen_level_raises():
    X = _features(_lead())
    X.loc[0, "form_variant"] = "E"
    with pytest.raises(ValueError, match=r"feature 'form_variant' has values outside its declared levels .*'E'"):
        fixed_design(X, CATS_V2)


def test_feature_without_a_level_list_is_refused():
    with pytest.raises(ValueError, match="no level list"):
        fixed_columns({"new_feature": "x"})


def test_legacy_design_stays_data_driven(v1_result):
    X = v1_result.X
    assert list(feature_design(X, FeatureSet.LEGACY).columns) == list(design(X).columns)
    assert list(design(X).columns) == list(v1_result.design.columns)
    assert len(design(X.iloc[:1]).columns) < len(design(X).columns)  # one row: only the levels it has
    assert CATS == feature_cats(FeatureSet.LEGACY)


def test_no_equal_coefficients_on_data_v2():
    """ADR 0011: the 'no identical coefficients' criterion is evaluated on data/v2."""
    w = run(REPO / "data" / "v2").weights.log_odds
    assert [(a, b) for a, b in combinations(w.index, 2) if w[a] == w[b]] == []
