"""Plan 3.1 fixed-schema deal-value design and plan 3.3 two-stage columns (value_at_submit / value_at_close)."""
import numpy as np
import pandas as pd
import pytest

from emva.constants import AS_OF, DEAL_VALUE_FEATURES, DEAL_VALUE_LEVELS
from emva.feature_spec import feature_spec
from emva.features import FeatureSet
from emva.pipeline import VALUE_SCORE_COLUMNS
from emva.value import fixed_deal_value_columns, fixed_deal_value_design, value_at_close


def _one_lead(**over: str) -> pd.DataFrame:
    row = {"band": "51-200", "sector": "Software", "channel": "linkedin", "spend": "£5k-£25k"} | over
    return pd.DataFrame([row], index=["L1"])


def test_fixed_deal_value_design_gives_one_row_the_full_schema():
    M = fixed_deal_value_design(_one_lead())
    assert list(M.columns) == fixed_deal_value_columns()
    assert M.shape == (1, sum(len(DEAL_VALUE_LEVELS[c]) for c in DEAL_VALUE_FEATURES))
    assert M.loc["L1"].sum() == len(DEAL_VALUE_FEATURES)
    assert M.loc["L1", "band=51-200"] == 1 and M.loc["L1", "sector=Software"] == 1
    assert M.loc["L1", "sector=missing"] == 0


@pytest.mark.parametrize("feature,value", [("sector", "Aerospace"), ("band", "5000+"), ("channel", "tiktok"),
                                           ("spend", "none at all")])
def test_fixed_deal_value_design_refuses_an_unseen_level(feature, value):
    with pytest.raises(ValueError, match=feature):
        fixed_deal_value_design(_one_lead(**{feature: value}))


def test_v2_feature_set_uses_the_fixed_value_design_and_legacy_the_data_driven_one(v1_horizon):
    one = v1_horizon.X.head(1)
    assert list(feature_spec(FeatureSet.V2).value_design(one).columns) == fixed_deal_value_columns()
    assert feature_spec(FeatureSet.LEGACY).value_design(one).shape[1] == len(DEAL_VALUE_FEATURES)


def _close_frame() -> pd.DataFrame:
    H = pd.Timedelta(days=120)
    old = AS_OF - H - pd.Timedelta(days=10)   # mature
    young = AS_OF - pd.Timedelta(days=10)     # immature
    return pd.DataFrame({
        "created_at": [old, old, young, young, old, old],
        "won_at": [old + pd.Timedelta(days=5), pd.NaT, young + pd.Timedelta(days=2), pd.NaT,
                   old + H + pd.Timedelta(days=3), old + pd.Timedelta(days=1)],
        "deal_value": [12000.0, np.nan, 3000.0, np.nan, 7000.0, np.nan],
    }, index=["won_mature", "lost_mature", "won_young", "open_young", "won_late", "won_blank"])


def test_value_at_close_rules():
    X = _close_frame()
    out = value_at_close(X, horizon_days=120, as_of=AS_OF)
    # won -> recorded deal value at won_at, mature or not, within H or later
    for lead, value in [("won_mature", 12000.0), ("won_young", 3000.0), ("won_late", 7000.0)]:
        assert out.loc[lead, "value_at_close"] == value
        assert out.loc[lead, "value_at_close_ts"] == X.loc[lead, "won_at"]
    # won with a blank deal value -> 0 (ADR 0002 revenue convention), still at won_at
    assert out.loc["won_blank", "value_at_close"] == 0
    assert out.loc["won_blank", "value_at_close_ts"] == X.loc["won_blank", "won_at"]
    # mature non-win -> 0 at matured_at
    assert out.loc["lost_mature", "value_at_close"] == 0
    assert out.loc["lost_mature", "value_at_close_ts"] == X.loc["lost_mature", "created_at"] + pd.Timedelta(days=120)
    # immature and not won -> not known yet
    assert np.isnan(out.loc["open_young", "value_at_close"]) and pd.isna(out.loc["open_young", "value_at_close_ts"])


def test_value_at_close_at_the_maturity_boundary_and_for_a_win_after_as_of():
    created = AS_OF - pd.Timedelta(days=120)
    X = pd.DataFrame({"created_at": [created, created + pd.Timedelta(seconds=1), created],
                      "won_at": [pd.NaT, pd.NaT, AS_OF + pd.Timedelta(days=1)],
                      "deal_value": [np.nan, np.nan, 5000.0]}, index=["exactly", "one_second_young", "future_win"])
    out = value_at_close(X, horizon_days=120, as_of=AS_OF)
    assert out.loc["exactly", "value_at_close"] == 0 and out.loc["exactly", "value_at_close_ts"] == AS_OF
    assert np.isnan(out.loc["one_second_young", "value_at_close"])
    # a Won row dated after as_of is not known at as_of; the lead is mature, so 0 at matured_at
    assert out.loc["future_win", "value_at_close"] == 0


def test_scores_carry_two_stage_columns_in_horizon_mode(v1_horizon):
    s = v1_horizon.scores()
    for c in ("value_at_submit", "value_at_submit_ts", "value_at_close", "value_at_close_ts", "label_source",
              "matured_at"):
        assert c in s.columns
    assert (s.value_at_submit_ts == v1_horizon.X.created_at).all()
    assert s.value_at_submit.notna().all() and (s.value_at_submit >= 25).all()
    assert (s.value_at_close.notna() == s.value_at_close_ts.notna()).all()
    won = s.label_source == "won"
    assert (s.value_at_close[won] > 0).mean() > 0.9 and (s.value_at_close[~won].fillna(0) == 0).all()
    # the transform is fitted on the training leads' values
    assert v1_horizon.value_transform.cap == pytest.approx(
        np.percentile(v1_horizon.X.value_formula[v1_horizon.train], 97))


def test_legacy_scores_have_no_two_stage_columns(v1_result):
    assert not set(VALUE_SCORE_COLUMNS) & set(v1_result.scores().columns)
    assert v1_result.value_transform is None
