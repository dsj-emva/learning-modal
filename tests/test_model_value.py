import numpy as np
import pandas as pd
import pytest

from emva.model import coefficients, scorecard
from emva.value import (
    DealValueModel,
    expected_value,
    fit_deal_value,
    predict_deal_value,
    realised_revenue,
    residual_sd,
)


class _FakeLR:
    coef_ = np.array([[np.log(2), -0.5, 0.0]])


def test_scorecard_points_and_order():
    w = scorecard(_FakeLR(), pd.Index(["a", "b", "c"]))
    assert w.index.tolist() == ["b", "c", "a"]
    assert w.loc["a", "points"] == 20 and w.loc["a", "odds_multiplier"] == 2.0
    assert w.loc["b", "log_odds"] == -0.5


def test_coefficients_are_the_unrounded_scorecard():
    lr = _FakeLR()
    lr.coef_ = np.array([[np.log(2) + 1e-7, -0.5004, 0.0]])
    c = coefficients(lr, pd.Index(["a", "b", "c"]))
    assert c.index.tolist() == ["a", "b", "c"] and c.loc["a", "log_odds"] == np.log(2) + 1e-7
    assert c.loc["a", "points"] == pytest.approx(20) and c.loc["a", "points"] != 20
    assert c.loc["b", "odds_multiplier"] == np.exp(-0.5004)
    w = scorecard(lr, pd.Index(["a", "b", "c"]))
    pd.testing.assert_frame_equal(w, c.round({"log_odds": 3, "odds_multiplier": 2, "points": 0}).sort_values("log_odds"))


def test_pipeline_result_keeps_the_fitted_model(v1_horizon):
    assert v1_horizon.model.coef_.shape == (1, v1_horizon.design.shape[1])
    assert list(v1_horizon.model.feature_names_in_) == list(v1_horizon.design.columns)


class _FakeRidge:
    def predict(self, M):
        return np.log(np.full(len(M), 1000.0))


def test_deal_value_lognormal_correction():
    model = DealValueModel(_FakeRidge(), log_residual_sd=0.3, estimated=False, n_train=10)
    got = predict_deal_value(model, pd.DataFrame({"x": [1.0, 2.0]}))
    assert got == pytest.approx(1000 * np.exp(0.3 ** 2 / 2))
    assert model.correction == pytest.approx(np.exp(0.045))


def _synthetic_deals(sd: float, n: int = 4000, seed: int = 3) -> tuple[pd.DataFrame, pd.Series]:
    """One-hot design with 3 groups; log value = group mean + N(0, sd)."""
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 3, n)
    M = pd.DataFrame({f"g_{k}": (g == k).astype(float) for k in range(3)})
    log_v = np.array([8.0, 9.0, 10.0])[g] + rng.normal(0, sd, n)
    return M, pd.Series(np.exp(log_v))


@pytest.mark.parametrize("sd", [0.3, 0.6, 1.0])
def test_residual_sd_is_recovered_from_synthetic_lognormal_data(sd):
    M, v = _synthetic_deals(sd)
    tr = pd.Series(True, index=M.index)
    model = fit_deal_value(M, v, tr)
    assert model.estimated and model.n_train == len(M)
    assert model.log_residual_sd == pytest.approx(sd, rel=0.04)
    # the corrected prediction is an unbiased estimate of the mean deal value per group
    pred = predict_deal_value(model, M)
    for k in range(3):
        grp = M[f"g_{k}"] == 1
        assert pred[grp.values].mean() == pytest.approx(v[grp].mean(), rel=0.06)


def test_fit_deal_value_uses_training_wins_only_and_a_fixed_sd_when_given():
    M, v = _synthetic_deals(0.5, n=400)
    v.iloc[::4] = np.nan  # no recorded deal value
    tr = pd.Series(np.arange(len(M)) < 300, index=M.index)
    fixed = fit_deal_value(M, v, tr, log_residual_sd=0.2)
    assert not fixed.estimated and fixed.log_residual_sd == 0.2 and fixed.n_train == int((tr & v.notna()).sum())
    assert fit_deal_value(M, v, tr).n_train == fixed.n_train


def test_residual_sd():
    assert residual_sd(np.array([1.0, -1.0, 1.0, -1.0])) == pytest.approx(np.std([1, -1, 1, -1], ddof=1))
    with pytest.raises(ValueError):
        residual_sd(np.array([1.0]))


def test_expected_value_and_realised_revenue():
    T = pd.DataFrame({"y": [1.0, 1.0, 0.0], "deal_value": [5000.0, np.nan, np.nan],
                      "deal_value_hat": [100.0, 200.0, 300.0]})
    assert expected_value(pd.Series([0.5]), pd.Series([1000.0]), 0.2).iloc[0] == 100
    assert realised_revenue(T).tolist() == [5000.0, 200.0, 0.0]
