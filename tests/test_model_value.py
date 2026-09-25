import numpy as np
import pandas as pd
import pytest

from emva.model import scorecard
from emva.value import DEAL_LOG_RESIDUAL_SD_FROM_GENERATOR, expected_value, predict_deal_value, realised_revenue


class _FakeLR:
    coef_ = np.array([[np.log(2), -0.5, 0.0]])


def test_scorecard_points_and_order():
    w = scorecard(_FakeLR(), pd.Index(["a", "b", "c"]))
    assert w.index.tolist() == ["b", "c", "a"]
    assert w.loc["a", "points"] == 20 and w.loc["a", "odds_multiplier"] == 2.0
    assert w.loc["b", "log_odds"] == -0.5


class _FakeRidge:
    def predict(self, M):
        return np.log(np.full(len(M), 1000.0))


def test_deal_value_lognormal_correction():
    got = predict_deal_value(_FakeRidge(), pd.DataFrame({"x": [1.0, 2.0]}))
    assert got == pytest.approx(1000 * np.exp(DEAL_LOG_RESIDUAL_SD_FROM_GENERATOR ** 2 / 2))


def test_expected_value_and_realised_revenue():
    T = pd.DataFrame({"y": [1.0, 1.0, 0.0], "deal_value": [5000.0, np.nan, np.nan],
                      "deal_value_hat": [100.0, 200.0, 300.0]})
    assert expected_value(pd.Series([0.5]), pd.Series([1000.0]), 0.2).iloc[0] == 100
    assert realised_revenue(T).tolist() == [5000.0, 200.0, 0.0]
