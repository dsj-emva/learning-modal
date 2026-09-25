"""Interaction detectability: the product columns, their coefficient CIs, and the simulated sampling distribution."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from emva.eval import interactions as it


def test_interaction_columns_use_the_pipeline_features():
    X = pd.DataFrame({"channel": ["linkedin", "linkedin", "google", "linkedin"],
                      "band": ["51-200", "11-50", "1000+", "missing"],
                      "seniority": ["senior", "senior", "mid", "senior"],
                      "form_variant": ["D", "A", "D", "D"]})
    c = it.interaction_columns(X)
    assert c.linkedin_x_51plus.tolist() == [1, 0, 0, 0]
    assert c.senior_x_formD.tolist() == [1, 0, 0, 1]
    D = it.with_interactions(pd.DataFrame({"a": [0.0] * 4}, index=X.index), X)
    assert list(D.columns) == ["a", *it.INTERACTION_COLUMNS]


def test_a_planted_interaction_is_recovered():
    rng = np.random.default_rng(0)
    n = 6000
    X = pd.DataFrame({"channel": rng.choice(["linkedin", "google"], n), "band": rng.choice(["1-10", "51-200"], n),
                      "seniority": rng.choice(["senior", "junior/ic"], n), "form_variant": rng.choice(["A", "D"], n)})
    D0 = pd.DataFrame({"li": X.channel.eq("linkedin").astype(float), "big": X.band.eq("51-200").astype(float),
                       "sen": X.seniority.eq("senior").astype(float), "d": X.form_variant.eq("D").astype(float)})
    DI = it.with_interactions(D0, X)
    z = -1 + 0.3 * D0.li + 0.5 * D0.big + 1.5 * DI.linkedin_x_51plus
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-z))).astype(float))
    cis = it.interaction_coef_cis(DI, y, pd.Series(True, index=X.index), n_resamples=50)
    assert cis["linkedin_x_51plus"].lo > 0.8                  # planted +1.5, CI excludes zero
    assert cis["senior_x_formD"].lo < 0 < cis["senior_x_formD"].hi   # not planted


def _null_truth(n: int, p: float, stalled: bool = False, slow: bool = False) -> it.NullTruth:
    return it.NullTruth(np.full(n, p), np.full(n, slow), np.full(n, stalled), it.TimeToWin(40.0, 1.09, 1.8))


def test_simulated_labels_follow_the_mechanism():
    rng = np.random.default_rng(0)
    assert it.simulate_horizon_labels(_null_truth(1000, 1.0, stalled=True), 120, rng).sum() == 0   # stalled: never won
    assert it.simulate_horizon_labels(_null_truth(1000, 0.0), 120, rng).sum() == 0
    fast = it.simulate_horizon_labels(_null_truth(20000, 1.0), 120, rng).mean()
    slow = it.simulate_horizon_labels(_null_truth(20000, 1.0, slow=True), 120, rng).mean()
    # P(lognormal(ln 40, 1.09) <= 120) = 0.843; x1.8 for true band 1000+: P(<= 120 / 1.8) = 0.681
    assert abs(fast - 0.843) < 0.01 and abs(slow - 0.681) < 0.01
    days = it.TimeToWin(1.0, 3.0, 1.0).draw(np.zeros(5000, dtype=bool), rng)
    assert days.min() >= it.CLOSE_DAYS_MIN and days.max() <= it.CLOSE_DAYS_MAX


def test_null_coefficients_on_a_tiny_frame(tmp_path):
    rng = np.random.default_rng(1)
    n = 800
    idx = [f"L{i}" for i in range(n)]
    D = pd.DataFrame({"a": rng.integers(0, 2, n), "linkedin_x_51plus": rng.integers(0, 2, n),
                      "senior_x_formD": rng.integers(0, 2, n)}, index=idx).astype(float)
    p = 1 / (1 + np.exp(-(-1 + D.a)))        # no interaction planted
    pd.DataFrame({"lead_id": idx, "p_close_true": p.round(4).values, "employee_band": "1-10", "stalled": False,
                  "outcome": "lost"}).to_csv(tmp_path / "ground_truth_labels.csv", index=False)
    (tmp_path / "ground_truth.md").write_text("Won deals take a lognormal number of days (median 40, sd 1.09); 1000+\n"
                                              "companies take 1.8x longer.\n")
    truth = it.read_null_truth(tmp_path, D.index)
    assert truth.time_to_win == it.TimeToWin(40.0, 1.09, 1.8) and not truth.stalled.any()
    y = pd.Series(it.simulate_horizon_labels(truth, 120, np.random.default_rng(9)), index=D.index)
    train = pd.Series(np.arange(n) < 600, index=D.index)
    nd = it.null_coefficients(D, y, train, truth, 120, n_draws=40, seed=3)
    assert nd.draws.shape == (40, 2) and nd.columns == it.INTERACTION_COLUMNS
    assert np.array_equal(nd.draws, it.null_coefficients(D, y, train, truth, 120, n_draws=40, seed=3).draws)
    s = nd.summary()
    assert list(s.term) == list(it.INTERACTION_COLUMNS) and (s.sim_lo < s.sim_hi).all()
    assert s.sim_mean.abs().max() < 0.3 and s.share_as_extreme.between(0, 1).all()
    assert abs(nd.sim_rate - nd.observed_rate) < 0.05
    with pytest.raises(ValueError, match="no ground-truth row"):
        it.read_null_truth(tmp_path, pd.Index(["missing"]))
