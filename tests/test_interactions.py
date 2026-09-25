"""Interaction detectability: the two product columns and their coefficient CIs."""
from __future__ import annotations

import numpy as np
import pandas as pd

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
