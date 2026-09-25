"""Subsampling curve and customer-sized test set (plan 4.2, 4.3): shapes, determinism, GBDT design and constraints."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from emva.constants import V2_LEVELS
from emva.eval import subsampling as ss
from emva.features import CATS_V2


def _v2_features(n: int = 40, seed: int = 0) -> pd.DataFrame:
    """Random valid values of every v2 model feature (plus the indicators the GBDT reads)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f: rng.choice(V2_LEVELS[f], n) for f in CATS_V2})


def test_subsample_indices_are_deterministic_and_sized():
    pool = np.arange(100, 200)
    a, b = ss.subsample_indices(pool, 30, 7), ss.subsample_indices(pool, 30, 7)
    assert np.array_equal(a, b) and len(a) == 30 and len(set(a)) == 30 and set(a) <= set(pool)
    assert not np.array_equal(a, ss.subsample_indices(pool, 30, 8))
    assert np.array_equal(ss.subsample_indices(pool, None, 3), pool)
    with pytest.raises(ValueError):
        ss.subsample_indices(pool, 101, 0)


def _toy(n: int = 400, seed: int = 1) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    D = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-2 * D.x))).astype(float))
    train = pd.Series(np.arange(n) < 300)
    return D, y, train, ~train


def test_subsample_curve_shape_and_seed_determinism():
    D, y, train, test = _toy()
    models = {"LR": ss.lr_model(D, y), "LR again": ss.lr_model(D, y)}
    long = ss.subsample_curve(y, train, test, models, sizes=(50, 100, None, 1000), n_seeds=3)
    # 2 sizes × 3 seeds + one full-pool fit, for each of 2 models; 1000 > pool is skipped
    assert len(long) == 2 * (3 + 3 + 1)
    assert set(long.N) == {50, 100, 300} and long[long.full].N.unique().tolist() == [300]
    assert long.equals(ss.subsample_curve(y, train, test, models, sizes=(50, 100, None, 1000), n_seeds=3))
    lr, again = long[long.model == "LR"].auc.values, long[long.model == "LR again"].auc.values
    assert np.array_equal(lr, again)   # same rows for every model within a seed
    t = ss.curve_table(long)
    assert t.N.tolist() == ["50", "100", "300 (full)"] and list(t.columns) == ["N", "LR", "LR again"]
    assert "±" in t.LR[0] and "±" not in t.LR[2]


def test_constraint_vector_matches_the_gbdt_design_columns():
    X = _v2_features()
    g = ss.gbdt_design(X)
    assert list(g.matrix.columns) == list(CATS_V2)
    assert g.monotone == ss.constraint_vector()
    assert len(g.monotone) == g.matrix.shape[1] == len(g.categorical)
    for col, m, cat in zip(g.matrix.columns, g.monotone, g.categorical):
        assert (m == 1) == (col in ss.ORDERED_LEVELS) == (not cat)
    assert {c for c, m in zip(g.matrix.columns, g.monotone) if m} == {"band", "spend", "time_on_page", "seniority"}
    assert ss.constraint_table().constraint.tolist().count("+1") == 4


def test_gbdt_design_ranks_and_absent_values():
    X = _v2_features(6)
    X["band"] = ["1-10", "11-50", "51-200", "201-1000", "1000+", "missing"]
    X["seniority"] = ["student", "junior/ic", "mid", "senior", "not_asked/blank", "senior"]
    X["enrichment_missing"] = ["no", "yes", "no", "no", "no", "no"]
    X["spend"] = ["none", "under £5k", "£5k-£25k", "£25k-£100k", "£100k+", "under £5k"]
    X["session_missing"] = ["yes", "no", "no", "no", "no", "no"]
    X["time_on_page"] = ["15-60s", "15-60s", "60-300s", "300-600s", ">600s", "15-60s"]
    m = ss.gbdt_design(X).matrix
    assert m.band.tolist()[:5] == [0, 1, 2, 3, 4] and np.isnan(m.band[5])
    assert m.seniority.tolist()[:4] == [0, 1, 2, 3] and np.isnan(m.seniority[4])
    assert np.isnan(m.spend[1]) and m.spend[0] == 0 and m.spend[4] == 4     # no enrichment -> NaN
    assert np.isnan(m.time_on_page[0]) and m.time_on_page[4] == 3           # no session -> NaN
    X.loc[0, "band"] = "huge"
    with pytest.raises(ValueError, match="band"):
        ss.gbdt_design(X)


def test_gbdt_models_fit_and_score():
    X = _v2_features(300, seed=3)
    y = pd.Series((X.band.isin(["51-200", "201-1000"]) | (np.arange(300) % 7 == 0)).astype(float))
    models = ss.gbdt_models(X, y)
    tr, te = np.arange(200), np.arange(200, 300)
    for fit_score in models.values():
        s = fit_score(tr, te)
        assert s.shape == (100,) and ((0 <= s) & (s <= 1)).all()


def test_customer_sized_draws_are_deterministic_with_a_20pct_holdout():
    rng = np.random.default_rng(0)
    n = 1500
    D = pd.DataFrame({"x": rng.normal(size=n)})
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-(2 * D.x - 1)))).astype(float))
    created = pd.Series(pd.Timestamp("2026-01-01", tz="UTC") + pd.to_timedelta(rng.uniform(0, 200, n), unit="D"))
    draws = ss.customer_sized(D, y, pd.Series(True, index=D.index), created, n=500, holdout=0.2, draws=3, n_resamples=50)
    assert [d.seed for d in draws] == [0, 1, 2]
    assert all(d.n_test == 100 and 0 < d.width < 1 and 0.5 < d.auc <= 1 for d in draws)
    again = ss.customer_sized(D, y, pd.Series(True, index=D.index), created, n=500, holdout=0.2, draws=3, n_resamples=50)
    assert draws == again
