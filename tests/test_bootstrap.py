import numpy as np
import pytest

from sklearn.linear_model import LogisticRegression

from emva.eval.bootstrap import auc_ci, coef_bootstrap, paired_auc


def _synthetic(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n).astype(float)
    return y, y + rng.normal(0, 1.5, n)


def test_ci_contains_point_estimate():
    y, s = _synthetic(500)
    ci = auc_ci(y, s, n_resamples=300)
    assert ci.lo <= ci.point <= ci.hi
    assert len(ci.samples) == 300


def test_ci_width_shrinks_with_n():
    small = auc_ci(*_synthetic(200), n_resamples=300)
    large = auc_ci(*_synthetic(5000), n_resamples=300)
    assert large.width < small.width / 2


def test_seed_makes_it_deterministic():
    y, s = _synthetic(300)
    assert auc_ci(y, s, n_resamples=100, seed=7).lo == auc_ci(y, s, n_resamples=100, seed=7).lo


def test_paired_identical_scores_have_zero_difference():
    y, s = _synthetic(300)
    c = paired_auc(y, s, s, n_resamples=200)
    assert c.diff.point == 0 and c.diff.lo == 0 and c.diff.hi == 0 and c.p_value == 1.0


def test_paired_detects_a_clearly_better_score():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000).astype(float)
    good, noise = y + rng.normal(0, 1, 2000), rng.normal(0, 1, 2000)
    c = paired_auc(y, noise, good, n_resamples=300)
    assert c.diff.lo > 0 and c.p_value < 0.01
    assert c.auc_b - c.auc_a == pytest.approx(c.diff.point)


def test_resamples_with_one_class_are_redrawn():
    y = np.array([1.0] + [0.0] * 29)  # most resamples miss the only positive
    ci = auc_ci(y, np.arange(30.0), n_resamples=50)
    assert len(ci.samples) == 50


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        auc_ci(np.zeros(10), np.arange(10.0))
    with pytest.raises(ValueError):
        auc_ci(np.array([0.0, 1.0]), np.array([0.1, np.nan]))


def _lr_coef(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return LogisticRegression(C=1e6, max_iter=1000).fit(X, y).coef_[0]


def test_coef_bootstrap_covers_planted_coefficients():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(3000, 2))
    y = (rng.uniform(size=3000) < 1 / (1 + np.exp(-(1.0 * X[:, 0] + 0.0 * X[:, 1])))).astype(float)
    cis = coef_bootstrap(X, y, _lr_coef, n_resamples=200, seed=1)
    assert len(cis) == 2 and all(len(c.samples) == 200 for c in cis)
    assert cis[0].lo < 1.0 < cis[0].hi and cis[1].lo < 0.0 < cis[1].hi
    assert cis[0].lo > 0.5  # clearly non-zero
    assert all(c.lo <= c.point <= c.hi for c in cis)


def test_coef_bootstrap_is_seeded_and_checks_input():
    rng = np.random.default_rng(4)
    X, y = rng.normal(size=(200, 1)), rng.integers(0, 2, 200).astype(float)
    assert coef_bootstrap(X, y, _lr_coef, n_resamples=20, seed=5)[0].lo == \
        coef_bootstrap(X, y, _lr_coef, n_resamples=20, seed=5)[0].lo
    with pytest.raises(ValueError):
        coef_bootstrap(X[:10], y, _lr_coef, n_resamples=5)
    with pytest.raises(ValueError):
        coef_bootstrap(X, np.zeros(200), _lr_coef, n_resamples=5)
