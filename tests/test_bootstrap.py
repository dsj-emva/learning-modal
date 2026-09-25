import numpy as np
import pytest

from emva.eval.bootstrap import auc_ci, paired_auc


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
