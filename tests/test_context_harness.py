"""emva.eval.context_harness: sample design, cross-fitting, power curve and gap-recovered ratio (synthetic frames)."""
import numpy as np
import pandas as pd
import pytest

from emva.eval.context_harness import (
    crossfit,
    folds,
    gap_recovered,
    oracle_power_curve,
    sample_leads,
    true_probability_gap,
)


def test_sample_is_seeded_stratified_and_labelled_only():
    idx = pd.Index([f"L{i:04d}" for i in range(400)])
    labelled = pd.Series(np.arange(400) % 4 != 0, index=idx)          # 300 labelled
    persona = pd.Series(np.where(np.arange(400) % 5 == 0, "competitor", "none"), index=idx)
    ids = sample_leads(labelled, persona, n=100, n_persona=30, seed=0)
    assert len(ids) == 100 and ids.is_unique and ids.is_monotonic_increasing
    assert labelled[ids].all() and (persona[ids] != "none").sum() == 30
    assert ids.equals(sample_leads(labelled, persona, n=100, n_persona=30, seed=0))
    assert not ids.equals(sample_leads(labelled, persona, n=100, n_persona=30, seed=1))
    with pytest.raises(ValueError, match="need"):
        sample_leads(labelled, persona, n=100, n_persona=80)


def synthetic(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    D = rng.normal(size=(n, 3))
    persona = rng.random(n) < 0.3
    logit = -1 + D @ np.array([1.0, -0.5, 0.3]) - 3.0 * persona
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    return D, y, persona


def test_crossfit_predicts_every_row_out_of_fold():
    D, y, _ = synthetic(600)
    splits = folds(y, n_folds=5, seed=0)
    held = np.concatenate([h for _, h in splits])
    assert sorted(held) == list(range(600))
    p = crossfit(D, y, splits)
    assert p.shape == (600,) and ((p > 0) & (p < 1)).all()


def test_power_curve_falls_with_noise_on_a_tiny_synthetic_frame():
    D, y, persona = synthetic()
    curve, ref = oracle_power_curve(D, y, persona, sds=(0.25, 4.0), n_draws=3, effect=-3.0, seed=0)
    assert list(curve.noise_sd) == [0.25, 4.0]
    assert ref["gap"] > 0.02 and ref["auc_oracle"] == pytest.approx(ref["auc_formula"] + ref["gap"])
    low, high = curve.frac_mean
    assert 0.8 < low <= 1.05 and high < low
    assert (curve.frac_p05 <= curve.frac_mean).all() and (curve.frac_mean <= curve.frac_p95).all()


def test_gap_recovered_is_one_when_context_equals_oracle_and_zero_when_it_equals_formula():
    D, y, persona = synthetic(800)
    splits = folds(y)
    p_f = crossfit(D, y, splits)
    p_o = crossfit(np.column_stack([D, persona]), y, splits)
    assert gap_recovered(y, p_f, p_o, p_o, n_resamples=50).point == pytest.approx(1.0)
    same = gap_recovered(y, p_f, p_f, p_o, n_resamples=50)
    assert same.point == 0 and same.lo == 0 and same.hi == 0


def test_true_probability_gap_measures_the_planted_term():
    D, y, persona = synthetic()
    p = 1 / (1 + np.exp(-(-1 + D @ np.array([1.0, -0.5, 0.3]) - 3.0 * persona)))
    g = true_probability_gap(y, p, persona, effect=-3.0)
    assert g["true_gap"] > 0.02 and g["auc_true"] == pytest.approx(g["auc_true_without_persona"] + g["true_gap"])
    assert true_probability_gap(y, p, np.zeros(len(y)))["true_gap"] == 0


def test_coefficient_criterion_needs_significance_and_agreement_with_the_oracle():
    from emva.eval.bootstrap import BootstrapCI
    from emva.eval.context_harness import coefficient_criterion

    def ci(point, lo, hi):
        return BootstrapCI(point=point, lo=lo, hi=hi, samples=np.array([]))

    oracle = ci(-0.61, -1.02, -0.21)
    assert coefficient_criterion(ci(-0.61, -1.07, -0.18), oracle)
    assert not coefficient_criterion(ci(-0.40, -0.90, 0.10), oracle)     # CI includes zero
    assert not coefficient_criterion(ci(-1.50, -2.00, -1.10), oracle)    # significant but outside the oracle CI
    assert not coefficient_criterion(ci(0.50, 0.10, 0.90), oracle)       # wrong sign
