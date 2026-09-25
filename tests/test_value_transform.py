"""emva.value_transform: cap, compression, floor and tiers (plan 3.1)."""
import numpy as np
import pytest

from emva.constants import VALUE_CAP_PERCENTILE, VALUE_FLOOR_GBP
from emva.value_transform import IDENTITY, Compression, ValueTransform

REF = np.arange(1.0, 101.0)  # 1..100


def test_default_is_cap_p97_log_floor_25():
    t = ValueTransform()
    assert (t.cap_percentile, t.floor, t.compression, t.tiers) == (VALUE_CAP_PERCENTILE, VALUE_FLOOR_GBP,
                                                                   Compression.LOG, None)
    assert t.describe() == "cap p97 + log + floor £25"
    assert IDENTITY.describe() == "identity"


def test_identity_changes_nothing():
    v = np.array([0.0, 3.5, 1e6])
    assert np.array_equal(IDENTITY.fit(REF).apply(v), v)


def test_cap_at_percentile_of_the_reference():
    f = ValueTransform(cap_percentile=90, floor=None, compression="none").fit(REF)
    assert f.cap == pytest.approx(np.percentile(REF, 90))
    out = f.apply(np.array([5.0, 90.0, 95.0, 1e9]))
    assert out.tolist() == [5.0, 90.0, f.cap, f.cap]


def test_floor():
    f = ValueTransform(cap_percentile=None, floor=25, compression="none").fit(REF)
    assert f.apply(np.array([0.0, 24.9, 25.0, 80.0])).tolist() == [25.0, 25.0, 25.0, 80.0]


@pytest.mark.parametrize("compression", ["log", "sqrt"])
def test_compression_is_monotone_concave_and_keeps_the_reference_total(compression):
    t = ValueTransform(cap_percentile=None, floor=None, compression=compression)
    f = t.fit(REF)
    assert f.anchor == pytest.approx(np.median(REF))
    grid = np.linspace(0, 1000, 2001)
    out = f.apply(grid)
    assert (np.diff(out) > 0).all()             # strictly increasing
    assert (np.diff(out, 2) <= 1e-9).all()      # concave
    assert out[0] == 0
    assert f.apply(REF).sum() == pytest.approx(REF.sum())
    assert f.apply(REF).max() / np.median(f.apply(REF)) < REF.max() / np.median(REF)


def test_log_and_sqrt_formulas():
    ref = np.array([1.0, 4.0, 16.0])  # median 4
    for comp, fn in [("sqrt", lambda v: np.sqrt(v * 4)), ("log", lambda v: 4 * np.log2(1 + v / 4))]:
        f = ValueTransform(cap_percentile=None, floor=None, compression=comp).fit(ref)
        assert f.scale == pytest.approx(ref.sum() / fn(ref).sum())
        assert f.apply(np.array([9.0]))[0] == pytest.approx(fn(9.0) * f.scale)


def test_cap_then_compress_uses_the_capped_median_and_total():
    f = ValueTransform(cap_percentile=50, floor=None, compression="sqrt").fit(REF)
    capped = np.minimum(REF, f.cap)
    assert f.anchor == pytest.approx(np.median(capped))
    assert f.apply(REF).sum() == pytest.approx(capped.sum())


@pytest.mark.parametrize("n", [2, 5, 10])
def test_tiers_count_monotone_and_tier_means(n):
    rng = np.random.default_rng(0)
    ref = rng.lognormal(6, 1.5, 5000)
    f = ValueTransform(cap_percentile=None, floor=None, compression="none", tiers=n).fit(ref)
    assert len(f.tier_values) == n and len(f.tier_edges) == n - 1
    assert (np.diff(f.tier_values) > 0).all()
    out = f.apply(ref)
    assert len(np.unique(out)) == n
    assert out.sum() == pytest.approx(ref.sum())  # tier means preserve the reference total
    grid = np.sort(rng.lognormal(6, 2, 1000))
    assert (np.diff(f.apply(grid)) >= 0).all()
    low = ref <= f.tier_edges[0]
    assert f.tier_values[0] == pytest.approx(ref[low].mean())


def test_tiers_refuse_too_few_distinct_values():
    with pytest.raises(ValueError, match="empty"):
        ValueTransform(cap_percentile=None, floor=None, compression="none", tiers=5).fit(np.array([1.0] * 10 + [2.0]))


def test_one_lead_is_transformed_with_the_fitted_parameters():
    f = ValueTransform().fit(REF)
    assert f.apply(np.array([50.0])).tolist() == f.apply(np.array([1.0, 50.0, 99.0]))[1:2].tolist()


@pytest.mark.parametrize("kwargs", [{"cap_percentile": 0}, {"cap_percentile": 101}, {"floor": -1}, {"tiers": 1},
                                    {"compression": "cube"}])
def test_invalid_settings_raise(kwargs):
    with pytest.raises(ValueError):
        ValueTransform(**kwargs)


@pytest.mark.parametrize("bad", [np.array([]), np.array([1.0, np.nan]), np.array([1.0, -2.0])])
def test_invalid_reference_or_values_raise(bad):
    with pytest.raises(ValueError):
        ValueTransform().fit(bad)
    if len(bad):
        with pytest.raises(ValueError):
            ValueTransform().fit(REF).apply(bad)
