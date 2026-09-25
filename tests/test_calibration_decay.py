"""Calibration decay (plan 4.5): fit/evaluation windows and the calibration statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from emva.eval import calibration_decay as cd
from emva.eval import rolling as ro

AS_OF = pd.Timestamp("2026-06-15", tz="UTC")


def test_decay_windows_follow_the_rolling_rules():
    H = 10
    cut = pd.Timestamp("2026-02-01", tz="UTC")
    t = [cut - pd.Timedelta(days=H), cut - pd.Timedelta(days=H) + pd.Timedelta(seconds=1),
         cut, pd.Timestamp("2026-03-31T23:59:59", tz="UTC"), pd.Timestamp("2026-04-30", tz="UTC"),
         pd.Timestamp("2026-05-01", tz="UTC"), pd.Timestamp("2026-06-10", tz="UTC")]
    idx = [f"L{i}" for i in range(len(t))]
    created, y = pd.Series(t, index=idx), pd.Series([1, 0, 1, 0, 1, 0, 1], index=idx, dtype=float)
    train, evals = cd.decay_windows(ro.RollingSpec("s", H), created, y, "2026-01", as_of=AS_OF)
    assert train.tolist() == [True, False, False, False, False, False, False]   # mature by 2026-02-01 only
    assert [m for m, _ in evals] == ["2026-02", "2026-03", "2026-04"]
    assert [mask.tolist().index(True) if mask.any() else None for _, mask in evals] == [2, 3, 4]
    assert [int(mask.sum()) for _, mask in evals] == [1, 1, 1]   # 2026-05-01 is M+4; 2026-06-10 is not mature
    with pytest.raises(ValueError):
        cd.decay_windows(ro.LEGACY_TO_END, created, y, "2026-01", as_of=AS_OF)


def test_calibration_stats_detect_level_and_spread():
    rng = np.random.default_rng(0)
    z = rng.normal(-1.5, 1.0, 40000)
    p = 1 / (1 + np.exp(-z))
    y = (rng.random(len(p)) < p).astype(float)
    good = cd.calibration_stats(y, p)
    assert good.slope == pytest.approx(1, abs=0.05) and good.intercept == pytest.approx(0, abs=0.05)
    assert good.mean_p == pytest.approx(good.observed, abs=0.01) and good.n == 40000
    over = cd.calibration_stats(y, 1 / (1 + np.exp(-(z + 0.5))))      # predicts too high
    assert over.intercept == pytest.approx(-0.5, abs=0.05)
    sharp = cd.calibration_stats(y, 1 / (1 + np.exp(-2 * z)))         # over-confident: slope ~ 0.5
    assert sharp.slope == pytest.approx(0.5, abs=0.05)
    empty = cd.calibration_stats(np.array([]), np.array([]))
    assert empty.n == 0 and np.isnan(empty.brier)
    one_class = cd.calibration_stats(np.zeros(5), np.full(5, 0.1))
    assert np.isnan(one_class.slope) and one_class.wins == 0


def test_calibration_decay_runs_and_tables():
    rng = np.random.default_rng(1)
    n = 4000
    D = pd.DataFrame({"x": rng.normal(size=n)})
    created = pd.Series(pd.Timestamp("2025-10-01", tz="UTC") + pd.to_timedelta(np.sort(rng.uniform(0, 250, n)), unit="D"))
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-(1.5 * D.x - 1.5)))).astype(float))
    res = cd.calibration_decay(ro.LEGACY_WINDOW, D, y, created, months=("2026-01", "2026-05"), as_of=AS_OF)
    assert [r.month for r in res] == ["2026-01", "2026-05"]
    assert [m for m, _ in res[0].evals] == ["2026-02", "2026-03", "2026-04"]
    assert all(s.n > 0 for _, s in res[0].evals)
    assert [s.n for _, s in res[1].evals] == [int(((created >= f"2026-0{k}-01") & (created < f"2026-0{k + 1}-01")).sum())
                                              for k in (6, 7, 8)]
    t = cd.decay_table(res)
    assert "eval month" in t and (t["fit through"] == "2026-01").sum() == 3
    assert len(cd.decile_table(res)) == 10
