"""Regularisation sweep (plan 4.6): the grid and the sweep table."""
from __future__ import annotations

import numpy as np
import pandas as pd

from emva.constants import LR_C
from emva.eval import regularisation as rg
from emva.eval import rolling as ro


def test_grid_is_a_log_grid_around_the_pipeline_c():
    g = np.array(rg.C_GRID)
    assert LR_C in rg.C_GRID
    assert (np.diff(g) > 0).all() and g[0] == 0.001 and g[-1] == 100
    assert rg.FLAT_RANGE[0] in rg.C_GRID and rg.FLAT_RANGE[1] in rg.C_GRID


def test_sweep_has_one_row_per_c_and_every_column():
    rng = np.random.default_rng(0)
    n = 3000
    D = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-(D.x - 1)))).astype(float))
    created = pd.Series(pd.Timestamp("2025-09-01", tz="UTC") + pd.to_timedelta(np.sort(rng.uniform(0, 300, n)), unit="D"))
    tr = pd.Series(np.arange(n) < 2000)
    t = rg.sweep({"frozen": (D, y, tr, ~tr)}, (ro.LEGACY_WINDOW, D, y, created), grid=(0.01, 0.5, 10.0),
                 brier_for=("frozen",))
    assert t.C.tolist() == [0.01, 0.5, 10.0]
    assert list(t.columns) == ["C", "frozen", "Brier frozen", "rolling-origin mean"]
    assert t.frozen.between(0.5, 1).all() and t["rolling-origin mean"].notna().all()
    f = rg.flatness(t, "frozen", 0.5, 10.0)
    assert f.lo <= f.hi and f.flat == (f.hi - f.lo < rg.FLAT_TOLERANCE)
    assert rg.make_lr_c(3.0).C == 3.0
