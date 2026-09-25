"""Rolling-origin evaluation (plan 4.1, ADR 0013): split membership rules, headline, and the v1 baseline spread."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from emva.eval import rolling as ro

S = pd.Timestamp("2026-03-01", tz="UTC")
AS_OF = pd.Timestamp("2026-04-15", tz="UTC")
H = 10
SEC = pd.Timedelta(seconds=1)


def _frame(times: list[pd.Timestamp], y: list[float]) -> tuple[pd.Series, pd.Series]:
    idx = [f"L{i}" for i in range(len(times))]
    return pd.Series(times, index=idx), pd.Series(y, index=idx, dtype=float)


def test_strict_training_needs_maturity_at_the_split():
    t = [S - pd.Timedelta(days=H),            # created + H == S: mature at S -> train
         S - pd.Timedelta(days=H) + SEC,      # mature 1 s after S -> not in strict training
         S - SEC]                             # before S, far from mature at S
    created, y = _frame(t, [1, 0, 1])
    strict, _ = ro.RollingSpec("s", H, strict=True).masks(created, y, S, AS_OF)
    relaxed, _ = ro.RollingSpec("r", H, strict=False).masks(created, y, S, AS_OF)
    assert strict.tolist() == [True, False, False]
    assert relaxed.tolist() == [True, True, True]  # all mature at AS_OF


def test_test_window_and_maturity_at_as_of():
    end = S + pd.DateOffset(months=1)
    t = [S - SEC, S, end - SEC, end, AS_OF - pd.Timedelta(days=H), AS_OF - pd.Timedelta(days=H) + SEC]
    created, y = _frame(t, [0, 1, 0, 1, 0, 1])
    _, test = ro.RollingSpec("s", H).masks(created, y, S, AS_OF)
    # S: in window, mature. end - 1 s: in window, created + H = 2026-04-11 <= AS_OF. end: outside the window.
    # AS_OF - H is 2026-04-05, outside the window; nothing after the window counts.
    assert test.tolist() == [False, True, True, False, False, False]
    late = pd.Timestamp("2026-03-20", tz="UTC")
    created2, y2 = _frame([late - SEC, AS_OF - pd.Timedelta(days=H) + SEC], [1, 1])
    _, test2 = ro.RollingSpec("s", H).masks(created2, y2, pd.Timestamp("2026-03-15", tz="UTC"), AS_OF)
    assert test2.tolist() == [True, False]  # the second is in the window but not mature at AS_OF


def test_unlabelled_rows_are_never_used_and_legacy_has_no_maturity_rule():
    created, y = _frame([S - pd.Timedelta(days=1), S, S + pd.Timedelta(days=1)], [np.nan, 1, np.nan])
    tr, te = ro.LEGACY_WINDOW.masks(created, y, S, AS_OF)
    assert tr.tolist() == [False, False, False] and te.tolist() == [False, True, False]
    created, y = _frame([S - pd.Timedelta(hours=1), S + pd.Timedelta(days=60)], [0, 1])
    tr, te = ro.LEGACY_TO_END.masks(created, y, S, AS_OF)
    assert tr.tolist() == [True, False] and te.tolist() == [False, True]   # to-end window has no upper bound
    _, te1 = ro.LEGACY_WINDOW.masks(created, y, S, AS_OF)
    assert te1.tolist() == [False, False]


def _synthetic(n: int = 3000, seed: int = 0) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    start = pd.Timestamp("2025-09-01", tz="UTC")
    created = pd.Series(start + pd.to_timedelta(np.sort(rng.uniform(0, 330, n)), unit="D"))
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-(2 * x - 1)))).astype(float))
    return pd.DataFrame({"x": x}), y, created


def test_rolling_origin_reports_every_split_and_flags_small_ones():
    D, y, created = _synthetic()
    splits = [pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2026-02-01", tz="UTC"),
              pd.Timestamp("2026-07-25", tz="UTC")]
    res = ro.rolling_origin(ro.LEGACY_WINDOW, D, y, created, splits=splits, n_resamples=50)
    assert [r.split for r in res] == splits
    assert res[0].status().startswith("no fit") and res[0].auc is None      # no lead before the first split
    assert res[1].status() == "ok" and res[1].ci.lo <= res[1].auc <= res[1].ci.hi
    assert res[2].n_test < ro.MIN_TEST_LEADS and not res[2].sufficient      # window cut by the end of the data
    h = ro.headline(res, n_resamples=50)
    assert h.n_splits == 1 and h.mean == pytest.approx(res[1].auc) and h.lo <= h.mean <= h.hi
    assert ro.mean_auc(res) == pytest.approx(res[1].auc)
    table = ro.results_table(res)
    assert len(table) == 3 and table.status.tolist()[1] == "ok"


def test_headline_none_and_paired_headline_of_identical_models_is_zero():
    D, y, created = _synthetic()
    res = ro.rolling_origin(ro.LEGACY_WINDOW, D, y, created, splits=ro.SPLITS[:2], n_resamples=30)
    assert ro.headline([r for r in res if False]) is None
    d = ro.paired_headline(res, res, n_resamples=30)
    assert d.point == 0 and d.lo == 0 and d.hi == 0


def test_v1_baseline_spread_is_reproduced(data_v1):
    # The review's baseline spread 0.814 to 0.836: legacy labels and features, test = every lead from S.
    F = ro.prepare(data_v1)
    res = ro.rolling_origin(ro.LEGACY_TO_END, F.D_legacy, F.legacy_y, F.X.created_at, with_ci=False)
    aucs = [round(r.auc, 3) for r in res]
    assert min(aucs) == 0.814 and max(aucs) == 0.836
    assert aucs[3] == 0.814   # S = 2026-05-01 is the frozen legacy test set: the baseline's AUC
    tr, te = F.frozen_horizon()
    assert int(te.sum()) == 458   # the frozen mature test set (b)
