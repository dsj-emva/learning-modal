import numpy as np
import pandas as pd
import pytest

from emva.constants import AS_OF
from emva.labels import label, split_masks


def _row(stage: str, age_days: float, idle_days: float) -> dict:
    return {"final_stage": stage, "created_at": AS_OF - pd.Timedelta(days=age_days),
            "last_change": AS_OF - pd.Timedelta(days=idle_days)}


def _label(stage: str, age_days: float, idle_days: float) -> float:
    return label(pd.DataFrame([_row(stage, age_days, idle_days)])).iloc[0]


@pytest.mark.parametrize("age,idle", [(1, 0), (400, 300)])
def test_won_is_one_regardless_of_age(age, idle):
    assert _label("Won", age, idle) == 1


@pytest.mark.parametrize("age,idle", [(1, 0), (400, 300)])
def test_lost_is_zero_regardless_of_age(age, idle):
    assert _label("Lost", age, idle) == 0


@pytest.mark.parametrize("stage", ["Contacted", "Qualified", "Demo booked", "Proposal"])
def test_stalled_boundary_at_90_idle_days(stage):
    assert np.isnan(_label(stage, 200, 89))
    assert np.isnan(_label(stage, 200, 89.99))  # .dt.days floors: 89 days 23 h is still 89
    assert _label(stage, 200, 90) == 0
    assert _label(stage, 200, 150) == 0


def test_ghosted_boundary_at_90_days_since_creation():
    assert np.isnan(_label("New", 89, 89))
    assert _label("New", 90, 90) == 0


def test_ghost_rule_uses_age_not_idle():
    # New lead created 30 days ago whose last CRM row is (impossibly) old: still unlabelled.
    assert np.isnan(_label("New", 30, 120))


def test_open_stage_rule_uses_idle_not_age():
    assert np.isnan(_label("Proposal", 500, 10))


def test_unknown_stage_is_unlabelled():
    assert np.isnan(_label(None, 500, 500))


def test_split_masks_use_test_from_and_labels():
    X = pd.DataFrame({
        "created_at": pd.to_datetime(["2026-04-30T23:59:59Z", "2026-05-01T00:00:00Z", "2026-06-01T00:00:00Z"]),
        "y": [1.0, 0.0, np.nan],
    })
    tr, te = split_masks(X)
    assert tr.tolist() == [True, False, False]
    assert te.tolist() == [False, True, False]
