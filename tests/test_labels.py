import json

import numpy as np
import pandas as pd
import pytest

from emva.constants import AS_OF, HORIZON_DAYS, LABEL_SOURCES
from emva.io import load
from emva.labels import (
    HORIZON,
    LEGACY,
    LabelConfig,
    LabelMode,
    assign_labels,
    ghosted_at_horizon,
    horizon_label,
    is_mature,
    label,
    label_source,
    matured_at,
    split_masks,
    won_within_h,
)


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


# ---------------------------------------------------------------------------------------------
# Phase 1: label_source and the fixed-horizon label, on hand-built CRM histories run through
# emva.io.load (so won_at / first_contact_at come from the real CRM parsing).
# AS_OF = 2026-09-24T00:00Z and H = 120 days, so a lead is mature iff created <= 2026-05-27T00:00Z.
# ---------------------------------------------------------------------------------------------
H = pd.Timedelta(days=HORIZON_DAYS)
MATURE = pd.Timestamp("2026-03-01T10:00:00Z")          # matured_at 2026-06-29, well before AS_OF
YOUNG = pd.Timestamp("2026-07-01T10:00:00Z")           # matured_at 2026-10-29, after AS_OF
EDGE = AS_OF - H                                        # 2026-05-27T00:00Z: mature exactly at AS_OF
d = pd.Timedelta(days=1)

# lead_id -> (created_at, [(stage, changed_at), ...])
HISTORIES: dict[str, tuple[pd.Timestamp, list[tuple[str, pd.Timestamp]]]] = {
    "won_at_h": (MATURE, [("New", MATURE), ("Contacted", MATURE + d), ("Closed Won", MATURE + H)]),
    "won_at_h_plus_1d": (MATURE, [("New", MATURE), ("Contacted", MATURE + d), ("won", MATURE + H + d)]),
    "won_at_h_plus_1s": (MATURE, [("New", MATURE), ("Contacted", MATURE + d),
                                  ("Won", MATURE + H + pd.Timedelta(seconds=1))]),
    "won_early": (MATURE, [("New", MATURE), ("Qualified", MATURE + d), ("Won", MATURE + 10 * d)]),
    "lost_mature": (MATURE, [("New", MATURE), ("Contacted", MATURE + d), ("Closed Lost", MATURE + 5 * d)]),
    "lost_after_h": (MATURE, [("New", MATURE), ("Contacted", MATURE + d), ("lost", MATURE + H + 30 * d)]),
    "ghosted_mature": (MATURE, [("New", MATURE)]),
    "contacted_after_h_then_lost": (MATURE, [("New", MATURE), ("Contacted", MATURE + H + d),
                                             ("Lost", MATURE + H + 5 * d)]),
    "contacted_exactly_at_h_then_lost": (MATURE, [("New", MATURE), ("Contacted", MATURE + H),
                                                  ("Lost", MATURE + H + 5 * d)]),
    "stalled_mature_90d": (MATURE, [("New", MATURE), ("Proposal", AS_OF - 90 * d)]),
    "open_mature_89d": (MATURE, [("New", MATURE), ("Demo booked", AS_OF - 89 * d)]),
    "open_mature_89_99d": (MATURE, [("New", MATURE),
                                    ("Qualified", AS_OF - 90 * d + pd.Timedelta(minutes=15))]),
    "young_open": (YOUNG, [("New", YOUNG), ("Contacted", YOUNG + d)]),
    "young_won": (YOUNG, [("New", YOUNG), ("Contacted", YOUNG + d), ("Won", YOUNG + 20 * d)]),
    "young_lost": (YOUNG, [("New", YOUNG), ("Contacted", YOUNG + d), ("Lost", YOUNG + 3 * d)]),
    "young_new": (YOUNG, [("New", YOUNG)]),
    "edge_mature_lost": (EDGE, [("New", EDGE), ("Contacted", EDGE + d), ("Lost", EDGE + 2 * d)]),
    "edge_plus_1s_lost": (EDGE + pd.Timedelta(seconds=1),
                          [("New", EDGE + pd.Timedelta(seconds=1)), ("Contacted", EDGE + d), ("Lost", EDGE + 2 * d)]),
    "young_stalled": (AS_OF - 100 * d, [("New", AS_OF - 100 * d), ("Contacted", AS_OF - 95 * d)]),
    "edge_ghosted": (EDGE, [("New", EDGE)]),
    "contacted_after_h_still_open": (MATURE, [("New", MATURE), ("Contacted", MATURE + H + d)]),
}

# lead_id -> (label_source, won_within_h, y default, y --include-ghosted, y --stalled-as-lost, legacy y)
N = np.nan
EXPECTED: dict[str, tuple[str, float, float, float, float, float]] = {
    "won_at_h": ("won", 1, 1, 1, 1, 1),
    "won_at_h_plus_1d": ("won", 0, 0, 0, 0, 1),
    "won_at_h_plus_1s": ("won", 0, 0, 0, 0, 1),
    "won_early": ("won", 1, 1, 1, 1, 1),
    "lost_mature": ("crm_lost", 0, 0, 0, 0, 0),
    "lost_after_h": ("crm_lost", 0, 0, 0, 0, 0),
    "ghosted_mature": ("ghosted", 0, N, 0, N, 0),
    "contacted_after_h_then_lost": ("crm_lost", 0, N, 0, N, 0),      # still New at H: ghosted at H
    "contacted_exactly_at_h_then_lost": ("crm_lost", 0, 0, 0, 0, 0),  # contacted at H: not ghosted
    "stalled_mature_90d": ("stalled", 0, N, N, 0, 0),
    "open_mature_89d": ("open", 0, 0, 0, 0, N),
    "open_mature_89_99d": ("open", 0, 0, 0, 0, N),                   # whole days: 89 d 23 h 45 m is 89
    "young_open": ("open", N, N, N, N, N),
    "young_won": ("won", 1, 1, 1, 1, 1),                              # already Won before H: known 1
    "young_lost": ("crm_lost", N, N, N, N, 0),                        # young, not Won: unlabelled, never Lost
    "young_new": ("open", N, N, N, N, N),                             # New but younger than H: not ghosted yet
    "contacted_after_h_still_open": ("ghosted", 0, N, 0, N, N),       # not contacted by H beats open
    "edge_mature_lost": ("crm_lost", 0, 0, 0, 0, 0),
    "edge_plus_1s_lost": ("crm_lost", N, N, N, N, 0),
    "young_stalled": ("stalled", N, N, N, N, 0),                      # flags never label a young lead
    "edge_ghosted": ("ghosted", 0, N, 0, N, 0),
}


@pytest.fixture(scope="module")
def crm_leads(tmp_path_factory) -> pd.DataFrame:
    """Write HISTORIES as the three CSVs ``load`` reads and load them."""
    root = tmp_path_factory.mktemp("crm")
    ids = list(HISTORIES)
    pd.DataFrame({"lead_id": ids, "created_at": [HISTORIES[i][0].isoformat() for i in ids],
                  "answers": [json.dumps({"country": "UK"})] * len(ids),
                  "company_domain": ["none.example"] * len(ids)}).to_csv(root / "historical_leads.csv", index=False)
    rows = [{"lead_id": i, "stage": s, "deal_value": 1000 if s.lower() in ("won", "closed won") else None,
             "changed_at": t.isoformat()} for i in ids for s, t in HISTORIES[i][1]]
    pd.DataFrame(rows).sample(frac=1, random_state=0).to_csv(root / "crm_history.csv", index=False)  # load sorts
    pd.DataFrame({"company_name": ["Acme Ltd"], "domain": ["acme.example"], "sector": ["Software"], "employee_band": ["51-200"],
                  "monthly_ad_spend_band": ["none"], "crm_platform": ["HubSpot"], "is_hiring": [False]}
                 ).to_csv(root / "companies.csv", index=False)
    return load(root)


def _assert_same(got: pd.Series, i: int) -> None:
    """Compare ``got`` with column ``i`` of EXPECTED, treating NaN == NaN, and list every mismatch."""
    want = pd.Series({k: v[i] for k, v in EXPECTED.items()}).reindex(got.index)
    mismatches = {k: (got[k], want[k]) for k in got.index
                  if not (pd.isna(got[k]) and pd.isna(want[k])) and got[k] != want[k]}
    assert mismatches == {}, mismatches


def test_every_history_has_an_expectation():
    assert set(HISTORIES) == set(EXPECTED)
    assert {e[0] for e in EXPECTED.values()} == set(LABEL_SOURCES)


def test_label_source(crm_leads):
    _assert_same(label_source(crm_leads), 0)


def test_won_within_h(crm_leads):
    _assert_same(won_within_h(crm_leads), 1)


def test_horizon_label_default_excludes_ghosted_and_censors_stalled(crm_leads):
    _assert_same(horizon_label(crm_leads, HORIZON), 2)


def test_include_ghosted_counts_ghosted_as_zero(crm_leads):
    _assert_same(horizon_label(crm_leads, LabelConfig(include_ghosted=True)), 3)


def test_stalled_as_lost_counts_stalled_as_zero(crm_leads):
    _assert_same(horizon_label(crm_leads, LabelConfig(stalled_as_lost=True)), 4)


def test_both_flags_together(crm_leads):
    y = horizon_label(crm_leads, LabelConfig(include_ghosted=True, stalled_as_lost=True))
    assert y["ghosted_mature"] == 0 and y["stalled_mature_90d"] == 0 and y["contacted_after_h_then_lost"] == 0
    assert np.isnan(y["young_stalled"]) and np.isnan(y["young_new"]) and np.isnan(y["young_lost"])


def test_legacy_label_on_the_same_histories(crm_leads):
    _assert_same(label(crm_leads), 5)


def test_matured_at_and_maturity(crm_leads):
    m = matured_at(crm_leads)
    assert (m == crm_leads.created_at + pd.Timedelta(days=120)).all()
    assert m["won_at_h"] == pd.Timestamp("2026-06-29T10:00:00Z")
    assert m["edge_mature_lost"] == AS_OF
    mature = is_mature(crm_leads)
    assert mature["edge_mature_lost"] and not mature["edge_plus_1s_lost"] and not mature["young_won"]
    assert mature["won_at_h"]


def test_ghosted_at_horizon_uses_first_contact_time(crm_leads):
    g = ghosted_at_horizon(crm_leads)
    assert g["ghosted_mature"] and g["contacted_after_h_then_lost"] and g["young_new"]
    assert label_source(crm_leads)["young_new"] == "open"  # ghosted at H needs a mature lead
    assert not g["contacted_exactly_at_h_then_lost"] and not g["won_early"] and not g["stalled_mature_90d"]


def test_horizon_days_is_a_parameter(crm_leads):
    # With H = 220 the day-121 win is inside the horizon, and a lead created 2026-03-01 is no longer mature
    # (matured_at 2026-10-07 > AS_OF), so its CRM loss is unlabelled.
    y = horizon_label(crm_leads, LabelConfig(horizon_days=220))
    assert y["won_at_h_plus_1d"] == 1
    assert np.isnan(y["lost_mature"]) and not is_mature(crm_leads, 220)["lost_mature"]
    # With H = 5 the day-10 win is late (0) and the day-5 loss is still a 0.
    y5 = won_within_h(crm_leads, 5)
    assert y5["won_early"] == 0 and y5["lost_mature"] == 0


def test_assign_labels_columns(crm_leads):
    X = assign_labels(crm_leads.copy(), HORIZON)
    assert {"label_source", "matured_at", "won_within_h", "y"} <= set(X.columns)
    X = assign_labels(crm_leads.copy(), LEGACY)
    assert "label_source" not in X and "matured_at" not in X and "won_within_h" not in X
    assert X.y.equals(label(crm_leads))


def test_config_eligible_and_extra_columns(crm_leads):
    assert LEGACY.eligible(crm_leads).all()
    assert HORIZON.eligible(crm_leads).equals(is_mature(crm_leads))
    assert LEGACY.extra_score_columns() == ()
    assert HORIZON.extra_score_columns() == ("won_within_h", "label_source", "matured_at")


def test_label_mode_accepts_enum_or_string():
    assert LabelConfig(mode="legacy") == LEGACY and LEGACY.mode is LabelMode.LEGACY and LEGACY.is_legacy
    assert LabelConfig(mode=LabelMode.HORIZON) == HORIZON and not HORIZON.is_legacy


def test_legacy_mode_tolerates_an_unrecognised_stage(tmp_path):
    # The baseline leaves an unmapped final stage unlabelled and carries on; legacy mode must too.
    t0 = pd.Timestamp("2026-01-05T09:00:00Z")
    pd.DataFrame({"lead_id": ["L1", "L2"], "created_at": [t0.isoformat()] * 2, "answers": ["{}"] * 2,
                  "company_domain": ["a.example"] * 2}).to_csv(tmp_path / "historical_leads.csv", index=False)
    pd.DataFrame({"lead_id": ["L1", "L1", "L2", "L2"], "stage": ["New", "On hold", "New", "Closed Lost"],
                  "deal_value": [None] * 4,
                  "changed_at": [t0.isoformat(), (t0 + d).isoformat(), t0.isoformat(), (t0 + d).isoformat()]}
                 ).to_csv(tmp_path / "crm_history.csv", index=False)
    pd.DataFrame({"company_name": ["A Inc"], "domain": ["a.example"], "sector": ["x"], "employee_band": ["1-10"], "monthly_ad_spend_band": ["none"],
                  "crm_platform": ["x"], "is_hiring": [False]}).to_csv(tmp_path / "companies.csv", index=False)
    X = assign_labels(load(tmp_path), LEGACY)
    assert np.isnan(X.y["L1"]) and X.y["L2"] == 0 and "label_source" not in X
    with pytest.raises(ValueError, match="no recognised final CRM stage"):
        assign_labels(load(tmp_path), HORIZON)


def test_split_masks_eligible_restricts_both_sets():
    X = pd.DataFrame({
        "created_at": pd.to_datetime(["2026-04-01T00:00:00Z", "2026-05-10T00:00:00Z", "2026-06-10T00:00:00Z"]),
        "y": [1.0, 0.0, 1.0],
    })
    mature = pd.Series([True, True, False])
    tr, te = split_masks(X, eligible=mature)
    assert tr.tolist() == [True, False, False] and te.tolist() == [False, True, False]


def test_label_source_rejects_unknown_stage():
    X = pd.DataFrame([_row("Won", 10, 1), _row(None, 10, 1)]).assign(first_contact_at=pd.NaT)
    with pytest.raises(ValueError, match="no recognised final CRM stage"):
        label_source(X)


@pytest.mark.parametrize("kwargs", [dict(mode="other"), dict(mode="legacy", include_ghosted=True),
                                    dict(mode="legacy", stalled_as_lost=True), dict(horizon_days=0)])
def test_label_config_validation(kwargs):
    with pytest.raises(ValueError):
        LabelConfig(**kwargs)


def test_label_config_describe():
    assert LEGACY.describe() == "legacy" and HORIZON.describe() == "horizon H=120"
    assert LabelConfig(include_ghosted=True, stalled_as_lost=True).describe() == \
        "horizon H=120 +ghosted +stalled-as-lost"
