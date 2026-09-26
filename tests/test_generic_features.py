"""Generic feature set (Phase 10, ADR 0024): the extras' encoder (emva.generic), the mapping's [[features]], the
converter's extra_features.csv, the pipeline, the model bundle and single-lead scoring, the draft contract and the
report's "Generic vs v2" section. Every dataset here is a hand-built fixture (seeded rng, not real data)."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from emva.constants import GENERIC_MIN_LEVEL_COUNT, MISSING
from emva.context.cache import ReplyCache
from emva.context.contract import ContractError
from emva.design import fixed_columns
from emva.features import CATS_V2, FeatureSet
from emva.generic import (
    ALL,
    EXTRA_FEATURES_FILE,
    OTHER,
    ExtraFeature,
    ExtraKind,
    bin_labels,
    fit_encoder,
    numeric_edges,
    read_extra_features,
)
from emva.ingest.convert import convert, convert_leads
from emva.ingest.draft import RESPONSE_SCHEMA, draft_mapping
from emva.ingest.mapping import Expr, dump_mapping, load_mapping
from emva.ingest.profile import profile
from emva.persist import load_bundle, save_bundle
from emva.pipeline import run
from emva.scoring import points_breakdown, score_leads

from conftest import DATA_V1
from test_ingest import FakeClient, _reply, message

CAT, NUM = ExtraKind.CATEGORICAL, ExtraKind.NUMERIC
AS_OF, TEST_FROM = "2025-12-31", "2025-07-01"

MAPPING = f'''
name = "fixture_generic"
as_of = "{AS_OF}"
test_from = "{TEST_FROM}"
outcome_confirmed = true
features_confirmed = true

[[sources]]
name = "l"
file = "leads.csv"

[[sources]]
name = "o"
file = "orgs.csv"
join_on = "org"

[lead]
lead_id = "l.id"
created_at = "l.signup"
contacted_at = "l.signup"
won_at = "l.closed"
close_at = "l.closed"
deal_value = "l.amount"

[outcome]
kind = "stage"
column = "l.stage"

[outcome.stage_map]
"Closed Won" = "Won"
"Closed Lost" = "Lost"

[[fields]]
source = "l.source"
[fields.value_map]
ppc = {{ utm_source = "google", utm_medium = "cpc" }}
fb = {{ utm_source = "facebook", utm_medium = "paid_social" }}
seo = {{ utm_medium = "organic" }}

[[features]]
source = "l.tier"
kind = "categorical"
name = "tier"
reason = "planted signal"
confidence = "high"

[[features]]
source = {{ op = "sum", args = ["l.seats", "l.extra_seats"] }}
kind = "numeric"
name = "seats"

[[features]]
source = "o.sector"
kind = "categorical"
name = "sector"
'''


def generic_frames(n: int = 800, seed: int = 0) -> dict[str, pd.DataFrame]:
    """A synthetic CRM export with a planted signal in the extra ``tier`` (gold wins 70%, bronze 10%), a weak one in
    ``seats`` and none in ``sector``; ``source`` (the only v2 input) carries no signal."""
    rng = np.random.default_rng(seed)
    signup = pd.Timestamp("2025-01-01") + pd.to_timedelta(np.sort(rng.uniform(0, 240, n)), unit="D")
    tier = rng.choice(["gold", "silver", "bronze"], n)
    seats = rng.integers(1, 50, n)
    p_win = np.select([tier == "gold", tier == "silver"], [0.7, 0.35], 0.1) + 0.002 * seats
    won = rng.uniform(size=n) < p_win
    leads = pd.DataFrame({
        "id": [f"G{i:04d}" for i in range(n)],
        "signup": signup.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": np.where(won, "Closed Won", "Closed Lost"),
        "closed": (signup + pd.to_timedelta(rng.uniform(2, 60, n), unit="D")).strftime("%Y-%m-%d %H:%M:%S"),
        "amount": np.where(won, rng.integers(500, 20000, n).astype(str), None),
        "source": rng.choice(["ppc", "fb", "seo"], n),
        "tier": tier,
        "seats": seats.astype(str),
        "extra_seats": np.where(rng.uniform(size=n) < 0.1, None, "1"),
        "org": [f"org{i % 5}" for i in range(n)],
    })
    orgs = pd.DataFrame({"org": [f"org{i}" for i in range(5)],
                         "sector": ["retail", "retail", "software", "health", "software"]})
    return {"leads.csv": leads.astype(object), "orgs.csv": orgs.astype(object)}


@pytest.fixture(scope="module")
def mapping():
    return load_mapping(MAPPING, require_confirmed=True)


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    return generic_frames()


@pytest.fixture(scope="module")
def dataset(frames, mapping, tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("generic")
    for name, content in convert(frames, mapping).items():
        (d / name).write_bytes(content)
    return d


@pytest.fixture(scope="module")
def generic_run(dataset):
    return run(dataset, features=FeatureSet.GENERIC)


# --- encoder -------------------------------------------------------------------------------------------------------

def _frame(**cols: list) -> pd.DataFrame:
    return pd.DataFrame({f"x_{k}": pd.Series(v, dtype=object) for k, v in cols.items()})


def test_categorical_levels_are_fitted_on_training_rows_only() -> None:
    train = _frame(c=["a"] * 40 + ["b"] * 35 + ["rare"] * 3 + [None] * 2)
    enc = fit_encoder((ExtraFeature("c", CAT),), train)
    e = enc.extras[0]
    assert e.levels == ("a", "b", OTHER, MISSING) and e.reference == "a"
    assert e.train_counts == {"a": 40, "b": 35, OTHER: 3, MISSING: 2}
    # "z" is frequent at scoring (or in the test period) but was never in training: other, never a new column
    test = _frame(c=["z"] * 100 + ["b", " a ", "", None])
    assert enc.levels(test).x_c.tolist() == [OTHER] * 100 + ["b", "a", MISSING, MISSING]
    assert enc.columns() == ["x_c=b", "x_c=other", "x_c=missing"]
    D = enc.design(test)
    assert list(D.columns) == enc.columns() and D["x_c=other"].sum() == 100 and D.values.sum() == 103


def test_min_count_boundary() -> None:
    n = GENERIC_MIN_LEVEL_COUNT
    enc = fit_encoder((ExtraFeature("c", CAT),), _frame(c=["kept"] * n + ["dropped"] * (n - 1) + ["big"] * (n + 5)))
    assert enc.extras[0].levels == ("big", "kept", OTHER, MISSING)
    assert enc.levels(_frame(c=["dropped"])).x_c.tolist() == [OTHER]


def test_reference_ties_and_no_kept_level() -> None:
    tie = fit_encoder((ExtraFeature("c", CAT),), _frame(c=["b"] * 30 + ["a"] * 30)).extras[0]
    assert tie.reference == "a" and tie.levels == ("a", "b", OTHER, MISSING)
    rare = fit_encoder((ExtraFeature("c", CAT),), _frame(c=["x", "y", None, None, None])).extras[0]
    assert rare.reference == MISSING and rare.levels == (MISSING, OTHER)
    # raw values spelled like the reserved levels fall into them
    reserved = fit_encoder((ExtraFeature("c", CAT),), _frame(c=["other"] * 40 + ["missing"] * 40)).extras[0]
    assert reserved.levels == (OTHER, MISSING) and reserved.train_counts == {OTHER: 40, MISSING: 40}


def test_numeric_bins_are_deterministic_with_duplicate_edges() -> None:
    v = np.arange(1, 101, dtype=float)
    assert numeric_edges(v) == (20.0, 40.0, 60.0, 80.0)  # observed values (inverted_cdf)
    assert numeric_edges(v[::-1]) == numeric_edges(v)  # order-free
    # many ties: the quantiles collapse to fewer edges, deduplicated
    assert numeric_edges(np.array([0.0] * 80 + [5.0] * 20)) == (0.0,)
    # an edge at the maximum would leave the top bin empty: moved down to the largest value below it
    assert numeric_edges(np.array([0.0] * 10 + [1.0] * 90)) == (0.0,)
    assert numeric_edges(np.array([7.0] * 10)) == () and numeric_edges(np.array([])) == ()
    assert bin_labels((0.0,)) == ("<=0", ">0") and bin_labels(()) == (ALL,)
    assert bin_labels((1.0, 2.5)) == ("<=1", "(1, 2.5]", ">2.5")
    assert bin_labels((1.0000001, 1.0000002))[0] == "<=1.0000001"  # printed in full when 6 digits would clash


def test_numeric_encoding_edges_missing_and_out_of_range() -> None:
    train = _frame(n=[str(i) for i in range(1, 101)] + [None, ""])
    enc = fit_encoder((ExtraFeature("n", NUM),), train)
    e = enc.extras[0]
    assert e.reference == "<=20" and e.levels[-1] == MISSING and len(e.levels) == 6
    assert e.train_counts[MISSING] == 2 and e.train_counts["<=20"] == 20 and e.train_counts[">80"] == 20
    got = enc.levels(_frame(n=["-1000", "20", "20.01", "1e9", None])).x_n.tolist()
    assert got == ["<=20", "<=20", "(20, 40]", ">80", MISSING]
    with pytest.raises(ValueError, match="non-numbers.*'lots'"):
        enc.levels(_frame(n=["lots"]))


def test_all_missing_numeric_column() -> None:
    enc = fit_encoder((ExtraFeature("n", NUM),), _frame(n=[None] * 10))
    e = enc.extras[0]
    assert e.levels == (ALL, MISSING) and e.reference == MISSING and enc.columns() == ["x_n=all"]
    assert enc.design(_frame(n=["3"]))["x_n=all"].tolist() == [1.0]


def test_extra_feature_names_are_slugs() -> None:
    with pytest.raises(ValueError, match="lower-case"):
        ExtraFeature("Lead Time", NUM)
    with pytest.raises(ValueError):
        ExtraFeature("ok", "text")


# --- mapping and converter -----------------------------------------------------------------------------------------

def test_features_round_trip_through_toml(mapping) -> None:
    assert [f.name for f in mapping.features] == ["tier", "seats", "sector"]
    assert mapping.features[1].source.op == "sum" and mapping.features[0].review.confidence == "high"
    text = dump_mapping(mapping)
    assert load_mapping(text) == mapping and dump_mapping(load_mapping(text)) == text
    assert "features_confirmed = true" in text
    assert mapping.source_columns() == {"leads.csv": ["id", "signup", "source", "tier", "seats", "extra_seats", "org"],
                                        "orgs.csv": ["org", "sector"]}


def test_a_mapping_without_features_dumps_as_in_phase_9() -> None:
    from test_ingest import MAPPING as PHASE9

    m = load_mapping(PHASE9)
    assert m.features == () and not m.features_confirmed
    assert "features" not in dump_mapping(m) and load_mapping(dump_mapping(m)) == m


@pytest.mark.parametrize("edit, match", [
    (lambda t: t.replace('kind = "numeric"', 'kind = "text"'), "kind 'text'"),
    (lambda t: t.replace('name = "seats"', 'name = "tier"'), "feature names must be unique"),
    (lambda t: t.replace('name = "seats"', 'name = "Seats"'), "lower-case"),
    (lambda t: t.replace('source = "o.sector"', 'source = "q.sector"'), "names no source"),
    (lambda t: t.replace('name = "sector"\n', ""), "needs \\['name'\\]"),
])
def test_bad_features_are_refused(edit, match) -> None:
    with pytest.raises(ValueError, match=match):
        load_mapping(edit(MAPPING))


def test_convert_writes_extra_features_and_refuses_unconfirmed(frames, mapping) -> None:
    files = convert(frames, mapping)
    E = pd.read_csv(pd.io.common.BytesIO(files[EXTRA_FEATURES_FILE]), dtype=str, keep_default_na=False)
    src = frames["leads.csv"]
    assert list(E.columns) == ["lead_id", "tier", "seats", "sector"] and list(E.lead_id) == list(src.id)
    assert list(E.tier) == list(src.tier)
    blank_extra = src.extra_seats.isna().to_numpy()
    assert (E.seats[blank_extra] == "").all()  # a blank argument of sum is a blank value
    assert E.seats[~blank_extra].astype(float).tolist() == (src.seats[~blank_extra].astype(float) + 1).tolist()
    assert set(E.sector) == {"retail", "software", "health"}
    meta = json.loads(files["dataset.json"])
    assert meta["features"] == [{"name": "tier", "kind": "categorical", "source": "l.tier"},
                                {"name": "seats", "kind": "numeric",
                                 "source": '{ op = "sum", args = ["l.seats", "l.extra_seats"] }'},
                                {"name": "sector", "kind": "categorical", "source": "o.sector"}]
    # historical_leads.csv is the fixed schema whatever the features
    assert "tier" not in files["historical_leads.csv"].decode().splitlines()[0]
    with pytest.raises(ValueError, match="features_confirmed is false"):
        convert(frames, replace(mapping, features_confirmed=False))
    with pytest.raises(ValueError, match="features_confirmed is false"):
        load_mapping(MAPPING.replace("features_confirmed = true", "features_confirmed = false"),
                     require_confirmed=True)
    bad = {**frames, "leads.csv": frames["leads.csv"].assign(seats="many")}
    with pytest.raises(ValueError, match="feature seats"):
        convert(bad, mapping)


def test_a_date_expression_is_not_a_feature(frames, mapping) -> None:
    from emva.ingest.mapping import FeatureMap

    dated = replace(mapping, features=(FeatureMap(Expr("minus_days", args=(Expr.col("l.signup"),
                                                                            Expr.col("l.seats"))), "numeric", "d"),))
    with pytest.raises(ValueError, match="date expression"):
        convert(frames, dated)


def test_mapping_without_features_writes_no_extra_file() -> None:
    from test_ingest import MAPPING as PHASE9, fixture_frames

    files = convert(fixture_frames(), load_mapping(PHASE9))
    assert EXTRA_FEATURES_FILE not in files and "features" not in json.loads(files["dataset.json"])


# --- pipeline ------------------------------------------------------------------------------------------------------

def test_generic_design_is_v2_plus_x_columns(generic_run) -> None:
    r = generic_run
    x_cols = r.extras.columns()
    assert list(r.design.columns) == fixed_columns(CATS_V2) + x_cols
    assert all(c.startswith(("x_tier=", "x_seats=", "x_sector=")) for c in x_cols)
    tier = r.extras.extras[0]
    assert set(tier.levels) == {"gold", "silver", "bronze", OTHER, MISSING}
    # fitted on the training leads only
    assert sum(tier.train_counts.values()) == int(r.train.sum())
    assert r.weights.index.str.startswith("x_tier=").any()


def test_generic_beats_v2_on_a_planted_signal(dataset, generic_run) -> None:
    """Sanity check on a hand-built fixture only: the planted ``tier`` signal is learned by generic, not by v2."""
    from sklearn.metrics import roc_auc_score

    v2 = run(dataset, features=FeatureSet.V2)
    te = generic_run.test
    assert te.equals(v2.test)
    y = generic_run.X.y[te]
    auc_g, auc_v2 = roc_auc_score(y, generic_run.X.p_formula[te]), roc_auc_score(y, v2.X.p_formula[te])
    assert auc_g > 0.7 and auc_g > auc_v2 + 0.1


def test_v2_ignores_extra_features(dataset, tmp_path) -> None:
    """v2 on a dataset with extra_features.csv gives exactly the scores it gives without the file."""
    v2 = run(dataset, features=FeatureSet.V2)
    bare = tmp_path / "bare"
    bare.mkdir()
    for p in dataset.iterdir():
        if p.name != EXTRA_FEATURES_FILE:
            (bare / p.name).write_bytes(p.read_bytes())
    pd.testing.assert_frame_equal(v2.scores(), run(bare, features=FeatureSet.V2).scores())
    assert v2.extras is None and not any(c.startswith("x_") for c in v2.design.columns)
    with pytest.raises(ValueError, match=EXTRA_FEATURES_FILE):
        run(bare, features=FeatureSet.GENERIC)


def test_generic_refuses_a_dataset_without_declared_extras() -> None:
    with pytest.raises(ValueError, match="declares 'features'"):
        run(DATA_V1, features=FeatureSet.GENERIC)


def test_read_extra_features_refuses_malformed_files(dataset, tmp_path) -> None:
    for p in dataset.iterdir():
        (tmp_path / p.name).write_bytes(p.read_bytes())
    E = pd.read_csv(tmp_path / EXTRA_FEATURES_FILE, dtype=str)
    features, raw = read_extra_features(tmp_path)
    assert [f.column for f in features] == list(raw.columns) == ["x_tier", "x_seats", "x_sector"]
    E.rename(columns={"tier": "grade"}).to_csv(tmp_path / EXTRA_FEATURES_FILE, index=False)
    with pytest.raises(ValueError, match="expected \\['lead_id', 'tier'"):
        read_extra_features(tmp_path)
    pd.concat([E, E.head(1)]).to_csv(tmp_path / EXTRA_FEATURES_FILE, index=False)
    with pytest.raises(ValueError, match="unique"):
        read_extra_features(tmp_path)
    E.assign(lead_id=E.lead_id.where(E.index > 0, "nobody")).to_csv(tmp_path / EXTRA_FEATURES_FILE, index=False)
    with pytest.raises(ValueError, match="not in historical_leads.csv.*nobody"):
        run(tmp_path, features=FeatureSet.GENERIC)


# --- bundle and scoring --------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bundle(dataset, generic_run, tmp_path_factory):
    path = tmp_path_factory.mktemp("bundle") / "model.joblib"
    save_bundle(generic_run, path, dataset, 1.0)
    return load_bundle(path)


def test_bundle_round_trip_scores_equal_the_batch_run(bundle, generic_run, frames, mapping, dataset) -> None:
    assert bundle.feature_set is FeatureSet.GENERIC and bundle.extras == generic_run.extras
    assert bundle.design_columns == tuple(generic_run.design.columns)
    ids = list(generic_run.X.index[generic_run.test][:5]) + list(generic_run.X.index[generic_run.train][:5])
    new = frames["leads.csv"].set_index("id").loc[ids].reset_index().drop(columns=["stage", "closed", "amount"])
    leads = convert_leads({"leads.csv": new, "orgs.csv": frames["orgs.csv"]}, mapping)
    assert [c for c in leads.columns if c.startswith("x_")] == ["x_tier", "x_seats", "x_sector"]
    scores = score_leads(bundle, leads, dataset)
    np.testing.assert_allclose(scores.p_formula, generic_run.X.p_formula.loc[ids], rtol=1e-12)
    np.testing.assert_allclose(scores.value_at_submit, generic_run.X.value_at_submit.loc[ids], rtol=1e-12)
    pts = points_breakdown(bundle, leads.head(1), dataset)
    assert pts.feature.str.startswith("x_").any()


def test_unseen_value_at_scoring_is_other_and_absent_extras_are_missing(bundle, frames, mapping, dataset) -> None:
    new = frames["leads.csv"].head(3).drop(columns=["stage", "closed", "amount"]).assign(tier=["platinum", "gold",
                                                                                                "gold"])
    leads = convert_leads({"leads.csv": new, "orgs.csv": frames["orgs.csv"]}, mapping)
    got = score_leads(bundle, leads, dataset)
    assert np.isfinite(got.p_formula).all()
    other = bundle.model.coef_[0][list(bundle.design_columns).index("x_tier=other")]
    pts = points_breakdown(bundle, leads.head(1), dataset)
    assert pts.loc[pts.feature == "x_tier", "level"].tolist() == [OTHER]
    assert pts.loc[pts.feature == "x_tier", "log_odds"].iloc[0] == pytest.approx(other)
    # a lead without its extra columns (e.g. from the EMVA form) scores with every extra missing
    bare = score_leads(bundle, leads.drop(columns=["x_tier", "x_seats", "x_sector"]), dataset)
    assert np.isfinite(bare.p_formula).all()
    # the fixed-schema columns still refuse an unknown level (ADR 0009)
    with pytest.raises(ValueError, match="form_variant"):
        score_leads(bundle, leads.assign(form_variant="Z"), dataset)


def test_a_v2_bundle_refuses_extra_columns(dataset, frames, mapping, tmp_path) -> None:
    save_bundle(run(dataset, features=FeatureSet.V2), tmp_path / "m.joblib", dataset, 1.0)
    leads = convert_leads({"leads.csv": frames["leads.csv"].head(2), "orgs.csv": frames["orgs.csv"]}, mapping)
    with pytest.raises(ValueError, match="unknown columns.*x_seats"):
        score_leads(load_bundle(tmp_path / "m.joblib"), leads, dataset)


def test_cli_trains_and_scores_generic(dataset, frames, tmp_path, capsys) -> None:
    import io

    from emva.__main__ import main as emva_main
    from emva.ingest.__main__ import main as ingest_main

    emva_main(["--data", str(dataset), "--out", str(tmp_path), "--feature-set", "generic"])
    assert "x_tier=" in (tmp_path / "weights.csv").read_text()
    (tmp_path / "mapping.toml").write_text(MAPPING)
    frames["leads.csv"].head(4).drop(columns=["stage", "closed", "amount"]).to_csv(tmp_path / "new.csv", index=False)
    frames["orgs.csv"].to_csv(tmp_path / "orgs.csv", index=False)
    capsys.readouterr()
    ingest_main(["score", "--mapping", str(tmp_path / "mapping.toml"), "--raw", str(tmp_path / "new.csv"), "--run",
                 str(tmp_path), "--data", str(dataset)])
    out = pd.read_csv(io.StringIO(capsys.readouterr().out), index_col="lead_id")
    assert list(out.index) == list(frames["leads.csv"].id.head(4)) and out.p_formula.between(0, 1).all()


# --- draft contract ------------------------------------------------------------------------------------------------

def test_draft_may_propose_features_but_never_confirms_them(tmp_path) -> None:
    from test_ingest import fixture_frames

    frames = fixture_frames()
    reply = _reply()
    reply["columns"] = [c for c in reply["columns"] if c["column"] != "notes"] + [
        {"file": "leads.csv", "column": "notes", "target": "feature", "feature_kind": "categorical",
         "value_map": [], "reason": "call outcome notes", "confidence": "low"},
        {"file": "leads.csv", "column": "size", "target": "feature", "feature_kind": "categorical",
         "value_map": [], "reason": "company size", "confidence": "medium"},
        {"file": "orgs.csv", "column": "country", "target": "feature", "feature_kind": "categorical",
         "value_map": [], "reason": "org country", "confidence": "medium"}]
    items = RESPONSE_SCHEMA["properties"]["columns"]["items"]
    assert "feature" in items["properties"]["target"]["enum"]
    assert items["properties"]["feature_kind"]["enum"] == ["numeric", "categorical", "none"]
    cache = ReplyCache(tmp_path / "cache.json")
    client = FakeClient(message(json.dumps(reply)))
    m = draft_mapping(profile(frames), client, cache, "drafted")
    assert [(f.name, f.kind, f.source.column) for f in m.features] == [
        ("notes", "categorical", "leads.notes"), ("size", "categorical", "leads.size"),
        ("country", "categorical", "orgs.country")]
    assert not m.features_confirmed and not m.outcome_confirmed
    assert load_mapping(dump_mapping(m)) == m
    assert draft_mapping(profile(frames), None, cache, "drafted") == m and len(client.calls) == 1  # cached
    # a feature needs a kind, and only a feature has one
    for target, kind in (("feature", "none"), ("email", "numeric")):
        bad = _reply()
        bad["columns"][4] = {**bad["columns"][4], "target": target, "feature_kind": kind}
        with pytest.raises(ContractError, match="feature_kind"):
            draft_mapping(profile(fixture_frames(seed=3 if target == "feature" else 4)),
                          FakeClient(message(json.dumps(bad))), ReplyCache(tmp_path / f"{target}.json"), "x")


# --- report --------------------------------------------------------------------------------------------------------

def test_report_has_the_generic_vs_v2_section(dataset) -> None:
    from emva.eval.report import build_report

    text = build_report(dataset, n_resamples=50, features=FeatureSet.GENERIC)
    section = text[text.index("## Generic vs v2"):]
    assert "`generic` features" in text and "| (b) Horizon labels, mature test set |" in section
    assert "Phase 10 criterion (pre-registered): PASS" in section  # the planted signal clears the bar here
    assert "### Leakage screen" in section and "| tier | categorical |" in section
    assert "## Generic vs v2" not in build_report(dataset, n_resamples=20)  # v2 candidate: no section


def test_leakage_screen_flags_a_leaky_extra(dataset, generic_run) -> None:
    from emva.eval.generic_report import LEAKAGE_FLAG_TEXT, leakage_screen

    screen = leakage_screen(generic_run).set_index("extra")
    assert screen.loc["tier", "training AUC"] > 0.7 and screen.loc["tier", "flag"] == ""
    assert screen.loc["sector", "training AUC"] < 0.6
    r = generic_run
    leaky = replace(r, X=r.X.assign(x_tier=np.where(r.X.y == 1, "gold", "bronze")))
    enc = fit_encoder(tuple(e.feature for e in r.extras.extras), leaky.X[r.train])
    flagged = leakage_screen(replace(leaky, extras=enc)).set_index("extra")
    assert flagged.loc["tier", "training AUC"] == pytest.approx(1.0) and flagged.loc["tier", "flag"] == LEAKAGE_FLAG_TEXT
    with pytest.raises(ValueError, match="generic"):
        leakage_screen(replace(r, extras=None))
