"""Generic dataset converter (``emva.ingest``): mapping TOML, profiles and redaction, drafting (fake client),
conversion, refusals, coverage, validation, and the pipeline and report on a converted dataset (ADR 0020, 0022)."""
from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx2
import numpy as np
import pandas as pd
import pytest

from app.validation import MIN_LEADS, validate_files
from emva.context.cache import ReplyCache
from emva.context.contract import ContractError
from emva.design import fixed_columns
from emva.features import CATS_V2
from emva.ingest.convert import PLACEHOLDER_DOMAIN, convert, convert_leads, evaluate, frames_from_bytes
from emva.ingest.draft import RESPONSE_SCHEMA, draft_mapping, is_cached, render_profiles
from emva.ingest.mapping import LEAD_COLUMNS, TARGETS, DatasetMapping, Expr, dump_mapping, load_mapping
from emva.ingest.profile import NAME_TOKEN, profile, redact
from emva.io import load
from emva.pipeline import run

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "ingest"
AS_OF, TEST_FROM = "2025-12-31", "2025-08-01"
SOURCES = ["ppc", "fb", "seo", ""]

MAPPING = f'''
name = "fixture_crm"
as_of = "{AS_OF}"
test_from = "{TEST_FROM}"
source_url = "https://example.invalid/fixture"
licence = "test fixture"
outcome_confirmed = true

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
"In progress" = "Qualified"
"Open" = "New"

[review]
created_at = {{ reason = "sign-up time", confidence = "high" }}

[[fields]]
source = "l.mail"
target = "email"
confidence = "high"

[[fields]]
source = "l.source"
reason = "paid channels"
confidence = "medium"
[fields.value_map]
ppc = {{ utm_source = "google", utm_medium = "cpc" }}
fb = {{ utm_source = "facebook", utm_medium = "paid_social" }}
seo = {{ utm_medium = "organic" }}

[[fields]]
source = "l.size"
[fields.value_map]
small = {{ "answers.company_size" = "1-10" }}
mid = {{ "answers.company_size" = "51-200" }}
large = {{ "answers.company_size" = "1000+" }}

[[fields]]
source = "o.country"
target = "answers.country"

[[fields]]
source = "l.notes"
target = "ignore"
reason = "free text written after the call (leakage)"
'''


def fixture_frames(n: int = 300, seed: int = 0) -> dict[str, pd.DataFrame]:
    """A synthetic CRM export: hand-specified structure, values drawn with a seeded rng (not real data).

    Leads sign up 2025-01-01 .. 2025-10-31; ppc leads win more often; 10% stay "In progress", 5% "Open".
    """
    rng = np.random.default_rng(seed)
    signup = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.uniform(0, 303, n), unit="D")
    source = [SOURCES[i % 4] for i in range(n)]
    p_win = np.where(np.array(source) == "ppc", 0.6, 0.25)
    u = rng.uniform(size=n)
    stage = np.where(u < 0.05, "Open", np.where(u < 0.15, "In progress",
                                                np.where(rng.uniform(size=n) < p_win, "Closed Won", "Closed Lost")))
    closed = (signup + pd.to_timedelta(rng.uniform(2, 60, n), unit="D")).where(
        pd.Series(stage).isin(["Closed Won", "Closed Lost"]).values)
    leads = pd.DataFrame({
        "id": [f"X{i:04d}" for i in range(n)],
        "signup": signup.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage,
        "closed": closed.strftime("%Y-%m-%d %H:%M:%S").where(pd.notna(closed), None),
        "amount": np.where(stage == "Closed Won", rng.integers(500, 20000, n).astype(str), None),
        "mail": [f"buyer{i}@firm{i}.example" for i in range(n)],
        "source": [s or None for s in source],
        "size": rng.choice(["small", "mid", "large", None], n),
        "org": [f"org{i % 7}" for i in range(n)],
        "notes": "call notes",
    })
    orgs = pd.DataFrame({"org": [f"org{i}" for i in range(6)], "country": ["UK", "DE", "FR", "UK", "US", "UK"]})
    return {"leads.csv": leads.astype(object), "orgs.csv": orgs.astype(object)}


@pytest.fixture(scope="module")
def mapping() -> DatasetMapping:
    return load_mapping(MAPPING)


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    return fixture_frames()


@pytest.fixture(scope="module")
def converted(frames: dict[str, pd.DataFrame], mapping: DatasetMapping) -> dict[str, bytes]:
    return convert(frames, mapping)


def _csv(files: dict[str, bytes], name: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(files[name]), dtype=str)


def _training(files: dict[str, bytes]) -> dict[str, bytes]:
    """The files the app's validation takes (dataset.json is the converter's metadata, not a training file)."""
    return {k: v for k, v in files.items() if k != "dataset.json"}


# --- mapping -------------------------------------------------------------------------------------------------------

def test_lead_columns_are_the_v1_header() -> None:
    header = (REPO / "data" / "v1" / "historical_leads.csv").read_text().splitlines()[0].split(",")
    assert tuple(header) == LEAD_COLUMNS


def test_mapping_round_trips_through_toml(mapping: DatasetMapping) -> None:
    text = dump_mapping(mapping, header="a header\nsecond line")
    assert text.startswith("# a header\n# second line\n")
    assert load_mapping(text) == mapping
    assert dump_mapping(load_mapping(text)) == dump_mapping(mapping)  # deterministic


def test_hotel_style_mapping_round_trips() -> None:
    m = load_mapping((REPO / "mappings" / "hotel_bookings.toml").read_text())
    assert m.lead.lead_id is None and m.lead.created_at.op == "minus_days"
    assert load_mapping(dump_mapping(m)) == m


def test_derived_operations_hotel_style() -> None:
    raw = pd.DataFrame({"b.y": ["2016", "2017", None], "b.m": ["July", "feb", "3"], "b.d": ["1", "28", "4"],
                        "b.lead": ["10", "0.5", "3"], "b.adr": ["100.5", "80", "90"], "b.we": ["2", "0", "1"],
                        "b.wk": ["3", "1", None]})
    arrival = Expr("date_from_parts", args=(Expr.col("b.y"), Expr.col("b.m"), Expr.col("b.d")))
    booked = evaluate(Expr("minus_days", args=(arrival, Expr.col("b.lead"))), raw)
    assert booked.iloc[0] == pd.Timestamp("2016-06-21", tz="UTC")
    assert booked.iloc[1] == pd.Timestamp("2017-02-27T12:00", tz="UTC")
    assert pd.isna(booked.iloc[2])  # a blank part gives a blank date
    value = evaluate(Expr("product", args=(Expr.col("b.adr"), Expr("sum", args=(Expr.col("b.we"),
                                                                                Expr.col("b.wk"))))), raw)
    assert value.iloc[0] == pytest.approx(502.5) and value.iloc[1] == 80 and pd.isna(value.iloc[2])
    with pytest.raises(ValueError, match="impossible date"):
        evaluate(arrival, raw.assign(**{"b.d": ["31", "30", "1"]}))
    with pytest.raises(ValueError, match="not months"):
        evaluate(arrival, raw.assign(**{"b.m": ["Julyish", "feb", "3"]}))
    with pytest.raises(ValueError, match="takes 2 arguments"):
        Expr("minus_days", args=(arrival,))
    with pytest.raises(ValueError, match="unknown op"):
        Expr("eval", args=(arrival, arrival))


@pytest.mark.parametrize("edit, match", [
    (lambda t: t.replace('created_at = "l.signup"\n', ""), "no created_at"),
    (lambda t: t.replace('target = "answers.country"', 'target = "shoe_size"'), "unknown target column 'shoe_size'"),
    (lambda t: t.replace('utm_medium = "organic"', 'favourite_colour = "blue"'), "unknown target column"),
    (lambda t: t.replace('kind = "stage"', 'kind = "vibes"'), "outcome kind"),
    (lambda t: t.replace('"Open" = "New"', '"Open" = "Maybe"'), "unknown stage"),
    (lambda t: t.replace('licence = "test fixture"', 'licence = "x"\nlicense = "typo"'), "unknown key"),
    (lambda t: t.replace('target = "answers.country"', 'target = "email"'), "set by both"),
    (lambda t: t.replace(f'test_from = "{TEST_FROM}"', 'test_from = "2026-06-01"'), "after as_of"),
    (lambda t: t.replace('column = "l.stage"', 'column = "stage"'), "column reference"),
])
def test_loader_refuses_bad_mappings(edit, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        load_mapping(edit(MAPPING))


def test_unconfirmed_mapping_loads_for_review_but_never_converts(frames: dict[str, pd.DataFrame]) -> None:
    draft_text = MAPPING.replace("outcome_confirmed = true", "outcome_confirmed = false")
    draft = load_mapping(draft_text)  # a draft can be loaded and saved for review
    assert load_mapping(dump_mapping(draft)) == draft
    with pytest.raises(ValueError, match="draft: outcome_confirmed is false"):
        load_mapping(draft_text, require_confirmed=True)
    with pytest.raises(ValueError, match="outcome_confirmed is false"):
        convert(frames, draft)


# --- conversion ----------------------------------------------------------------------------------------------------

def test_stage_map_value_map_and_crm_rows(converted: dict[str, bytes], frames: dict[str, pd.DataFrame]) -> None:
    L, C = _csv(converted, "historical_leads.csv"), _csv(converted, "crm_history.csv")
    src = frames["leads.csv"].set_index("id")
    L = L.set_index("lead_id")
    ppc = src.index[src.source == "ppc"]
    assert (L.loc[ppc, "utm_source"] == "google").all() and (L.loc[ppc, "utm_medium"] == "cpc").all()
    seo = src.index[src.source == "seo"]
    assert L.loc[seo, "utm_source"].isna().all() and (L.loc[seo, "utm_medium"] == "organic").all()
    assert L.loc[src.index[src.source.isna()], "utm_medium"].isna().all()
    answers = L.answers.map(json.loads)
    assert answers.loc[src.index[src["size"] == "large"][0]]["company_size"] == "1000+"
    assert "company_size" not in answers.loc[src.index[src["size"].isna()][0]]  # blanks omitted, not invented
    final = C.groupby("lead_id").tail(1).set_index("lead_id").stage
    expected = src.stage.map({"Closed Won": "Won", "Closed Lost": "Lost", "In progress": "Qualified", "Open": "New"})
    assert final.reindex(expected.index).equals(expected.rename("stage"))
    won_id = src.index[src.stage == "Closed Won"][0]
    rows = C[C.lead_id == won_id]
    assert list(rows.stage) == ["New", "Contacted", "Won"]
    assert float(rows.deal_value.iloc[-1]) == float(src.amount[won_id])
    assert list(C[C.lead_id == src.index[src.stage == "Open"][0]].stage) == ["New"]
    times = C.assign(t=pd.to_datetime(C.changed_at)).groupby("lead_id").t
    assert (times.diff().dropna() > pd.Timedelta(0)).all()  # strictly ordered per lead


def test_unmapped_columns_are_blank_and_email_placeholder_is_derived(frames: dict[str, pd.DataFrame],
                                                                     mapping: DatasetMapping) -> None:
    no_email = replace(mapping, fields=tuple(f for f in mapping.fields if f.source != "l.mail"))
    files = convert(frames, no_email)
    L = _csv(files, "historical_leads.csv")
    assert tuple(L.columns) == LEAD_COLUMNS
    assert (L.email == L.lead_id + "@" + PLACEHOLDER_DOMAIN).all()
    assert (L.form_variant == "A").all()
    blank = [c for c in LEAD_COLUMNS if c not in {"lead_id", "created_at", "answers", "email", "form_variant",
                                                  "utm_source", "utm_medium"}]
    assert L[blank].isna().all().all()
    meta = json.loads(files["dataset.json"])
    assert meta["placeholder_emails"] == len(L)
    assert all(c["status"] == "unfilled" for c in meta["coverage"] if c["feature"] == "email")


def test_won_flag_outcome(frames: dict[str, pd.DataFrame], mapping: DatasetMapping) -> None:
    f = {**frames, "leads.csv": frames["leads.csv"].assign(
        won=np.where(frames["leads.csv"].stage == "Closed Won", "yes", "no"),
        closed=frames["leads.csv"].closed.fillna(frames["leads.csv"].signup))}
    m = replace(mapping, outcome=replace(mapping.outcome, kind="won_flag", column="l.won", stage_map={},
                                         won_values=("yes",), lost_values=("no",)))
    C = _csv(convert(f, m), "crm_history.csv")
    final = C.groupby("lead_id").tail(1).set_index("lead_id").stage
    assert (final == "Won").sum() == (frames["leads.csv"].stage == "Closed Won").sum()
    assert set(final) == {"Won", "Lost"}
    with pytest.raises(ValueError, match=r"does not cover: \['maybe'\]"):
        convert({**f, "leads.csv": f["leads.csv"].assign(won=lambda d: d.won.mask(d.index == 3, "maybe"))}, m)


def test_presence_outcome_dates_lost_at_as_of_only_when_asked(frames: dict[str, pd.DataFrame],
                                                            mapping: DatasetMapping) -> None:
    m = replace(mapping, outcome=replace(mapping.outcome, kind="presence", column="l.amount", stage_map={}),
                lead=replace(mapping.lead, close_at=None))
    with pytest.raises(ValueError, match="close_at is blank"):
        convert(frames, m)
    files = convert(frames, replace(m, outcome=replace(m.outcome, lost_without_close="as_of")))
    C = _csv(files, "crm_history.csv")
    lost = C[C.stage == "Lost"]
    assert (lost.changed_at == f"{AS_OF}T00:00:00Z").all()
    assert json.loads(files["dataset.json"])["lost_dated_at_as_of"] == len(lost)


def test_won_at_is_optional_and_a_win_without_one_takes_close_at(frames: dict[str, pd.DataFrame],
                                                                 mapping: DatasetMapping) -> None:
    """Spec: "optional won_at". Unmapped, or blank on a row, a Won lead is dated at its close_at (R36)."""
    text = MAPPING.replace('won_at = "l.closed"\n', "")
    m = load_mapping(text)
    assert m.lead.won_at is None and load_mapping(dump_mapping(m)) == m
    assert convert(frames, m)["crm_history.csv"] == convert(frames, mapping)["crm_history.csv"]
    leads = frames["leads.csv"]
    first_win = leads.index[leads.stage == "Closed Won"][0]
    partial = leads.assign(won=leads.closed.mask(leads.index == first_win, None)).assign(
        won=lambda d: d.won.where(d.stage == "Closed Won"))
    m2 = load_mapping(MAPPING.replace('won_at = "l.closed"', 'won_at = "l.won"'))
    assert convert({**frames, "leads.csv": partial}, m2)["crm_history.csv"] == \
        convert(frames, mapping)["crm_history.csv"]


def test_a_win_before_the_lead_is_moved_counted_listed_and_warned(frames: dict[str, pd.DataFrame],
                                                                  mapping: DatasetMapping) -> None:
    """R37: the conversion keeps going, but dataset.json names the leads and validation warns with them."""
    leads = frames["leads.csv"]
    win = leads.index[leads.stage == "Closed Won"][0]
    early = (pd.Timestamp(leads.signup[win]) - pd.Timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    files = convert({**frames, "leads.csv": leads.assign(closed=leads.closed.mask(leads.index == win, early))}, mapping)
    meta = json.loads(files["dataset.json"])
    assert meta["crm_times_reordered"] == 1 and meta["crm_times_reordered_ids"] == [leads.id[win]]
    C = _csv(files, "crm_history.csv")
    rows = C[C.lead_id == leads.id[win]]
    assert list(rows.stage) == ["New", "Contacted", "Won"] and rows.changed_at.is_monotonic_increasing
    report = validate_files({**files, "mapping.toml": MAPPING.encode()})
    warned = [w for w in report.warnings if w.file == "crm_history.csv" and w.column == "changed_at"]
    assert len(warned) == 1 and leads.id[win] in warned[0].message and "before" in warned[0].message
    assert json.loads(convert(frames, mapping)["dataset.json"])["crm_times_reordered_ids"] == []


@pytest.mark.parametrize("change, match", [
    (lambda d: d.assign(stage=d.stage.mask(d.index == 5, "Negotiation")), r"does not cover: \['Negotiation'\]"),
    (lambda d: d.assign(signup=d.signup.mask(d.index == 5, None)), "created_at is blank"),
    (lambda d: d.assign(id=d.id.mask(d.index == 5, "X0000")), "lead_id repeats"),
    (lambda d: d.assign(closed=d.closed.mask(d.stage == "Closed Won", None)), "won_at and close_at are both blank"),
    (lambda d: d.assign(closed=d.closed.mask(d.index == d.index[d.stage == "Closed Won"][0], "2026-03-01")),
     "after as_of"),
    (lambda d: d.assign(signup=d.signup.mask(d.index == 5, "last Tuesday")), "not ISO dates"),
    (lambda d: d.assign(source=d.source.mask(d.index == 5, "tiktok")), r"\['tiktok'\] are not in its value_map"),
    (lambda d: d.assign(amount=d.amount.mask(d.stage == "Closed Won", "lots")), "not numbers"),
])
def test_conversion_refuses_what_the_mapping_does_not_cover(frames: dict[str, pd.DataFrame],
                                                            mapping: DatasetMapping, change, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        convert({**frames, "leads.csv": change(frames["leads.csv"])}, mapping)


def test_values_outside_form_options_are_refused(frames: dict[str, pd.DataFrame], mapping: DatasetMapping) -> None:
    bad = load_mapping(MAPPING.replace('large = { "answers.company_size" = "1000+" }',
                                       'large = { "answers.company_size" = "huge" }'))
    with pytest.raises(ValueError, match=r"answers.company_size: value\(s\) \['huge'\] are not form options"):
        convert(frames, bad)


def test_blank_created_at_rows_can_be_dropped_and_are_counted(frames: dict[str, pd.DataFrame],
                                                               mapping: DatasetMapping) -> None:
    f = {**frames, "leads.csv": frames["leads.csv"].assign(signup=lambda d: d.signup.mask(d.index < 3, None))}
    meta = json.loads(convert(f, replace(mapping, drop_rows_without_created_at=True))["dataset.json"])
    assert meta["rows_dropped_without_created_at"] == 3 and meta["leads"] == len(frames["leads.csv"]) - 3


def test_join_refuses_a_repeated_key(frames: dict[str, pd.DataFrame], mapping: DatasetMapping) -> None:
    orgs = pd.concat([frames["orgs.csv"], frames["orgs.csv"].head(1)])
    with pytest.raises(ValueError, match="join key 'org' repeats"):
        convert({**frames, "orgs.csv": orgs}, mapping)


def test_two_won_rows_are_rejected_by_io_load(converted: dict[str, bytes], tmp_path: Path) -> None:
    """The converter refuses repeated lead ids (test above), so it cannot write two Won rows for a lead; a converted
    dataset edited to hold two is still rejected by ``emva.io.load``."""
    for name, content in converted.items():
        (tmp_path / name).write_bytes(content)
    load(tmp_path)  # the converted dataset loads
    C = _csv(converted, "crm_history.csv")
    won = C[C.stage == "Won"].head(1).assign(changed_at="2025-12-01T00:00:00Z")
    pd.concat([C, won]).to_csv(tmp_path / "crm_history.csv", index=False)
    with pytest.raises(ValueError, match="more than one Won row"):
        load(tmp_path)


def test_coverage_counts(converted: dict[str, bytes], frames: dict[str, pd.DataFrame]) -> None:
    meta = json.loads(converted["dataset.json"])
    cov = {c["column"]: c for c in meta["coverage"]}
    assert list(cov) == fixed_columns(CATS_V2) and len(cov) == 39
    n = len(frames["leads.csv"])  # no bots (blank time on page) and no repeated emails: every lead is kept
    assert all(c["leads"] == n for c in cov.values())
    assert cov["channel=meta"]["leads_with_1"] == (frames["leads.csv"].source == "fb").sum()
    assert cov["channel=meta"]["status"] == "data" and cov["channel=meta"]["inputs_mapped"] == ["utm_source",
                                                                                               "utm_medium"]
    assert cov["channel=linkedin"]["status"] == "constant"  # inputs mapped, no linkedin lead
    assert cov["band=1000+"]["status"] == "data"
    assert cov["session_missing=yes"] == {**cov["session_missing=yes"], "leads_with_1": n, "status": "unfilled"}
    assert cov["email=free"]["status"] == "constant"
    assert meta["mapped_targets"] == ["email", "utm_source", "utm_medium", "answers.company_size", "answers.country"]
    assert (meta["as_of"], meta["test_from"], meta["emva_dataset_format"]) == (AS_OF, TEST_FROM, 1)


def test_converted_output_passes_app_validation(converted: dict[str, bytes]) -> None:
    report = validate_files(_training(converted))
    assert report.ok, report.errors


# --- pipeline and report on a converted dataset ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def converted_dir(converted: dict[str, bytes], tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("converted")
    for name, content in converted.items():
        (d / name).write_bytes(content)
    return d


def test_pipeline_reads_dates_from_dataset_json(converted_dir: Path) -> None:
    r = run(converted_dir)
    assert (r.as_of, r.test_from) == (pd.Timestamp(AS_OF, tz="UTC"), TEST_FROM)
    created = r.X.created_at
    assert created[r.test].min() >= pd.Timestamp(TEST_FROM, tz="UTC") > created[r.train].max()
    assert (created[r.train | r.test] + pd.Timedelta(days=120)).max() <= pd.Timestamp(AS_OF, tz="UTC")
    assert r.test.sum() > 0 and r.X.y[r.test].nunique() == 2
    assert r.weights.shape[0] > 1  # fitted


def test_report_runs_on_a_converted_dataset(converted_dir: Path) -> None:
    from emva.eval.report import build_report

    text = build_report(converted_dir, n_resamples=20)
    assert "Baseline omitted" in text and "Status quo omitted" in text
    assert f"test_from {TEST_FROM} (from dataset.json)" in text
    assert "| candidate |" in text and "| baseline |" not in text


# --- profiles and redaction ----------------------------------------------------------------------------------------

EMAILS = ["ann.lee@acme.co.uk", "bob@example.com"]
PHONES = ["+44 20 7946 0958", "(555) 123-4567", "07700 900123"]
IPS = ["203.0.113.67", "2001:db8:85a3::8a2e:370:7334", "2001:0db8:0000:0000:0000:ff00:0042:8329"]


def _leaky_frames() -> dict[str, pd.DataFrame]:
    n = 12
    return {"raw.csv": pd.DataFrame({
        "email": [EMAILS[i % 2] for i in range(n)],
        "phone": [PHONES[i % 3] for i in range(n)],
        "ip_address": [IPS[i % 3] for i in range(n)],
        "comment": [f"call me on {PHONES[i % 3]} or mail {EMAILS[i % 2]} from {IPS[i % 3]}" for i in range(n)],
        "contact_name": [["Ann Lee", "Bob Smith"][i % 2] for i in range(n)],
        "manager": [["Cara Losch", "Dustin Brinkmann", "Rocco Neubert"][i % 3] for i in range(n)],
        "handler": [["Summer Sewald", "Celia Rouche"][i % 2] for i in range(n)],  # caught by value
        "hotel": [["City Hotel", "Resort Hotel"][i % 2] for i in range(n)],
        "stage": ["Won", "Lost"] * 6,
        "created": ["2025-01-0%d" % (1 + i % 9) for i in range(n)],
        "amount": [str(1000 + i) for i in range(n)],
        "mobile": [str(447700900000 + i % 2) for i in range(n)],
    }).astype(str)}


def test_no_email_phone_ip_or_name_ever_reaches_a_profile() -> None:
    profiles = profile(_leaky_frames())
    text = render_profiles(profiles)
    for secret in [*EMAILS, *PHONES, *IPS, "Ann Lee", "Bob Smith", "Cara Losch", "Dustin Brinkmann", "Rocco Neubert",
                   "Summer Sewald", "Celia Rouche", "447700900000", "@acme"]:
        assert secret not in text, secret
    by = {p.name: p for p in profiles}
    assert by["manager"].examples == by["handler"].examples == (NAME_TOKEN,)
    assert set(by["hotel"].examples) == {"City Hotel", "Resort Hotel"}  # a shared word: not people
    assert by["comment"].examples and all("<email>" in e and "<phone>" in e and "<ip>" in e
                                          for e in by["comment"].examples)
    assert by["contact_name"].examples == (NAME_TOKEN,)
    assert set(by["stage"].examples) == {"Won", "Lost"} and by["stage"].type == "text"
    assert by["created"].type == "date" and by["created"].min == "2025-01-01"
    assert by["amount"].type == "number" and by["amount"].max == "1011"


def test_person_columns_by_name() -> None:
    from emva.ingest.profile import is_person_column

    person = ["name", "first_name", "firstName", "contact", "contact_name", "manager", "managers", "seller",
              "assigned_to", "created_by", "assignee", "owner", "sdr", "sales_rep", "sales_agent", "buyer", "employee",
              "author", "lead", "team_head", "account_manager", "account_owner", "company_contact", "customer"]
    other = ["company_name", "product_name", "hotel", "lead_id", "lead_type", "lead_time", "lead_source", "seller_id",
             "sdr_id", "customer_type", "owner_status", "lead_behaviour_profile", "origin", "deal_stage", "sector",
             "market_segment", "created_at", "subsidiary_of"]
    assert [c for c in person if not is_person_column(c)] == []
    assert [c for c in other if is_person_column(c)] == []


def test_name_like_values_are_redacted_unless_the_name_rules_it_out() -> None:
    from emva.ingest.profile import looks_like_names, redact_as_names

    assert looks_like_names(["Cara Losch", "Dustin Brinkmann", "Mary-Ann O'Neil Smith"])
    assert not looks_like_names(["City Hotel", "Resort Hotel"])  # last words repeat
    assert not looks_like_names(["paid_search", "social", "Direct Traffic"]) and not looks_like_names([])
    assert redact_as_names("subsidiary", ["Acme Corporation", "Bubba Gump"])  # fail safe: companies may read as names
    assert not redact_as_names("deposit_type", ["No Deposit", "Non Refund", "Refundable"])


def test_high_cardinality_columns_get_no_examples() -> None:
    df = pd.DataFrame({"ref": [f"R{i}" for i in range(100)], "tier": ["a", "b"] * 50})
    by = {p.name: p for p in profile({"f.csv": df})}
    assert by["ref"].examples == () and by["ref"].distinct == 100
    assert by["tier"].examples == ("a", "b") and by["tier"].share_missing == 0


def test_redact_keeps_dates_and_small_numbers() -> None:
    assert redact("signed 2018-02-26 19:58:54, 3 seats, £4500") == "signed 2018-02-26 19:58:54, 3 seats, £4500"
    assert redact("x@y.org / 10.0.0.1 / +1 415 555 0100") == "<email> / <ip> / <phone>"


# --- drafting (fake client, no API) ---------------------------------------------------------------------------------

def _reply() -> dict:
    col = lambda file, column, target, **kw: {"file": file, "column": column, "target": target,  # noqa: E731
                                               "feature_kind": kw.get("kind", "none"),
                                               "value_map": kw.get("vm", []), "reason": "r", "confidence": "high"}
    return {
        "primary_file": "leads.csv", "joins": [{"file": "orgs.csv", "join_on": "org"}],
        "columns": [
            col("leads.csv", "id", "lead.lead_id"), col("leads.csv", "signup", "lead.created_at"),
            col("leads.csv", "closed", "lead.won_at"), col("leads.csv", "amount", "lead.deal_value"),
            col("leads.csv", "mail", "email"), col("leads.csv", "notes", "ignore"),
            col("leads.csv", "source", "value_map", vm=[
                {"source_value": "ppc", "target": "utm_source", "target_value": "google"},
                {"source_value": "ppc", "target": "utm_medium", "target_value": "cpc"},
                {"source_value": "fb", "target": "utm_source", "target_value": "facebook"}]),
            col("orgs.csv", "country", "answers.country")],
        "outcome": {"kind": "stage", "file": "leads.csv", "column": "stage",
                    "stage_map": [{"value": "Closed Won", "stage": "Won"}, {"value": "Closed Lost", "stage": "Lost"}],
                    "won_values": [], "lost_values": [], "reason": "stage column", "confidence": "medium"},
    }


def test_a_draft_needs_created_at_but_not_won_at(frames: dict[str, pd.DataFrame]) -> None:
    from emva.ingest.draft import check_reply

    profiles = profile(frames)
    reply = _reply()
    no_won = {**reply, "columns": [c for c in reply["columns"] if c["target"] != "lead.won_at"]}
    assert check_reply(no_won, profiles) is no_won
    no_created = {**reply, "columns": [c for c in reply["columns"] if c["target"] != "lead.created_at"]}
    with pytest.raises(ContractError, match="created_at"):
        check_reply(no_created, profiles)


def test_a_draft_names_its_primary_file_and_joins_each_other_file_once(frames: dict[str, pd.DataFrame]) -> None:
    from emva.ingest.draft import check_reply

    profiles = profile(frames)
    reply = _reply()
    with pytest.raises(ContractError, match="primary or joined once"):
        check_reply({**reply, "joins": [*reply["joins"], {"file": "leads.csv", "join_on": "org"}]}, profiles)
    with pytest.raises(ContractError, match="primary or joined once"):
        check_reply({**reply, "joins": reply["joins"] * 2}, profiles)
    with pytest.raises(ContractError, match="not in the profiles"):
        check_reply({**reply, "primary_file": "deals.csv"}, profiles)
    with pytest.raises(ContractError, match="not_a_column"):
        check_reply({**reply, "joins": [{"file": "orgs.csv", "join_on": "not_a_column"}]}, profiles)
    single = {**reply, "joins": [], "columns": [c for c in reply["columns"] if c["file"] == "leads.csv"]}
    assert check_reply(single, profiles) is single


def message(text: str, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)])


class FakeClient:
    """Stands in for ``anthropic.Anthropic``: each call pops the next scripted reply or exception."""

    def __init__(self, *script: object) -> None:
        self.script = list(script)
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def test_draft_is_unconfirmed_cached_and_sends_only_profiles(tmp_path: Path, frames: dict[str, pd.DataFrame]) -> None:
    profiles = profile(frames)
    cache = ReplyCache(tmp_path / "cache.json")
    client = FakeClient(message(json.dumps(_reply())))
    m = draft_mapping(profiles, client, cache, "fixture_draft")
    assert not m.outcome_confirmed and m.name == "fixture_draft"
    assert m.lead.created_at == Expr.col("leads.signup") and m.sources[1].join_on == "org"
    assert dict(m.outcome.stage_map) == {"Closed Won": "Won", "Closed Lost": "Lost"}
    vm = next(f for f in m.fields if f.value_map)
    assert dict(vm.value_map["ppc"]) == {"utm_source": "google", "utm_medium": "cpc"}
    assert m.review["created_at"].confidence == "high" and m.review["outcome"].confidence == "medium"
    # dates derived from the created_at profile: the day after the latest date, test from 80% of the span
    last = pd.to_datetime(frames["leads.csv"].closed).max().normalize() + pd.Timedelta(days=1)
    assert m.as_of == last.date().isoformat() and m.test_from == "2025-08-01"
    call = client.calls[0]
    assert call["model"] == "claude-haiku-4-5-20251001" and call["output_config"]["format"]["schema"] == RESPONSE_SCHEMA
    sent = call["messages"][0]["content"]
    assert "buyer1@firm1.example" not in sent and "X0001" not in sent  # no rows, no emails
    assert load_mapping(dump_mapping(m)) == m
    with pytest.raises(ValueError, match="outcome_confirmed is false"):
        convert(frames, m)
    # second call: a cache hit, no request (no client needed), same mapping; the cache survives a reload
    assert is_cached(profiles, ReplyCache(tmp_path / "cache.json"))
    assert draft_mapping(profiles, None, ReplyCache(tmp_path / "cache.json"), "fixture_draft") == m
    assert len(client.calls) == 1


def test_draft_parse_errors_are_cached_and_raised(tmp_path: Path, frames: dict[str, pd.DataFrame]) -> None:
    profiles = profile(frames)
    cache = ReplyCache(tmp_path / "cache.json")
    bad = _reply()
    bad["columns"] = [c for c in bad["columns"] if c["target"] != "lead.created_at"]
    client = FakeClient(message(json.dumps(bad)))
    with pytest.raises(ContractError, match="lead.created_at"):
        draft_mapping(profiles, client, cache, "x")
    with pytest.raises(ContractError, match="cached"):
        draft_mapping(profiles, client, cache, "x")
    assert len(client.calls) == 1
    unknown = _reply()
    unknown["columns"][0]["column"] = "not_there"
    with pytest.raises(ContractError, match="not_there"):
        draft_mapping(profile(fixture_frames(seed=1)), FakeClient(message(json.dumps(unknown))), cache, "x")


def test_draft_retries_transport_errors_and_does_not_cache_them(tmp_path: Path, frames: dict[str, pd.DataFrame],
                                                                caplog: pytest.LogCaptureFixture) -> None:
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    overloaded = anthropic.APIStatusError("overloaded", response=httpx2.Response(529, request=req), body=None)
    client = FakeClient(anthropic.APIConnectionError(request=req), overloaded, message(json.dumps(_reply())))
    waits: list[float] = []
    m = draft_mapping(profile(frames), client, ReplyCache(tmp_path / "c.json"), "x", as_of=AS_OF,
                      test_from=TEST_FROM, sleep=waits.append)
    assert (m.as_of, m.test_from) == (AS_OF, TEST_FROM) and waits == [1, 2] and len(client.calls) == 3
    assert caplog.text.count("transport error") == 2
    denied = anthropic.APIStatusError("bad key", response=httpx2.Response(401, request=req), body=None)
    with pytest.raises(anthropic.APIStatusError):
        draft_mapping(profile(fixture_frames(seed=2)), FakeClient(denied), ReplyCache(tmp_path / "d.json"), "x")
    assert json.loads((tmp_path / "c.json").read_text())["entries"]  # only the successful reply was cached


def test_draft_targets_are_the_schema_plus_roles() -> None:
    enum = RESPONSE_SCHEMA["properties"]["columns"]["items"]["properties"]["target"]["enum"]
    assert set(TARGETS) < set(enum) and {"ignore", "value_map", "lead.created_at"} <= set(enum)


# --- the committed Kaggle mappings on their hand-built fixtures -------------------------------------------------------

@pytest.mark.parametrize("name", ["olist_funnel", "crm_opportunities", "hotel_bookings"])
def test_committed_mapping_converts_its_fixture(name: str) -> None:
    text = (REPO / "mappings" / f"{name}.toml").read_text()
    m = load_mapping(text, require_confirmed=True)
    assert m.source_url and m.licence and "BY HAND" in text and "NOT an LLM draft" in text
    raw = FIXTURES / name
    files = convert(frames_from_bytes({s.file: (raw / s.file).read_bytes() for s in m.sources}), m)
    report = validate_files(files)  # with dataset.json: it declares the extras of extra_features.csv (Phase 10)
    assert [e.message for e in report.errors] == [f"only {report.summary['leads']} leads; training needs at least "
                                                  f"{MIN_LEADS}"]  # fixtures are a few dozen rows
    meta = json.loads(files["dataset.json"])
    assert len(meta["coverage"]) == 39 and meta["mapping"] == name


def test_cli_convert_writes_the_dataset_and_refuses_a_draft(tmp_path: Path,
                                                            capsys: pytest.CaptureFixture[str]) -> None:
    from emva.ingest.__main__ import main

    main(["convert", "--mapping", str(REPO / "mappings" / "olist_funnel.toml"), "--raw",
          str(FIXTURES / "olist_funnel"), "--out", str(tmp_path / "out")])
    printed = capsys.readouterr().out
    from emva.ingest.convert import DERIVED_COUNTS

    meta = json.loads((tmp_path / "out" / "dataset.json").read_text())
    assert all(f"{k} = {meta[k]}: " in printed for k in DERIVED_COUNTS) and meta["lost_dated_at_as_of"] > 0
    assert {p.name for p in (tmp_path / "out").iterdir()} == {
        "historical_leads.csv", "crm_history.csv", "companies.csv", "people.csv", "dataset.json", "mapping.toml",
        "extra_features.csv"}  # the mapping declares [[features]] (Phase 10)
    draft = tmp_path / "draft.toml"
    draft.write_text((REPO / "mappings" / "olist_funnel.toml").read_text().replace(
        "outcome_confirmed = true", "outcome_confirmed = false"))
    with pytest.raises(ValueError, match="outcome_confirmed is false"):
        main(["convert", "--mapping", str(draft), "--raw", str(FIXTURES / "olist_funnel"), "--out",
              str(tmp_path / "x")])


# --- scoring new leads given in the source format --------------------------------------------------------------------

def test_source_columns_split_submit_time_from_outcome(mapping: DatasetMapping) -> None:
    assert mapping.source_columns() == {"leads.csv": ["id", "signup", "mail", "source", "size", "org"],
                                        "orgs.csv": ["org", "country"]}
    everything = mapping.source_columns(submit_time_only=False)
    assert {"stage", "closed", "amount", "notes"} <= set(everything["leads.csv"])
    assert not {"stage", "closed", "amount", "notes"} & set(mapping.source_columns()["leads.csv"])


def test_new_source_rows_score_like_the_training_run(converted_dir: Path, frames: dict[str, pd.DataFrame],
                                                     mapping: DatasetMapping, tmp_path: Path) -> None:
    """Train on the converted fixture, then score five of its leads again from raw rows without any outcome column:
    the probabilities equal the batch run's (ADR 0018)."""
    from emva.persist import load_bundle, save_bundle
    from emva.scoring import score_leads

    r = run(converted_dir)
    save_bundle(r, tmp_path / "model.joblib", converted_dir, 1.0)
    new = frames["leads.csv"].iloc[[0, 1, 2, 3, 40]].drop(columns=["stage", "closed", "amount", "notes"])
    leads = convert_leads({"leads.csv": new, "orgs.csv": frames["orgs.csv"]}, mapping)
    assert list(leads.lead_id) == list(new.id) and tuple(leads.columns) == LEAD_COLUMNS
    scores = score_leads(load_bundle(tmp_path / "model.joblib"), leads, converted_dir)
    assert np.isfinite(scores.p_formula).all()
    np.testing.assert_allclose(scores.p_formula, r.X.p_formula.loc[new.id], rtol=1e-12)
    np.testing.assert_allclose(scores.value_at_submit, r.X.value_at_submit.loc[new.id], rtol=1e-12)


def test_form_text_in_a_column_blank_on_every_training_lead_scores(converted_dir: Path, frames: dict[str, pd.DataFrame],
                                                                  mapping: DatasetMapping, tmp_path: Path) -> None:
    """Phase 9 bug: the fixture maps no landing_url / ip_country, so they are blank on every training lead and the
    bundle's lead schema reads them as float64; a form lead that fills them in used to fail to parse its text as a
    number. It now scores, and leads from the source rows still score exactly like the batch run."""
    from emva.persist import load_bundle, save_bundle
    from emva.scoring import lead_from_form, score_leads

    r = run(converted_dir)
    save_bundle(r, tmp_path / "model.joblib", converted_dir, 1.0)
    bundle = load_bundle(tmp_path / "model.joblib")
    assert pd.api.types.is_float_dtype(bundle.lead_schema.dtypes["landing_url"])
    assert pd.api.types.is_float_dtype(bundle.lead_schema.dtypes["ip_country"])
    form = lead_from_form({"email": "ann@firm.example", "form_variant": "A", "utm_source": "google",
                           "utm_medium": "cpc", "ip_country": "UK", "landing_url": "https://firm.example/p"})
    assert np.isfinite(score_leads(bundle, form, converted_dir).p_formula).all()
    new = frames["leads.csv"].iloc[[0, 1, 2, 3, 40]].drop(columns=["stage", "closed", "amount", "notes"])
    leads = convert_leads({"leads.csv": new, "orgs.csv": frames["orgs.csv"]}, mapping)
    np.testing.assert_allclose(score_leads(bundle, leads, converted_dir).p_formula, r.X.p_formula.loc[new.id],
                               rtol=1e-12)


def test_convert_leads_without_ids_dates_or_outcome_columns(frames: dict[str, pd.DataFrame],
                                                            mapping: DatasetMapping) -> None:
    new = frames["leads.csv"].head(3)[["mail", "source", "size", "org"]]
    leads = convert_leads({"leads.csv": new, "orgs.csv": frames["orgs.csv"]}, mapping)
    assert list(leads.lead_id) == ["new-000001", "new-000002", "new-000003"]
    assert leads.created_at.notna().all() and str(leads.created_at.dt.tz) == "UTC"
    with pytest.raises(ValueError, match="column 'l.source' is not in the source files"):
        convert_leads({"leads.csv": new.drop(columns="source"), "orgs.csv": frames["orgs.csv"]}, mapping)
    with pytest.raises(ValueError, match="outcome_confirmed is false"):
        convert_leads({"leads.csv": new, "orgs.csv": frames["orgs.csv"]}, replace(mapping, outcome_confirmed=False))


def test_convert_leads_types_booleans_for_scoring() -> None:
    m = load_mapping((REPO / "mappings" / "hotel_bookings.toml").read_text())
    raw = pd.read_csv(FIXTURES / "hotel_bookings" / "hotel_bookings.csv", dtype=str).head(10)
    leads = convert_leads({"hotel_bookings.csv": raw.drop(columns=["is_canceled", "reservation_status_date"])}, m)
    assert set(leads.returned_visitor) <= {True, False} and leads.lead_id.iloc[0] == "new-000001"


def test_cli_score_prints_scores(converted_dir: Path, frames: dict[str, pd.DataFrame], tmp_path: Path,
                                 capsys: pytest.CaptureFixture[str]) -> None:
    from emva.ingest.__main__ import main
    from emva.persist import save_bundle

    save_bundle(run(converted_dir), tmp_path / "model.joblib", converted_dir, 1.0)
    (tmp_path / "mapping.toml").write_text(MAPPING)
    frames["leads.csv"].head(4).drop(columns=["stage", "closed", "amount"]).to_csv(tmp_path / "new.csv", index=False)
    frames["orgs.csv"].to_csv(tmp_path / "orgs.csv", index=False)
    main(["score", "--mapping", str(tmp_path / "mapping.toml"), "--raw", str(tmp_path / "new.csv"), "--run",
          str(tmp_path), "--data", str(converted_dir)])
    out = pd.read_csv(io.StringIO(capsys.readouterr().out), index_col="lead_id")
    assert list(out.index) == list(frames["leads.csv"].id.head(4)) and out.p_formula.between(0, 1).all()


def test_cli_draft_uses_the_cache_next_to_the_output(tmp_path: Path, frames: dict[str, pd.DataFrame]) -> None:
    """With the reply cached (here by a fake client) the CLI needs no API key and writes an unconfirmed draft."""
    from emva.ingest.__main__ import main

    raw = tmp_path / "raw"
    raw.mkdir()
    for name, df in frames.items():
        df.to_csv(raw / name, index=False)
    on_disk = frames_from_bytes({p.name: p.read_bytes() for p in sorted(raw.glob("*.csv"))})
    draft_mapping(profile(on_disk), FakeClient(message(json.dumps(_reply()))),
                  ReplyCache(tmp_path / ".draft_cache.json"), "ignored")
    main(["draft", "--raw", str(raw), "--out", str(tmp_path / "fixture_draft.toml")])
    text = (tmp_path / "fixture_draft.toml").read_text()
    assert text.startswith("# DRAFT by claude-haiku-4-5-20251001")
    m = load_mapping(text)
    assert m.name == "fixture_draft" and not m.outcome_confirmed


# --- ground rule 2: the converter never reads evaluation-only files ------------------------------------------------

def test_is_evaluation_only_matches_ground_truth_names_only() -> None:
    from emva.eval.evaluation_only import is_evaluation_only

    assert is_evaluation_only("ground_truth_labels.csv") and is_evaluation_only("data/v1/Ground_Truth.md")
    assert not is_evaluation_only("historical_leads.csv") and not is_evaluation_only("truth_ground.csv")


def test_source_file_must_be_a_plain_csv_name_and_not_ground_truth(mapping: DatasetMapping) -> None:
    from emva.ingest.mapping import Source

    assert Source("x", "Leads.CSV").file == "Leads.CSV"
    for bad, match in [("../leads.csv", "plain file name"), ("sub/leads.csv", "plain file name"),
                       ("..\\leads.csv", "plain file name"), ("", "plain file name"), ("leads.txt", ".csv file"),
                       ("ground_truth_labels.csv", "evaluation-only")]:
        with pytest.raises(ValueError, match=match):
            Source("x", bad)
    with pytest.raises(ValueError, match="evaluation-only"):
        load_mapping(MAPPING.replace('file = "leads.csv"', 'file = "ground_truth_labels.csv"'))


def test_frames_from_bytes_refuses_ground_truth() -> None:
    with pytest.raises(ValueError, match="evaluation-only"):
        frames_from_bytes({"leads.csv": b"a\n1\n", "ground_truth_labels.csv": b"a\n1\n"})


def test_cli_draft_refuses_ground_truth_and_more_than_three_files(tmp_path: Path) -> None:
    from emva.ingest.__main__ import main

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "leads.csv").write_text("a\n1\n")
    (raw / "ground_truth_labels.csv").write_text("a\n1\n")
    with pytest.raises(SystemExit, match="ground_truth_labels.csv"):
        main(["draft", "--raw", str(raw), "--out", str(tmp_path / "d.toml")])
    (raw / "ground_truth_labels.csv").unlink()
    for i in range(3):
        (raw / f"extra{i}.csv").write_text("a\n1\n")
    with pytest.raises(SystemExit, match="1 to 3 sources"):
        main(["draft", "--raw", str(raw), "--out", str(tmp_path / "d.toml")])
    assert not (tmp_path / "d.toml").exists()


def test_cli_score_refuses_a_ground_truth_file(tmp_path: Path) -> None:
    from emva.ingest.__main__ import main

    (tmp_path / "mapping.toml").write_text(MAPPING)
    (tmp_path / "ground_truth_labels.csv").write_text("id\nx\n")
    with pytest.raises(SystemExit, match="evaluation-only"):
        main(["score", "--mapping", str(tmp_path / "mapping.toml"), "--raw", str(tmp_path / "ground_truth_labels.csv"),
              "--run", str(tmp_path)])


def test_dump_escapes_what_toml_forbids_raw_and_refuses_lone_surrogates(mapping: DatasetMapping) -> None:
    from emva.ingest.mapping import Review

    odd = replace(mapping, review={"created_at": Review("tab\there, del \x7f, bell \x07, nul \x00, é ✓", "high")})
    text = dump_mapping(odd)
    assert "\x7f" not in text and "\\u007f" in text
    assert load_mapping(text) == odd
    with pytest.raises(ValueError, match="lone surrogate"):
        dump_mapping(replace(mapping, review={"created_at": Review("broken \ud800")}))


# --- the committed live drafts (Phase 10 live draft check) ----------------------------------------------------------

@pytest.mark.parametrize("name", ["olist", "crm_opportunities", "hotel_bookings"])
def test_committed_live_draft_stays_an_unconfirmed_draft(name: str) -> None:
    text = (REPO / "mappings" / "drafts" / f"{name}.draft.toml").read_text()
    m = load_mapping(text)
    assert text.startswith("# DRAFT by claude-haiku-4-5-20251001") and "BY HAND" not in text
    assert not m.outcome_confirmed and not m.features_confirmed
    with pytest.raises(ValueError, match="outcome_confirmed is false"):
        load_mapping(text, require_confirmed=True)


def test_committed_draft_cache_holds_a_current_reply_per_dataset() -> None:
    from emva.ingest.draft import PROMPT_VERSION, prompt_fingerprint

    entries = json.loads((REPO / "mappings" / "drafts" / "draft_cache.json").read_text())["entries"].values()
    current = [e for e in entries if e["prompt_fingerprint"] == prompt_fingerprint()]
    assert len(current) == 3 and all(e["status"] == "ok" and e["prompt_version"] == PROMPT_VERSION for e in current)
    assert all(e["model_id"] == "claude-haiku-4-5-20251001" for e in entries)
