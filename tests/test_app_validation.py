"""app.validation: the sample data passes; each class of broken upload is reported per file and column."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import pytest

from app.validation import MIN_LEADS, REQUIRED_COLUMNS, ValidationReport, validate_dir, validate_files

DATA_V1 = Path(__file__).resolve().parents[1] / "data" / "v1"
FILES = ("historical_leads.csv", "crm_history.csv", "companies.csv", "people.csv", "status_quo_rules.json")


@pytest.fixture(scope="module")
def good() -> dict[str, bytes]:
    """The v1 training files, cut to the first 400 leads (and their CRM rows) to keep the tests fast."""
    L = pd.read_csv(DATA_V1 / "historical_leads.csv", dtype=str).head(400)
    C = pd.read_csv(DATA_V1 / "crm_history.csv", dtype=str)
    C = C[C.lead_id.isin(L.lead_id)]
    out = {n: (DATA_V1 / n).read_bytes() for n in ("companies.csv", "people.csv", "status_quo_rules.json")}
    return {**out, "historical_leads.csv": L.to_csv(index=False).encode(), "crm_history.csv": C.to_csv(index=False).encode()}


def _edit(files: dict[str, bytes], name: str, fn) -> dict[str, bytes]:
    """A copy of ``files`` with CSV ``name`` passed through ``fn(frame) -> frame``."""
    df = fn(pd.read_csv(io.BytesIO(files[name]), dtype=str))
    return {**files, name: df.to_csv(index=False).encode()}


def _find(r: ValidationReport, file: str, column: str | None, text: str) -> None:
    """Assert an error on ``file``/``column`` whose message contains ``text``."""
    hits = [i for i in r.errors if i.file == file and i.column == column and text in i.message]
    assert hits, f"no error {file}/{column} ~ {text!r}; got {r.errors}"


def test_sample_data_is_valid() -> None:
    r = validate_dir(DATA_V1)
    assert r.ok, r.errors
    assert r.summary["leads"] == 10000 and r.summary["won_leads"] == 1199
    assert r.summary["created_from"] == "2025-09-24" and sum(r.summary["final_stage_counts"].values()) == 10000
    assert [w.column for w in r.warnings] == ["deal_value"]  # Won rows without a value


def test_subsample_is_valid(good: dict[str, bytes]) -> None:
    assert validate_files(good).ok


def test_required_columns_cover_what_the_pipeline_reads() -> None:
    leads = set(REQUIRED_COLUMNS["historical_leads.csv"])
    assert {"lead_id", "created_at", "answers", "email", "time_on_page_s", "landing_url", "pages_visited"} <= leads
    assert "company_name" not in leads  # optional in emva.io


def test_missing_column(good: dict[str, bytes]) -> None:
    r = validate_files(_edit(good, "crm_history.csv", lambda d: d.drop(columns="changed_at")))
    _find(r, "crm_history.csv", "changed_at", "required column is missing")


def test_missing_required_file(good: dict[str, bytes]) -> None:
    r = validate_files({k: v for k, v in good.items() if k != "companies.csv"})
    _find(r, "companies.csv", None, "required file is missing")


def test_non_numeric_deal_value(good: dict[str, bytes]) -> None:
    def bad(d: pd.DataFrame) -> pd.DataFrame:
        d.loc[d.index[3], "deal_value"] = "ten grand"
        return d
    r = validate_files(_edit(good, "crm_history.csv", bad))
    _find(r, "crm_history.csv", "deal_value", "not numbers")
    assert r.errors[0].rows == (4,)


def test_unparsable_date(good: dict[str, bytes]) -> None:
    def bad(d: pd.DataFrame) -> pd.DataFrame:
        d.loc[d.index[0], "created_at"] = "last tuesday"
        return d
    _find(validate_files(_edit(good, "historical_leads.csv", bad)), "historical_leads.csv", "created_at", "not ISO")


def test_non_json_answers(good: dict[str, bytes]) -> None:
    def bad(d: pd.DataFrame) -> pd.DataFrame:
        d.loc[d.index[[1, 2]], "answers"] = "{country: UK"
        return d
    r = validate_files(_edit(good, "historical_leads.csv", bad))
    _find(r, "historical_leads.csv", "answers", "2 row(s) are not a JSON object")


def test_answer_outside_form_options(good: dict[str, bytes]) -> None:
    def bad(d: pd.DataFrame) -> pd.DataFrame:
        d.loc[d.index[0], "answers"] = json.dumps({"budget": "loads"})
        return d
    _find(validate_files(_edit(good, "historical_leads.csv", bad)), "historical_leads.csv", "answers", "budget")


def test_unknown_stage(good: dict[str, bytes]) -> None:
    def bad(d: pd.DataFrame) -> pd.DataFrame:
        d.loc[d.index[5], "stage"] = "Negotiation"
        return d
    _find(validate_files(_edit(good, "crm_history.csv", bad)), "crm_history.csv", "stage", "Negotiation")


def test_orphan_crm_lead_ids(good: dict[str, bytes]) -> None:
    def bad(d: pd.DataFrame) -> pd.DataFrame:
        d.loc[d.index[0], "lead_id"] = "L99999"
        return d
    _find(validate_files(_edit(good, "crm_history.csv", bad)), "crm_history.csv", "lead_id", "not in historical_leads")


def test_lead_without_crm_rows(good: dict[str, bytes]) -> None:
    first = pd.read_csv(io.BytesIO(good["historical_leads.csv"]), dtype=str).lead_id.iloc[0]
    r = validate_files(_edit(good, "crm_history.csv", lambda d: d[d.lead_id != first]))
    _find(r, "historical_leads.csv", "lead_id", "no row in crm_history.csv")


def test_two_won_rows(good: dict[str, bytes]) -> None:
    def bad(d: pd.DataFrame) -> pd.DataFrame:
        won = d[d.stage.str.lower().isin(["won", "closed won"])].iloc[[0]]
        return pd.concat([d, won])
    _find(validate_files(_edit(good, "crm_history.csv", bad)), "crm_history.csv", "stage", "more than one Won")


def test_too_few_leads(good: dict[str, bytes]) -> None:
    small = _edit(good, "historical_leads.csv", lambda d: d.head(MIN_LEADS - 1))
    small = _edit(small, "crm_history.csv", lambda d: d[d.lead_id.isin(
        pd.read_csv(io.BytesIO(small["historical_leads.csv"]), dtype=str).lead_id)])
    _find(validate_files(small), "historical_leads.csv", None, f"training needs at least {MIN_LEADS}")


def test_duplicate_company_domain_and_unknown_sector(good: dict[str, bytes]) -> None:
    def bad(d: pd.DataFrame) -> pd.DataFrame:
        d.loc[d.index[1], "domain"] = d.domain.iloc[0].upper()
        d.loc[d.index[2], "sector"] = "Space"
        return d
    r = validate_files(_edit(good, "companies.csv", bad))
    _find(r, "companies.csv", "domain", "duplicate domain")
    _find(r, "companies.csv", "sector", "Space")


def test_ground_truth_file_rejected_without_parsing(good: dict[str, bytes]) -> None:
    r = validate_files({**good, "ground_truth_labels.csv": b"\x00 not even text"})
    _find(r, "ground_truth_labels.csv", None, "ground-truth")


def test_rules_optional_but_checked(good: dict[str, bytes]) -> None:
    r = validate_files({k: v for k, v in good.items() if k != "status_quo_rules.json"})
    assert r.ok and any(w.file == "status_quo_rules.json" for w in r.warnings)
    _find(validate_files({**good, "status_quo_rules.json": b"{not json"}), "status_quo_rules.json", None, "not valid JSON")
    _find(validate_files({**good, "status_quo_rules.json": b'{"value_rules": []}'}), "status_quo_rules.json",
          "lead_score", "missing")


def test_garbage_csv_never_raises(good: dict[str, bytes]) -> None:
    r = validate_files({**good, "historical_leads.csv": b"\xff\xfe\x00garbage", "companies.csv": b""})
    assert not r.ok and {i.file for i in r.errors} >= {"historical_leads.csv", "companies.csv"}


# --- a converted dataset's metadata files (Phase 9) ------------------------------------------------------------------

@pytest.fixture(scope="module")
def converted() -> dict[str, bytes]:
    """The hand-built Olist-shaped fixture converted with the committed mappings/olist_funnel.toml
    (``emva.ingest.convert``), with that mapping.toml and the converter's dataset.json."""
    from emva.ingest.convert import convert, frames_from_bytes
    from emva.ingest.mapping import load_mapping

    from conftest import REPO, olist_raw

    text = (REPO / "mappings" / "olist_funnel.toml").read_text(encoding="utf-8")
    return {**convert(frames_from_bytes(olist_raw()), load_mapping(text)), "mapping.toml": text.encode()}


def _warnings(r: ValidationReport, file: str) -> list[str]:
    return [i.message for i in r.warnings if i.file == file]


def test_converted_files_with_metadata_are_valid(converted: dict[str, bytes]) -> None:
    r = validate_files(converted)
    assert r.ok, r.errors
    assert any("leaves out the status-quo row" in m for m in _warnings(r, "status_quo_rules.json"))
    assert not any("snapshot" in m for m in _warnings(r, "historical_leads.csv"))


def test_bad_dataset_json_is_an_error(converted: dict[str, bytes]) -> None:
    for content, text in ((b"not json", "not valid JSON"), (b'{"as_of": "2018-11-15"}', "'as_of' and 'test_from'"),
                          (b'{"as_of": "2018-01-01", "test_from": "2018-03-01"}', "after as_of"),
                          (b'{"as_of": "2018-11-15", "test_from": "2018-3-1"}', "test_from must be")):
        _find(validate_files({**converted, "dataset.json": content}), "dataset.json", None, text)


def test_unconfirmed_or_broken_mapping_is_an_error(converted: dict[str, bytes]) -> None:
    text = converted["mapping.toml"].decode()
    draft = text.replace("outcome_confirmed = true", "outcome_confirmed = false").encode()
    _find(validate_files({**converted, "mapping.toml": draft}), "mapping.toml", None, "outcome_confirmed is false")
    _find(validate_files({**converted, "mapping.toml": b"name = "}), "mapping.toml", None, "not valid TOML")


def test_one_metadata_file_without_the_other_is_a_note(converted: dict[str, bytes]) -> None:
    alone = {k: v for k, v in converted.items() if k not in ("dataset.json", "extra_features.csv")}
    r = validate_files(alone)
    assert r.ok and any("without dataset.json" in m for m in _warnings(r, "mapping.toml"))
    r = validate_files({k: v for k, v in converted.items() if k != "mapping.toml"})
    assert r.ok and any("without mapping.toml" in m for m in _warnings(r, "dataset.json"))


def test_snapshot_warning_uses_the_datasets_own_as_of(converted: dict[str, bytes]) -> None:
    meta = json.loads(converted["dataset.json"])
    early = {**converted, "dataset.json": json.dumps({**meta, "as_of": "2018-04-15",
                                                      "test_from": "2018-03-01"}).encode()}
    r = validate_files(early)
    msgs = _warnings(r, "historical_leads.csv")
    assert any("after 2018-04-15, the dataset's snapshot date (dataset.json)" in m for m in msgs), msgs
    # without dataset.json the pipeline's AS_OF (2026) applies, so none of these 2017-18 leads is "after" it
    plain = validate_files({k: v for k, v in converted.items() if k not in ("dataset.json", "mapping.toml")})
    assert not any("snapshot date" in m for m in _warnings(plain, "historical_leads.csv"))


def test_ground_truth_is_still_refused_next_to_metadata(converted: dict[str, bytes]) -> None:
    r = validate_files({**converted, "ground_truth_labels.csv": b""})
    _find(r, "ground_truth_labels.csv", None, "ground-truth")


# --- extra_features.csv (Phase 10 (b), the generic feature set) -----------------------------------------------------

def _extras(converted: dict[str, bytes], fn) -> dict[str, bytes]:
    """``converted`` with extra_features.csv passed through ``fn(frame) -> frame`` (read as the pipeline reads it)."""
    E = pd.read_csv(io.BytesIO(converted["extra_features.csv"]), dtype=str, keep_default_na=False, na_values=[""])
    return {**converted, "extra_features.csv": fn(E).to_csv(index=False).encode()}


def _declare(converted: dict[str, bytes], features: list[dict]) -> dict[str, bytes]:
    """``converted`` with dataset.json declaring ``features``."""
    meta = json.loads(converted["dataset.json"])
    return {**converted, "dataset.json": json.dumps({**meta, "features": features}).encode()}


def test_converted_extras_are_valid_and_summarised(converted: dict[str, bytes]) -> None:
    assert "extra_features.csv" in converted  # the committed Olist mapping declares landing_page
    r = validate_files(converted)
    assert r.ok, r.errors
    assert r.summary["extra_features"] == [("landing_page", "categorical")]
    assert "extra_features" not in validate_dir(DATA_V1).summary


def test_extras_columns_must_match_the_declaration(converted: dict[str, bytes]) -> None:
    r = validate_files(_extras(converted, lambda E: E.rename(columns={"landing_page": "page"})))
    _find(r, "extra_features.csv", None, "expected ['lead_id', 'landing_page']")
    r = validate_files(_extras(converted, lambda E: E.assign(more="1")))
    _find(r, "extra_features.csv", None, "expected ['lead_id', 'landing_page']")


def test_extras_lead_ids_unique_and_matching_the_leads(converted: dict[str, bytes]) -> None:
    r = validate_files(_extras(converted, lambda E: pd.concat([E, E.head(1)])))
    _find(r, "extra_features.csv", "lead_id", "duplicate lead_id")
    r = validate_files(_extras(converted, lambda E: E.assign(lead_id=E.lead_id.where(E.index > 0, ""))))
    _find(r, "extra_features.csv", "lead_id", "have no lead_id")
    r = validate_files(_extras(converted, lambda E: E.assign(lead_id=E.lead_id.where(E.index > 0, "nobody"))))
    _find(r, "extra_features.csv", "lead_id", "not in historical_leads.csv")
    _find(r, "extra_features.csv", "lead_id", "1 lead(s) of historical_leads.csv have no row here")
    r = validate_files(_extras(converted, lambda E: E.iloc[1:]))
    _find(r, "extra_features.csv", "lead_id", "have no row here")


def test_numeric_extras_must_be_numbers(converted: dict[str, bytes]) -> None:
    numeric = _declare(converted, [{"name": "landing_page", "kind": "numeric", "source": "mql.landing_page_id"}])
    r = validate_files(_extras(numeric, lambda E: E.assign(landing_page=["12", "", *["3.5"] * (len(E) - 2)])))
    assert r.ok, r.errors  # numbers or blank
    r = validate_files(numeric)  # the landing page ids are hex strings
    _find(r, "extra_features.csv", "landing_page", "are not numbers")


def test_extras_and_their_declaration_come_together(converted: dict[str, bytes]) -> None:
    r = validate_files({k: v for k, v in converted.items() if k != "extra_features.csv"})
    _find(r, "extra_features.csv", None, "required file is missing: dataset.json declares the extra feature(s)")
    meta = {k: v for k, v in json.loads(converted["dataset.json"]).items() if k != "features"}
    r = validate_files({**converted, "dataset.json": json.dumps(meta).encode()})
    _find(r, "extra_features.csv", None, "provided without a dataset.json that declares its features")
    r = validate_files({k: v for k, v in converted.items() if k != "dataset.json"})
    _find(r, "extra_features.csv", None, "provided without a dataset.json")
    r = validate_files(_declare(converted, [{"name": "Bad Name", "kind": "categorical"}]))
    _find(r, "dataset.json", "features", "must be lower-case")
