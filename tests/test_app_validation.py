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
