"""Per-dataset dates (ADR 0020): ``emva.dataset_meta``, the pipeline and the report read ``dataset.json``."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from emva.constants import AS_OF, TEST_FROM
from emva.dataset_meta import (
    FORMAT_KEY,
    check_test_from,
    dataset_dates,
    has_dataset_meta,
    parse_as_of,
    read_dataset_meta,
)
from emva.labels import HORIZON
from emva.pipeline import run

TRAINING_CSVS = ("historical_leads.csv", "crm_history.csv", "companies.csv", "people.csv")


def _copy_v1(src: Path, dst: Path, meta: dict | None, rules: bool = False) -> Path:
    """The v1 training files in ``dst`` (never the ground truth), plus ``dataset.json`` = ``meta`` when given."""
    dst.mkdir()
    for f in TRAINING_CSVS + (("status_quo_rules.json",) if rules else ()):
        shutil.copy(src / f, dst / f)
    if meta is not None:
        (dst / "dataset.json").write_text(json.dumps({FORMAT_KEY: 1, **meta}), encoding="utf-8")
    return dst


def test_no_dataset_json_gives_the_constants(data_v1: Path) -> None:
    assert dataset_dates(data_v1) == (AS_OF, TEST_FROM)
    assert not has_dataset_meta(data_v1)


def test_dates_read_from_dataset_json(tmp_path: Path) -> None:
    (tmp_path / "dataset.json").write_text(json.dumps({FORMAT_KEY: 1, "as_of": "2018-12-01",
                                                       "test_from": "2018-03-01", "coverage": []}))
    as_of, test_from = dataset_dates(tmp_path)
    assert as_of == pd.Timestamp("2018-12-01", tz="UTC") and test_from == "2018-03-01"
    assert has_dataset_meta(tmp_path)


@pytest.mark.parametrize("meta, match", [
    ({FORMAT_KEY: 1, "as_of": "2018-12-01"}, "test_from"),
    (["2018-12-01", "2018-03-01"], "JSON object"),
    ({FORMAT_KEY: 2, "as_of": "2018-12-01", "test_from": "2018-03-01"}, "expected 1"),
    ({FORMAT_KEY: 1, "as_of": "2018-12-01", "test_from": "March 2018"}, "date string"),
    ({FORMAT_KEY: 1, "as_of": "", "test_from": "2018-03-01"}, "ISO"),
    ({FORMAT_KEY: 1, "as_of": "2018-01-01", "test_from": "2018-03-01"}, "after as_of"),
])
def test_malformed_dataset_json_is_refused(tmp_path: Path, meta: object, match: str) -> None:
    (tmp_path / "dataset.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match=match):
        dataset_dates(tmp_path)


def test_a_dataset_json_without_dates_is_refused(tmp_path: Path) -> None:
    """The app's pre-Phase 9 store record was also called dataset.json: refused loudly, never read as "no dates"."""
    (tmp_path / "dataset.json").write_text(json.dumps({"name": "x", "created_at": "2026-09-25", "rows": {}}))
    with pytest.raises(ValueError, match="dataset.json must be a JSON object with 'as_of' and 'test_from'"):
        dataset_dates(tmp_path)
    with pytest.raises(ValueError, match="reserved"):
        has_dataset_meta(tmp_path)


def test_read_dataset_meta_returns_the_whole_file(tmp_path: Path) -> None:
    assert read_dataset_meta(tmp_path) is None
    meta = {"as_of": "2018-12-01", "test_from": "2018-03-01", "coverage": [{"column": "channel=meta"}]}
    (tmp_path / "dataset.json").write_text(json.dumps(meta))
    assert read_dataset_meta(tmp_path) == meta


def test_parse_as_of_is_utc() -> None:
    assert parse_as_of("2018-12-01") == pd.Timestamp("2018-12-01", tz="UTC")
    assert parse_as_of("2018-12-01T02:00:00+02:00") == pd.Timestamp("2018-12-01T00:00:00", tz="UTC")
    assert parse_as_of("2018-12-01 09:30:00Z") == pd.Timestamp("2018-12-01T09:30:00", tz="UTC")
    with pytest.raises(ValueError):
        check_test_from("2026-02-30")
    for lenient in ("2018", "2018-12", "Dec 1 2018", "20181201", " 2018-12-01", "2018-02-30", "", None, 2018):
        with pytest.raises(ValueError):
            parse_as_of(lenient)


def test_pipeline_uses_the_dataset_dates(data_v1: Path, tmp_path: Path) -> None:
    """A dataset.json moves the split and the maturity cut; explicit arguments still win."""
    d = _copy_v1(data_v1, tmp_path / "v1", {"as_of": "2026-08-01", "test_from": "2026-03-01"})
    r = run(d, labels=HORIZON)
    assert (r.as_of, r.test_from) == (pd.Timestamp("2026-08-01", tz="UTC"), "2026-03-01")
    created = r.X.created_at
    assert created[r.test].min() >= pd.Timestamp("2026-03-01", tz="UTC") > created[r.train].max()
    # mature at the dataset's as_of, not at AS_OF: nothing created after 2026-08-01 - 120 days is eligible
    assert (created[r.train | r.test] + pd.Timedelta(days=120)).max() <= pd.Timestamp("2026-08-01", tz="UTC")
    explicit = run(d, labels=HORIZON, test_from=TEST_FROM, as_of=AS_OF)
    assert (explicit.as_of, explicit.test_from) == (AS_OF, TEST_FROM)


def test_report_omits_baseline_and_status_quo_on_a_converted_dataset(data_v1: Path, tmp_path: Path) -> None:
    """With dataset.json the frozen baseline is not run and, without rules, there is no status-quo row (ADR 0022)."""
    from emva.eval.report import build_report

    d = _copy_v1(data_v1, tmp_path / "conv", {"as_of": "2026-09-24", "test_from": "2026-05-01"})
    text = build_report(d, n_resamples=20)
    assert "Baseline omitted: this dataset carries its own dates in `dataset.json`" in text
    assert "Status quo omitted: the dataset has no `status_quo_rules.json`" in text
    assert "| baseline |" not in text and "| status quo |" not in text and "| candidate |" in text
    assert "Omitted: there is no baseline row" in text
    assert "Not computed: the criterion is defined against the baseline" in text
    assert "(from dataset.json)" in text


def test_report_keeps_the_baseline_without_dataset_json_but_drops_missing_rules(data_v1: Path, tmp_path: Path) -> None:
    from emva.eval.report import build_report

    d = _copy_v1(data_v1, tmp_path / "norules", None)
    text = build_report(d, n_resamples=20)
    assert "| baseline |" in text and "| status quo |" not in text
    assert "Status quo omitted" in text and "Baseline omitted" not in text


def test_baseline_failure_raises_unless_the_dataset_is_converted(data_v1: Path, tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """Only a converted dataset (dataset.json) omits the baseline row; elsewhere a failing script raises (ADR 0022)."""
    from emva.eval import report

    def fail(data: object, out: object) -> None:
        raise RuntimeError("baseline failed (1):\nboom")

    monkeypatch.setattr(report, "run_baseline", fail)
    with pytest.raises(RuntimeError, match="boom"):
        report.baseline_or_note(_copy_v1(data_v1, tmp_path / "plain", None))
    base, stdout, note = report.baseline_or_note(
        _copy_v1(data_v1, tmp_path / "conv", {"as_of": "2026-09-24", "test_from": "2026-05-01"}))
    assert base is None and stdout == "" and note.startswith("Baseline omitted: this dataset carries its own dates")
