"""The results page without a baseline row (ADR 0022): a converted dataset (``dataset.json``) or a baseline script
that cannot score the dataset leaves the model and status-quo rows, with a note instead of the paired comparison.
Every combination of baseline yes/no x rules yes/no, on copies of the shared trained run's dataset."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app import results, storage

from conftest import APP_REPORT_RESAMPLES

MAIN = str(Path(__file__).resolve().parents[1] / "app" / "main.py")
# The v1 dates, so the frozen test sets are the ones the shared run was scored on.
V1_DATES = b'{"as_of": "2026-09-24", "test_from": "2026-05-01"}\n'


def _dataset_files(run: storage.Run, rules: bool, converted: bool) -> dict[str, bytes]:
    """The shared run's training files (with or without the rules), plus a v1-dated ``dataset.json`` if converted."""
    names = [*storage.REQUIRED_FILES, *([storage.RULES_FILE] if rules else [])]
    files = {n: (Path(run.dataset_path) / n).read_bytes() for n in names}
    return {**files, storage.DATASET_META_FILE: V1_DATES} if converted else files


def _copy(run: storage.Run, dest: Path, rules: bool, converted: bool) -> Path:
    """``dest`` holding ``_dataset_files``; returns it."""
    dest.mkdir()
    for name, content in _dataset_files(run, rules, converted).items():
        (dest / name).write_bytes(content)
    return dest


def _no_baseline_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if the frozen baseline script is started."""
    def refuse(*args: object) -> None:
        raise AssertionError("the baseline script must not run on a converted dataset")
    monkeypatch.setattr(results, "run_baseline", refuse)


@pytest.mark.parametrize("rules", [True, False])
def test_converted_dataset_has_no_baseline_row(app_trained, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                               rules: bool) -> None:
    _, run = app_trained
    data = _copy(run, tmp_path / "converted", rules=rules, converted=True)
    full, _ = results.standard_table(results.evaluate(run.out_dir, run.dataset_path, "mature"),
                                     n_resamples=APP_REPORT_RESAMPLES)
    _no_baseline_run(monkeypatch)
    ev = results.evaluate(run.out_dir, data, "mature")
    assert ev.baseline_missing == results.BASELINE_CONVERTED and ev.model(results.BASELINE) is None
    head, paired = results.standard_table(ev, n_resamples=APP_REPORT_RESAMPLES)
    expected = [results.CANDIDATE, *([results.STATUS_QUO] if rules else [])]
    assert list(head.model) == expected and paired is None
    assert results.kpi_reference(head.model) == (results.STATUS_QUO if rules else None)
    assert any(storage.RULES_FILE in n for n in ev.notes) == (not rules)
    # the model's row is the same as with the baseline present: only the baseline row goes
    pd.testing.assert_series_equal(head.set_index("model").loc[results.CANDIDATE],
                                   full.set_index("model").loc[results.CANDIDATE])


@pytest.mark.parametrize("rules", [True, False])
def test_failing_baseline_script_drops_only_the_baseline(app_trained, tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch,
                                                         caplog: pytest.LogCaptureFixture, rules: bool) -> None:
    _, run = app_trained
    data = _copy(run, tmp_path / "plain", rules=rules, converted=False)

    def fail(*args: object) -> None:
        raise RuntimeError("baseline failed (1):\nTraceback ...")
    monkeypatch.setattr(results, "run_baseline", fail)
    with caplog.at_level(logging.WARNING, logger="app.results"):
        ev = results.evaluate(run.out_dir, data, "legacy")
    assert ev.baseline_missing == results.BASELINE_FAILED and "baseline script failed" in caplog.text
    head, paired = results.standard_table(ev, n_resamples=APP_REPORT_RESAMPLES)
    assert list(head.model) == [results.CANDIDATE, *([results.STATUS_QUO] if rules else [])] and paired is None
    assert results.kpi_reference(head.model) == (results.STATUS_QUO if rules else None)


@pytest.mark.parametrize("error", [OSError("no scores.csv"), ValueError("unparsable scores.csv")])
def test_unreadable_baseline_output_is_a_failure(app_trained, monkeypatch: pytest.MonkeyPatch,
                                                 error: Exception) -> None:
    _, run = app_trained

    def fail(*args: object) -> None:
        raise error
    monkeypatch.setattr(results, "run_baseline", fail)
    ids = pd.Index(["L1"])
    assert results.baseline_scores(run.dataset_path, ids) == (None, results.BASELINE_FAILED)


def test_baseline_that_misses_test_leads_is_a_failure(app_trained, monkeypatch: pytest.MonkeyPatch,
                                                      caplog: pytest.LogCaptureFixture) -> None:
    _, run = app_trained
    scored = pd.DataFrame({"p_formula": [0.2, None], "value_formula": [10.0, None]}, index=["a", "b"])
    monkeypatch.setattr(results, "run_baseline", lambda data, out: (scored, ""))
    assert results.baseline_scores(run.dataset_path, pd.Index(["a"]))[0] is scored
    assert results.baseline_scores(run.dataset_path, pd.Index(["a", "b"])) == (None, results.BASELINE_FAILED)
    assert "left 1 test leads" in caplog.text
    no_p = pd.DataFrame({"value_formula": [10.0]}, index=["a"])
    monkeypatch.setattr(results, "run_baseline", lambda data, out: (no_p, ""))
    assert results.baseline_scores(run.dataset_path, pd.Index(["a"])) == (None, results.BASELINE_FAILED)


def test_the_real_baseline_still_runs_on_an_unconverted_copy(app_trained, tmp_path: Path) -> None:
    _, run = app_trained
    data = _copy(run, tmp_path / "plain", rules=False, converted=False)
    ev = results.evaluate(run.out_dir, data, "mature")
    assert ev.baseline_missing is None and ev.model(results.BASELINE) is not None
    head, paired = results.standard_table(ev, n_resamples=APP_REPORT_RESAMPLES)
    assert list(head.model) == [results.BASELINE, results.CANDIDATE] and list(paired.model) == [results.CANDIDATE]
    assert results.kpi_reference(head.model) == results.BASELINE


@pytest.mark.parametrize("models,expected", [
    ([results.BASELINE, results.CANDIDATE, results.STATUS_QUO], results.STATUS_QUO),
    ([results.CANDIDATE, results.STATUS_QUO], results.STATUS_QUO),
    ([results.BASELINE, results.CANDIDATE], results.BASELINE),
    ([results.CANDIDATE], None),
])
def test_kpi_reference(models: list[str], expected: str | None) -> None:
    assert results.kpi_reference(models) == expected


@pytest.mark.parametrize("rules", [True, False])
def test_results_page_renders_a_converted_dataset(app_trained, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  rules: bool) -> None:
    """The page on a run whose dataset is converted: no exception, the note shown, the paired section hidden."""
    _, trained = app_trained
    root = storage.init_root(tmp_path / "root")
    ds = storage.save_dataset(root, "converted", _dataset_files(trained, rules=rules, converted=True))
    run = storage.register_run(root, storage.new_run(root, ds, dict(trained.args)))
    shutil.copytree(trained.out_dir, run.out_dir, dirs_exist_ok=True)
    storage.update_run(root, run.run_id, status="succeeded", metrics=trained.metrics, finished_at=storage.utc_now())
    monkeypatch.setenv("DATA_DIR", str(root))
    monkeypatch.setenv("APP_PASSWORD", "letmein")
    at = AppTest.from_file(MAIN, default_timeout=120)
    at.run()
    at.text_input(key="password").input("letmein")
    at.button[0].click()
    at.run()
    assert not at.exception, at.exception
    page = "\n".join(m.value for m in at.markdown)
    assert "Standard table" in page and "No baseline row: this dataset was converted" in page
    assert "Compared with the frozen baseline" not in page
    assert ("vs status quo" in page) == rules and "vs baseline" not in page
    captions = "\n".join(c.value for c in at.caption)
    assert "on that source's data, not on simulated data" in captions and "bundled synthetic sample" not in captions
