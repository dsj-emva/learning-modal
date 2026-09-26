"""A converted dataset in the app (Phase 9): its own as_of / test_from in the pre-training summary, the results page
and the base rate (ADR 0020); the job runs the standard report on it without rules (ADR 0022); the bundled sample's
behaviour is unchanged. The dataset is the hand-built Olist-shaped fixture (conftest.app_converted)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app import results, storage, training
from app.training import TrainingConfig, pre_training_summary
from emva.constants import AS_OF, TEST_FROM
from emva.dataset_meta import dataset_dates
from emva.eval.report import frozen_test_labels
from emva.io import load
from emva.pipeline import run as pipeline_run

from conftest import DATA_V1


def test_converted_dataset_carries_its_own_dates(app_converted) -> None:
    _, run = app_converted
    as_of, test_from = dataset_dates(run.dataset_path)
    assert (as_of.date().isoformat(), test_from) == ("2018-11-15", "2018-03-01")
    assert storage.is_converted(run.dataset_path) and (Path(run.dataset_path) / storage.MAPPING_FILE).is_file()


@pytest.mark.parametrize("mode", ["horizon", "legacy"])
def test_pre_training_summary_uses_the_dataset_dates(app_converted, mode: str) -> None:
    _, run = app_converted
    s = pre_training_summary(run.dataset_path, TrainingConfig(label_mode=mode))
    r = pipeline_run(run.dataset_path, labels=TrainingConfig(label_mode=mode).labels)  # the CLI's own split
    assert (s.train, s.test) == (int(r.train.sum()), int(r.test.sum()))
    assert s.test > 0 and s.train > 0 and s.test_wins > 0  # with the 2026 constants every lead would be training
    as_of, test_from = dataset_dates(run.dataset_path)
    assert s.mature_test == len(frozen_test_labels(load(run.dataset_path), as_of, test_from)[3]) > 0
    assert len(frozen_test_labels(load(run.dataset_path))[3]) == 0  # the constants' test window is empty here
    assert s.label_counts["stalled"]["leads"] == 0  # read at 2018-11-15, not at 2026-09-24


def test_job_runs_the_report_on_a_converted_dataset(app_converted) -> None:
    _, run = app_converted
    assert run.status == "succeeded" and run.has_report and run.metrics.get("auc") is not None
    report = run.report_path.read_text(encoding="utf-8")
    assert "Baseline omitted" in report and "Status quo omitted" in report and "from dataset.json" in report
    log = Path(run.log_path).read_text(encoding="utf-8")
    assert "[report] exit code 0" in log and "skipped" not in log


def test_report_available(app_converted, app_trained, tmp_path: Path) -> None:
    _, conv = app_converted
    _, sample = app_trained
    assert training.report_available(conv.dataset_path) and not training.rules_available(conv.dataset_path)
    assert training.report_available(sample.dataset_path) and training.report_available(DATA_V1)
    plain = tmp_path / "plain"
    plain.mkdir()
    for name in storage.REQUIRED_FILES:  # the v1 format without rules: no report, as before Phase 9
        (plain / name).write_bytes((Path(sample.dataset_path) / name).read_bytes())
    assert not training.report_available(plain)


def test_job_still_skips_the_report_for_the_v1_format_without_rules(app_trained, tmp_path: Path) -> None:
    import sys

    _, sample = app_trained
    root = storage.init_root(tmp_path / "root")
    files = {n: (Path(sample.dataset_path) / n).read_bytes() for n in storage.REQUIRED_FILES}
    ds = storage.save_dataset(root, "no-rules", files)
    run = storage.register_run(root, storage.new_run(root, ds, TrainingConfig().as_dict()))
    assert training.start_training(root, run, python=sys.executable, report_resamples=50).wait(timeout=180) == 0
    run = storage.get_run(root, run.run_id)
    assert run.status == "succeeded" and not run.has_report
    assert "[report] skipped" in Path(run.log_path).read_text(encoding="utf-8")


def test_evaluate_uses_the_dataset_dates(app_converted) -> None:
    _, run = app_converted
    L = load(run.dataset_path)
    as_of, test_from = dataset_dates(run.dataset_path)
    ev = results.evaluate(run.out_dir, run.dataset_path, "mature", leads=L)
    assert ev is not None and ev.baseline_missing == results.BASELINE_CONVERTED
    y = frozen_test_labels(L, as_of, test_from)[3]
    assert list(ev.test.y.index) == list(y.index) and (ev.test.leads.created_at >= test_from).all()
    assert [m.name for m in ev.models] == [results.CANDIDATE]
    s = results.load_scores(run.out_dir)
    assert ev.base_rate == results.training_base_rate(s, L.created_at, test_from)
    assert ev.base_rate != results.training_base_rate(s, L.created_at)  # the 2026 boundary puts every lead in train
    assert results.evaluate(run.out_dir, run.dataset_path, "legacy", leads=L) is not None


def test_sample_keeps_the_constant_dates(app_trained) -> None:
    _, run = app_trained
    assert dataset_dates(run.dataset_path) == (AS_OF, TEST_FROM)
    L = load(run.dataset_path)
    ev = results.evaluate(run.out_dir, run.dataset_path, "mature", leads=L)
    assert list(ev.test.y.index) == list(frozen_test_labels(L)[3].index)
    assert ev.base_rate == results.training_base_rate(results.load_scores(run.out_dir), L.created_at)
    assert isinstance(ev.base_rate, float) and not pd.isna(ev.base_rate)
