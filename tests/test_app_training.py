"""app.training / app.job: the pre-training summary uses the CLI's code; a training job runs end to end."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app import storage, training
from app.training import TrainingConfig, parse_summary, pre_training_summary
from emva.constants import TEST_FROM
from emva.eval.report import frozen_test_labels
from emva.features import FeatureSet
from emva.io import load
from emva.labels import HORIZON, LEGACY, split_masks
from emva.pipeline import build

from conftest import DATA_V1


@pytest.mark.parametrize("labels,mode", [(HORIZON, "horizon"), (LEGACY, "legacy")])
def test_pre_training_summary_matches_pipeline(labels, mode) -> None:
    s = pre_training_summary(DATA_V1, TrainingConfig(label_mode=mode))
    L = load(DATA_V1)
    X = build(L, labels, FeatureSet.V2)
    tr, te = split_masks(X, TEST_FROM, labels.eligible(X))
    assert (s["leads"], s["scored"], s["train"], s["test"]) == (len(L), len(X), int(tr.sum()), int(te.sum()))
    assert s["train_wins"] == int((X.y[tr] == 1).sum())
    assert s["mature_test"] == len(frozen_test_labels(L)[3]) == 458  # CLAUDE.md: the frozen horizon test set
    assert sum(v["leads"] for v in s["label_counts"].values()) == len(X)
    if mode == "horizon":
        assert s["train"] == 4049 and s["label_counts"]["stalled"]["unlabelled"] == 618  # standard report figures


def test_config_rejects_unknown_options() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(label_mode="sometimes")
    with pytest.raises(ValueError):
        TrainingConfig(feature_set="v9")


def test_commands_quote_nothing_and_pass_options(tmp_path: Path) -> None:
    cfg = TrainingConfig(label_mode="legacy", feature_set="legacy")
    cmd = training.training_command("/a b/data", tmp_path / "out dir", cfg, python="py")
    assert cmd == ["py", "-m", "emva", "--data", "/a b/data", "--out", str(tmp_path / "out dir"),
                   "--label-mode", "legacy", "--feature-set", "legacy"]
    assert training.report_command("d", cfg, python="py")[:4] == ["py", "-m", "emva.eval.report", "--data"]


def test_parse_summary() -> None:
    log = "$ python -m emva\n  model   auc  brier  top20_wins  top20_revenue\nformula 0.778 0.1265        0.45          0.765\n"
    assert parse_summary(log) == {"auc": 0.778, "brier": 0.1265, "top20_wins": 0.45, "top20_revenue": 0.765}
    assert parse_summary("no labelled test leads") == {}


def test_training_job_end_to_end(app_trained) -> None:
    root, run = app_trained
    assert run.status == "succeeded" and run.error is None and run.pid is None and run.finished_at
    out = Path(run.out_dir)
    for f in ("scores.csv", "weights.csv", "model.joblib", "report.txt", "train.log"):
        assert (out / f).stat().st_size > 0, f
    assert run.has_report and "# EMVA standard report" in run.report_path.read_text()
    assert set(run.metrics) == {"auc", "brier", "top20_wins", "top20_revenue"} and 0.5 < run.metrics["auc"] < 1
    log = Path(run.log_path).read_text()
    assert "$ python -m emva --data" in log and "[done] succeeded" in log
    assert [r.run_id for r in storage.list_runs(root)] == [run.run_id]


def test_failed_training_is_recorded(tmp_path: Path) -> None:
    import sys

    root = storage.init_root(tmp_path)
    bad = tmp_path / "bad-data"
    bad.mkdir()
    ds = storage.Dataset(name="bad", path=str(bad), created_at=storage.utc_now(), rows={})
    run = storage.register_run(root, storage.new_run(root, ds, TrainingConfig().as_dict()))
    assert training.start_training(root, run, python=sys.executable).wait(timeout=60) != 0
    done = storage.get_run(root, run.run_id)
    assert done.status == "failed" and "exit" in done.error and not done.has_report


def test_reconcile_marks_vanished_job_failed(tmp_path: Path) -> None:
    root = storage.init_root(tmp_path)
    run = storage.register_run(root, storage.new_run(root, storage.sample_dataset(), TrainingConfig().as_dict()))
    run = storage.update_run(root, run.run_id, status="running", pid=2 ** 22 + 12345)  # no such process
    assert training.reconcile(root, run).status == "failed"


def test_results_frames_agree_with_the_run(app_trained) -> None:
    """app.results on the trained run: the mature test set is the run's own test set (horizon defaults), so the
    headline equals the summary python -m emva printed."""
    from app import results

    _, run = app_trained
    ev = results.evaluate(run.out_dir, run.dataset_path, "mature")
    head = results.headline(ev, n_resamples=100)
    model = head.set_index("model").loc[results.CANDIDATE]
    assert round(model.auc, 3) == run.metrics["auc"] and round(model.brier, 4) == run.metrics["brier"]
    assert round(model.top20_wins, 3) == run.metrics["top20_wins"]
    assert list(head.model) == [results.CANDIDATE, results.STATUS_QUO] and np.isnan(head.brier.iloc[1])
    assert 0 < ev.base_rate < 1
    cal = results.calibration(ev)
    assert list(cal.decile) == list(range(1, 11)) and cal.n.sum() == len(ev.test.y)
    month = results.auc_month(ev, n_resamples=100)
    assert month.n.sum() == len(ev.test.y) and {"auc_lo", "auc_hi"} <= set(month.columns)
    card = results.scorecard(run.out_dir)
    assert card.points.is_monotonic_decreasing and "Starting score" not in set(card.feature)
    vd = results.value_distribution(run.out_dir).set_index("column")
    assert list(vd.index) == ["value_at_submit", "value_formula"] and (vd.max_over_median >= 1).all()
    assert results.evaluate(run.out_dir, run.dataset_path, "legacy") is not None
