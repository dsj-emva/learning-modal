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


def test_generic_is_offered_only_with_extras(app_converted) -> None:
    """Phase 10 (b): the training panel offers generic for a dataset with extra_features.csv, and its summary lists
    the extras as the pipeline reads them."""
    root, run = app_converted
    ds = storage.get_dataset(root, run.dataset)
    assert ds.has_extras and training.feature_set_options(ds) == ["v2", "generic", "legacy"]
    assert training.feature_set_options(storage.sample_dataset()) == ["v2", "legacy"]
    s = pre_training_summary(run.dataset_path, TrainingConfig(feature_set="generic"))
    v2 = pre_training_summary(run.dataset_path, TrainingConfig())
    assert s.extras == (("landing_page", "categorical"),) and v2.extras == ()
    assert (s.train, s.test, s.mature_test) == (v2.train, v2.test, v2.mature_test)  # same rows and split
    with pytest.raises(ValueError, match="needs a converted dataset whose dataset.json declares 'features'"):
        pre_training_summary(DATA_V1, TrainingConfig(feature_set="generic"))


def test_generic_comparison_reuses_the_reports_numbers(app_generic) -> None:
    """Phase 10 (b): Keel's Generic vs v2 is emva.eval.generic_report.compare on the selected test set."""
    from emva.eval.bootstrap import SEED
    from emva.eval.collinearity import check_collinearity
    from emva.eval.generic_report import compare
    from emva.features import FeatureSet
    from emva.labels import HORIZON

    root, run = app_generic
    g = results.generic_comparison(run.out_dir, run.dataset_path, "horizon", "mature", n_resamples=200)
    assert g is not None and g.test_set == "mature" and g.v2_columns == 39 and g.x_columns > 0
    as_of, test_from = dataset_dates(run.dataset_path)
    y = frozen_test_labels(load(run.dataset_path), as_of, test_from)[3]
    gen = pipeline_run(run.dataset_path, labels=HORIZON, features=FeatureSet.GENERIC)
    v2 = pipeline_run(run.dataset_path, labels=HORIZON, features=FeatureSet.V2)
    col = check_collinearity(gen.design[gen.train])
    ref = compare(gen, v2, [("mature", y)], "mature", col, 200, SEED)
    want = ref.paired["mature"]
    assert (g.paired.auc_a, g.paired.auc_b, g.paired.p_value) == (want.auc_a, want.auc_b, want.p_value)
    assert (g.paired.diff.point, g.paired.diff.lo, g.paired.diff.hi) == (want.diff.point, want.diff.lo, want.diff.hi)
    assert g.n == len(y) and g.collinearity == col
    pd.testing.assert_frame_equal(g.screen, ref.screen)
    assert list(g.screen.extra) == ["landing_page"] and g.flagged == []
    legacy = results.generic_comparison(run.out_dir, run.dataset_path, "horizon", "legacy", n_resamples=200)
    assert legacy is not None and legacy.test_set == "legacy"


def test_generic_comparison_refuses_scores_it_cannot_reproduce(app_generic, tmp_path: Path) -> None:
    import shutil

    _, run = app_generic
    shutil.copy(Path(run.out_dir) / "scores.csv", tmp_path / "scores.csv")
    s = pd.read_csv(tmp_path / "scores.csv")
    s.assign(p_formula=s.p_formula * 0.5).to_csv(tmp_path / "scores.csv", index=False)
    with pytest.raises(ValueError, match="does not reproduce this run's scores"):
        results.generic_comparison(tmp_path, run.dataset_path, "horizon", "mature", n_resamples=50)
    with pytest.raises(ValueError, match="needs a converted dataset"):
        results.generic_comparison(tmp_path, DATA_V1, "horizon", "mature", n_resamples=50)


def test_generic_scorecard_names_the_extras(app_generic) -> None:
    from emva.persist import load_bundle

    _, run = app_generic
    bundle = load_bundle(Path(run.out_dir) / "model.joblib")
    card = results.scorecard(run.out_dir, bundle.extras)
    x = card[card.column.str.startswith("x_landing_page=")]
    assert len(x) == len(bundle.extras.columns()) > 0
    assert set(x.feature) == {"Extra · landing_page"} and set(x.reference) == {bundle.extras.extras[0].reference}
    assert set(results.scorecard(run.out_dir).loc[x.index, "reference"]) == {""}  # without the bundle: unknown
