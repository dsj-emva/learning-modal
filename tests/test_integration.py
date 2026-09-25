"""The emva/ pipeline on data/v1 must equal the frozen baseline (Phase 0: no behaviour change)."""
import filecmp
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from emva.constants import TEST_FROM
from emva.eval.metrics import tie_averaged_top_share
from emva.eval.regression import EXPECTED_SUMMARY, FROZEN_WEIGHTS, read_weights, weights_mismatches
from emva.eval.status_quo import load_rules, status_quo_value
from emva.io import load
from emva.pipeline import run, write_outputs
from sklearn.metrics import roc_auc_score

from conftest import REPO


def test_pipeline_reproduces_baseline_metrics(v1_result):
    assert v1_result.summary.iloc[0].drop("model").to_dict() == EXPECTED_SUMMARY
    assert int(v1_result.test.sum()) == 2453 and int(v1_result.train.sum()) == 5588


def test_pipeline_reproduces_frozen_weights(v1_result):
    assert weights_mismatches(v1_result.weights, read_weights(FROZEN_WEIGHTS)) == []


def test_status_quo_on_frozen_test_set(v1_result, data_v1):
    X, te = v1_result.X, v1_result.test
    sq = status_quo_value(load(data_v1).loc[X.index[te]], load_rules(data_v1 / "status_quo_rules.json")).sq_value
    y = X.y[te]
    revenue = pd.Series(np.where(y == 1, X.deal_value[te].fillna(0), 0), index=y.index)
    assert round(roc_auc_score(y, sq), 3) == 0.639
    # The status quo has 203 test leads tied at the top-20% cut, so the unstable-sort figure the
    # report shows (0.392 wins) depends on numpy's CPU-specific sort kernels. Assert the
    # tie-averaged value instead: it is the expectation over all tie orders, so it is portable.
    assert round(tie_averaged_top_share(y, sq.values), 3) == 0.397
    assert round(tie_averaged_top_share(revenue, sq.values), 3) == 0.454


def test_context_mode_matches_baseline_script(tmp_path, data_v1):
    L = pd.read_csv(data_v1 / "historical_leads.csv").sample(3000, random_state=2)
    ctx = tmp_path / "ctx.csv"
    pd.DataFrame({"lead_id": L.lead_id, "context_score": np.random.default_rng(1).uniform(0, 1, len(L)).round(3)}
                 ).to_csv(ctx, index=False)
    b_out, e_out = tmp_path / "b", tmp_path / "e"
    b_out.mkdir(), e_out.mkdir()
    proc = subprocess.run([sys.executable, "emva_score.py", "--data", str(data_v1), "--out", str(b_out),
                           "--context", str(ctx)], cwd=REPO / "baseline", capture_output=True, text=True, check=True)
    result = run(data_v1, context=ctx)
    write_outputs(result, e_out)
    printed = "\n".join(result.messages + [result.summary.to_string(index=False)]) + "\n"
    assert printed == proc.stdout
    assert filecmp.cmp(b_out / "scores.csv", e_out / "scores.csv", shallow=False)
    assert filecmp.cmp(b_out / "weights.csv", e_out / "weights.csv", shallow=False)


@pytest.mark.parametrize("module", ["emva"])
def test_cli_entry_point_runs(tmp_path, module, data_v1):
    proc = subprocess.run([sys.executable, "-m", module, "--data", str(data_v1), "--out", str(tmp_path)],
                          cwd=REPO, capture_output=True, text=True, check=True)
    assert "formula 0.814 0.1006       0.571          0.795" in proc.stdout
    assert (tmp_path / "weights.csv").exists() and (tmp_path / "scores.csv").exists()


def test_test_set_is_frozen_definition(v1_result):
    X = v1_result.X
    assert (X.created_at[v1_result.test] >= TEST_FROM).all() and X.y[v1_result.test].notna().all()


def test_report_builds_with_status_quo_targets(data_v1):
    from emva.eval.report import build_report
    text = build_report(data_v1, n_resamples=20)
    # Only sort-independent numbers here: the table's status-quo top-20% cells use the unstable
    # sort (ties at the cut), so check the tie-averaged footnote values instead.
    assert "| status quo | 0.639 [" in text
    assert "ranking wins; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.397" in text
    assert "ranking revenue by p×value; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.454" in text
    assert "| baseline | 0.814 [" in text and "| candidate | 0.814 [" in text
