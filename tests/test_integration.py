"""The emva/ pipeline on data/v1: legacy labels must equal the frozen baseline; horizon labels are the new default."""
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
from emva.labels import LEGACY
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
    result = run(data_v1, context=ctx, labels=LEGACY)
    write_outputs(result, e_out)
    printed = "\n".join(result.messages + [result.summary.to_string(index=False)]) + "\n"
    assert printed == proc.stdout
    assert filecmp.cmp(b_out / "scores.csv", e_out / "scores.csv", shallow=False)
    assert filecmp.cmp(b_out / "weights.csv", e_out / "weights.csv", shallow=False)


def test_cli_legacy_mode_reproduces_baseline_byte_for_byte(tmp_path, data_v1):
    b_out, e_out = tmp_path / "b", tmp_path / "e"
    b_out.mkdir(), e_out.mkdir()
    base = subprocess.run([sys.executable, "emva_score.py", "--data", str(data_v1), "--out", str(b_out)],
                          cwd=REPO / "baseline", capture_output=True, text=True, check=True)
    proc = subprocess.run([sys.executable, "-m", "emva", "--data", str(data_v1), "--out", str(e_out),
                           "--label-mode", "legacy"], cwd=REPO, capture_output=True, text=True, check=True)
    assert proc.stdout == base.stdout
    assert "formula 0.814 0.1006       0.571          0.795" in proc.stdout
    for name in ("scores.csv", "weights.csv"):
        assert filecmp.cmp(b_out / name, e_out / name, shallow=False), name


def test_cli_default_is_horizon_and_writes_label_columns(tmp_path, data_v1):
    subprocess.run([sys.executable, "-m", "emva", "--data", str(data_v1), "--out", str(tmp_path)],
                   cwd=REPO, capture_output=True, text=True, check=True)
    scores = pd.read_csv(tmp_path / "scores.csv", index_col="lead_id")
    assert list(scores.columns[-4:]) == ["y", "won_within_h", "label_source", "matured_at"]
    assert set(scores.label_source) == {"won", "crm_lost", "stalled", "ghosted", "open"}
    assert (tmp_path / "weights.csv").exists()


@pytest.mark.parametrize("flags", [["--label-mode", "legacy", "--include-ghosted"],
                                   ["--label-mode", "legacy", "--stalled-as-lost"],
                                   ["--label-mode", "legacy", "--horizon-days", "90"],
                                   ["--horizon-days", "0"]])
def test_cli_rejects_horizon_options_without_horizon_mode(tmp_path, data_v1, flags):
    proc = subprocess.run([sys.executable, "-m", "emva", "--data", str(data_v1), "--out", str(tmp_path), *flags],
                          cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 2 and "error:" in proc.stderr


def test_make_baseline_passes():
    proc = subprocess.run(["make", "baseline"], cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS: baseline and emva/" in proc.stdout
    assert "emva scores.csv vs baseline run: byte-identical" in proc.stdout


def test_test_set_is_frozen_definition(v1_result):
    X = v1_result.X
    assert (X.created_at[v1_result.test] >= TEST_FROM).all() and X.y[v1_result.test].notna().all()


@pytest.fixture(scope="module")
def report_text(data_v1) -> str:
    from emva.eval.report import build_report
    return build_report(data_v1, n_resamples=20)


def _section(text: str, title: str) -> str:
    """The markdown between ``## title`` and the next ``## `` heading."""
    start = text.index(f"\n## {title}\n")
    end = text.find("\n## ", start + 1)
    return text[start:end if end != -1 else None]


def test_report_builds_with_status_quo_targets(report_text):
    # Section (a) is the Phase 0 report on the frozen legacy test set.
    text = _section(report_text, "(a) Legacy labels, legacy test set")
    # Only sort-independent numbers here: the table's status-quo top-20% cells use the unstable
    # sort (ties at the cut), so check the tie-averaged footnote values instead.
    assert "| status quo | 0.639 [" in text
    assert "ranking wins; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.397" in text
    assert "ranking revenue by p×value; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.454" in text
    # Phase 0 also asserted "| candidate | 0.814 [": no longer true by design, because the candidate is now
    # trained on horizon labels (Phase 1); the legacy-mode pipeline's 0.814 is asserted in
    # test_pipeline_reproduces_baseline_metrics instead.
    assert "| baseline | 0.814 [" in text and "| candidate | " in text
    assert "2453 leads labelled by the baseline rules" in text


def test_report_contains_both_label_definitions(report_text, v1_horizon):
    assert report_text.index("## (a) Legacy labels, legacy test set") < report_text.index("## (b) Horizon labels, mature test set")
    b = _section(report_text, "(b) Horizon labels, mature test set")
    n = int(v1_horizon.test.sum())
    assert f"{n} mature leads created on or after 2026-05-01 and up to 2026-05-26T20:06:58+00:00" in b
    for model in ("baseline", "candidate", "status quo"):
        assert f"| {model} | 0." in b
    for heading in ("### Headline", "### Paired AUC comparison vs baseline", "### Calibration by decile of p",
                    "### AUC by test month", "### Value scale"):
        assert heading in b and heading in _section(report_text, "(a) Legacy labels, legacy test set")
    labels = _section(report_text, "Label definitions")
    for src in ("won", "crm_lost", "stalled", "ghosted", "open", "all"):
        assert f"| {src} | " in labels
    assert "| legacy | 5588 (847) | 2453 (352) |" in labels
    assert "Candidate trained with: `horizon H=120`." in labels
    ghost = _section(report_text, "Bottom-decile ghosted share")
    assert "| all leads labelled by legacy rules (train + test) | 8041 | 16.5% | 27.5% |" in ghost


def test_horizon_pipeline_uses_mature_leads_only(v1_horizon):
    X, tr, te = v1_horizon.X, v1_horizon.train, v1_horizon.test
    mature_cut = pd.Timestamp("2026-05-27T00:00:00Z")
    assert (X.created_at[tr | te] <= mature_cut).all()
    assert (X.created_at[te] >= TEST_FROM).all() and (X.created_at[tr] < TEST_FROM).all()
    assert X.y[tr | te].notna().all()
    assert not X.label_source[tr | te].isin(["ghosted", "stalled"]).any()
    assert int(tr.sum()) == 4049 and int(te.sum()) == 458 and int(X.y[te].sum()) == 80
    assert v1_horizon.labels.describe() == "horizon H=120"
