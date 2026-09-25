"""Design collinearity check (plan 2.7)."""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from emva.eval.collinearity import CollinearityError, assert_no_collinearity, check_collinearity, summary_row

from conftest import REPO


def _random_design(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({c: rng.integers(0, 2, n).astype(float) for c in "abcd"})


def test_passes_on_independent_columns():
    res = assert_no_collinearity(_random_design())
    assert res.passed and res.pairs == [] and abs(res.max_pair[2]) < 0.2 and res.describe().startswith("PASS")


def test_fails_on_a_duplicated_column():
    D = _random_design()
    D["b_copy"] = D.b
    res = check_collinearity(D)
    assert not res.passed and res.pairs == [("b", "b_copy", pytest.approx(1.0))]
    assert res.describe().startswith("FAIL (1 pairs")
    with pytest.raises(CollinearityError, match="b ~ b_copy"):
        assert_no_collinearity(D)


def test_fails_on_a_complementary_column_and_respects_the_threshold():
    D = _random_design()
    D["not_a"] = 1 - D.a  # corr -1
    assert check_collinearity(D).pairs[0][:2] == ("a", "not_a")
    D2 = _random_design()
    D2["mostly_c"] = D2.c
    D2.loc[D2.index[:40], "mostly_c"] = 1 - D2.c[:40]  # corr about 0.84
    assert check_collinearity(D2).passed and not check_collinearity(D2, threshold=0.8).passed


def test_summary_row_handles_a_design_without_pairs():
    res = check_collinearity(pd.DataFrame({"a": [0.0, 1.0, 0.0]}))
    assert res.max_pair is None
    assert summary_row(res) == {"max |corr|": "n/a", "check": "pass", "pairs above threshold": "none"}
    assert summary_row(check_collinearity(_random_design()))["check"] == "pass"


def test_constant_columns_are_skipped():
    D = _random_design()
    D["zero"] = 0.0
    D["zero2"] = 0.0
    res = check_collinearity(D)
    assert res.passed and res.constant == ["zero", "zero2"]


def test_both_feature_sets_fail_on_v1_each_on_one_pair(v1_result, v1_horizon, v1_v2_legacy_labels):
    legacy = check_collinearity(v1_result.design[v1_result.train])
    assert {p[:2] for p in legacy.pairs} == {("email=free", "no_company=yes")}  # feature-design flaw
    for r in (v1_horizon, v1_v2_legacy_labels):
        res = check_collinearity(r.design[r.train])
        # a property of v1 (every session-less lead is a lead-ads lead); clears on data/v2
        assert not res.passed
        assert [p[:2] for p in res.pairs] == [("channel=meta_leadads", "session_missing=yes")]


@pytest.mark.parametrize("feature_set,pair", [("v2", "channel=meta_leadads ~ session_missing=yes"),
                                              ("legacy", "email=free ~ no_company=yes")])
def test_cli_exits_nonzero_and_lists_the_pair_on_v1(feature_set, pair):
    proc = subprocess.run([sys.executable, "-m", "emva.eval.collinearity", "--data", str(REPO / "data" / "v1"),
                           "--feature-set", feature_set], cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "FAIL (1 pairs" in proc.stdout and f"  - {pair}: +1.000" in proc.stdout


@pytest.mark.parametrize("labels", ["horizon", "legacy"])
def test_v2_features_pass_strict_on_data_v2(labels):
    """ADR 0011: the tie and collinearity criteria are evaluated on data/v2, where consent-declined leads separate
    session_missing from the lead-ads channel."""
    proc = subprocess.run([sys.executable, "-m", "emva.eval.collinearity", "--data", str(REPO / "data" / "v2"),
                           "--label-mode", labels], cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "- PASS: max |corr| = 0.8" in proc.stdout
