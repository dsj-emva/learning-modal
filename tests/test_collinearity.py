"""Design collinearity check (plan 2.7) and exact-alias pruning."""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from emva.design import drop_aliased_columns
from emva.eval.collinearity import CollinearityError, assert_no_collinearity, check_collinearity

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


def test_constant_columns_are_skipped():
    D = _random_design()
    D["zero"] = 0.0
    D["zero2"] = 0.0
    res = check_collinearity(D)
    assert res.passed and res.constant == ["zero", "zero2"]


def test_drop_aliased_columns_keeps_the_first_of_identical_columns():
    D = pd.DataFrame({"a": [1.0, 0, 1, 0], "b": [0.0, 1, 0, 1], "a2": [1.0, 0, 1, 0], "b2": [0.0, 1, 0, 0]})
    rows = pd.Series([True, True, True, False])
    P, aliases = drop_aliased_columns(D, rows)
    assert aliases == {"a2": "a", "b2": "b"}  # b2 equals b on the chosen rows only
    assert list(P.columns) == ["a", "b"] and len(P) == 4
    assert drop_aliased_columns(D, pd.Series(True, index=D.index))[1] == {"a2": "a"}


def test_legacy_design_fails_and_v2_passes_on_v1(v1_result, v1_horizon, v1_v2_legacy_labels):
    legacy = check_collinearity(v1_result.design[v1_result.train])
    assert not legacy.passed
    assert {p[:2] for p in legacy.pairs} == {("email=free", "no_company=yes")}
    for r in (v1_horizon, v1_v2_legacy_labels):
        res = check_collinearity(r.design[r.train])
        assert res.passed and abs(res.max_pair[2]) < 0.95


@pytest.mark.parametrize("feature_set,code", [("v2", 0), ("legacy", 1)])
def test_cli_exit_code(feature_set, code):
    proc = subprocess.run([sys.executable, "-m", "emva.eval.collinearity", "--data", str(REPO / "data" / "v1"),
                           "--feature-set", feature_set], cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == code, proc.stdout + proc.stderr
    assert ("FAIL" in proc.stdout) == bool(code)
