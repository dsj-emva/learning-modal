"""emva.eval.value_report: value-scale comparison (plan 3.2) and click-ID coverage (plan 3.5) on v1 and v2."""
import numpy as np
import pandas as pd
import pytest

from emva.eval.value_report import REPORT_TRANSFORMS, click_id_coverage, scale_stats, value_report_sections
from emva.pipeline import run

from conftest import REPO


def test_scale_stats():
    v = np.arange(1.0, 201.0)
    s = scale_stats(v)
    assert s["max"] == 200 and s["p50"] == pytest.approx(100.5)
    assert s["max_over_median"] == pytest.approx(200 / 100.5)
    assert s["top1pct_share"] == pytest.approx((200 + 199) / v.sum())


def test_report_transforms_cover_plan_3_2():
    names = [t.describe() for t in REPORT_TRANSFORMS]
    assert names == ["identity", "cap p97", "cap p97 + floor £25", "cap p97 + sqrt", "cap p97 + log", "tiers 5",
                     "tiers 10", "cap p97 + log + floor £25"]


def _revenue(result, ids: pd.Index) -> pd.Series:
    X = result.X.loc[ids]
    return pd.Series(np.where(X.y == 1, X.deal_value.fillna(0), 0), index=ids)


@pytest.mark.parametrize("data", ["v1", "v2"])
def test_value_report_runs_and_the_default_passes(data, v1_horizon):
    result = v1_horizon if data == "v1" else run(REPO / "data" / "v2")
    ids = result.X.index[result.test]
    lines = value_report_sections(result, [("test", _revenue(result, ids))], result.X.value_formula)
    text = "\n".join(lines)
    assert "## Value transforms (plan 3.2)" in text and "## Click-ID coverage (plan 3.5)" in text
    assert "| cap p97 + log + floor £25 | PASS |" in text
    assert "| identity | fail |" in text
    # candidate identity against itself as the "baseline": 100% retention
    assert "| candidate: identity |" in text and "| 100.0% |" in text


def test_click_id_coverage_on_v1(v1_horizon):
    cov = click_id_coverage(v1_horizon.X).set_index("channel")
    assert cov.loc["meta_leadads", "lead-form id"] == "100.0%"
    landing = cov.loc["landing-page paid (all but meta_leadads)"]
    assert landing["leads"] == 7454 and landing["no id at all"] == "28.8%"
    assert cov.loc["all paid", "leads"] == 8386
