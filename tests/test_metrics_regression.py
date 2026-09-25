import numpy as np
import pandas as pd
import pytest

from emva.eval.metrics import (
    auc_by_month,
    calibration_by_decile,
    tie_averaged_top_share,
    ties_at_cut,
    top_share,
    value_scale,
)
from emva.eval.regression import parse_summary, summary_mismatches, weights_mismatches


def test_top_share_and_ties():
    target = pd.Series([10.0, 0, 0, 0, 0, 0, 0, 0, 0, 5.0])
    score = np.arange(10.0)[::-1]  # row 0 ranked first
    assert top_share(target, score) == pytest.approx(10 / 15)
    assert ties_at_cut(score) == 1
    tied = np.array([1.0] * 10)
    assert ties_at_cut(tied) == 10
    assert tie_averaged_top_share(target, tied) == pytest.approx(0.2)


def test_calibration_by_decile():
    p = np.linspace(0.01, 0.99, 100)
    y = (p > 0.5).astype(float)
    cal = calibration_by_decile(pd.Series(y), p)
    assert cal.decile.tolist() == list(range(1, 11)) and cal.n.eq(10).all()
    assert cal.observed.iloc[0] == 0 and cal.observed.iloc[-1] == 1
    assert cal.mean_p.is_monotonic_increasing


def test_auc_by_month():
    created = pd.Series(pd.to_datetime(["2026-05-03", "2026-05-20", "2026-06-01", "2026-06-02"]))
    out = auc_by_month(pd.Series([0.0, 1.0, 1.0, 1.0]), np.array([0.1, 0.9, 0.5, 0.6]), created)
    assert out.month.tolist() == ["2026-05", "2026-06"]
    assert out.auc.iloc[0] == 1.0 and np.isnan(out.auc.iloc[1])


def test_value_scale():
    v = np.array([1.0] * 99 + [101.0])
    s = value_scale(v)
    assert s["max_over_median"] == 101 and s["top1pct_share"] == pytest.approx(101 / 200)


def test_parse_summary_and_mismatches():
    out = "  model   auc  brier  top20_wins  top20_revenue\nformula 0.814 0.1006       0.571          0.795\n"
    got = parse_summary(out)
    assert got == {"auc": 0.814, "brier": 0.1006, "top20_wins": 0.571, "top20_revenue": 0.795}
    assert summary_mismatches(got) == []
    assert summary_mismatches(dict(got, auc=0.813)) == ["auc: expected 0.814, got 0.813"]


def _w(rows):
    return pd.DataFrame(rows, columns=["name", "log_odds", "odds_multiplier", "points"]).set_index("name")


def test_weights_compare_ignores_row_order_but_not_values_or_rows():
    a = _w([("x", 0.16, 1.17, 5.0), ("y", 0.16, 1.17, 5.0), ("z", 1.0, 2.72, 29.0)])
    assert weights_mismatches(a.iloc[[1, 0, 2]], a) == []
    changed = a.copy()
    changed.loc["z", "log_odds"] = 1.001
    assert weights_mismatches(changed, a) == ["z log_odds: expected 1.0, got 1.001"]
    assert "missing rows: ['z']" in weights_mismatches(a.iloc[:2], a)


@pytest.mark.parametrize("fn", [lambda s: top_share(pd.Series(s), s), ties_at_cut,
                                lambda s: tie_averaged_top_share(pd.Series(s), s)])
def test_empty_top_cut_raises(fn):
    with pytest.raises(ValueError, match="is empty"):
        fn(np.array([0.3, 0.1, 0.2, 0.4]))  # int(4 * 0.2) == 0


def test_reordered_rows_rejects_length_mismatch(tmp_path):
    from emva.eval.regression import reordered_rows
    a = _w([("x", 0.1, 1.1, 3.0), ("y", 0.2, 1.2, 6.0)])
    a.to_csv(tmp_path / "a.csv")
    a.iloc[:1].to_csv(tmp_path / "b.csv")
    a.iloc[[1, 0]].to_csv(tmp_path / "c.csv")
    assert reordered_rows(tmp_path / "c.csv", tmp_path / "a.csv") == ["y", "x"]
    with pytest.raises(ValueError, match="2 and 1 rows"):
        reordered_rows(tmp_path / "a.csv", tmp_path / "b.csv")
