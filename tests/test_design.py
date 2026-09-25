import pandas as pd

from emva.constants import CATS
from emva.design import design
from conftest import REPO


def test_reference_levels_dropped_and_column_set(v1_result):
    D = v1_result.design
    for feat, ref in CATS.items():
        assert f"{feat}={ref}" not in D.columns
    assert all(c.split("=", 1)[0] in CATS for c in D.columns)
    assert len(D.columns) == 47  # 47 rows in baseline/weights.csv
    assert set(D.columns) == set(pd.read_csv(REPO / "baseline" / "weights.csv", index_col=0).index)
    assert D.dtypes.eq(float).all()


def test_design_on_toy_frame():
    row = {c: ref for c, ref in CATS.items()}
    other = dict(row, channel="meta", band="51-200")
    D = design(pd.DataFrame([row, other], index=["a", "b"]))
    assert list(D.columns) == ["channel=meta", "band=51-200"]
    assert D.loc["a"].tolist() == [0.0, 0.0] and D.loc["b"].tolist() == [1.0, 1.0]


def test_extra_columns_are_joined():
    row = {c: ref for c, ref in CATS.items()}
    X = pd.DataFrame([row, dict(row, text="vague")], index=["a", "b"])
    D = design(X, extra=pd.Series([0.5, None], index=["a", "b"], name="context_logit"))
    assert list(D.columns) == ["text=vague", "context_logit"]
