"""Plan 2.6 evaluation: the coefficient-bootstrap decision matches what the v2 feature set hard-codes."""
import pytest

from emva.eval import feature_selection
from emva.features import V2_DROPPED


def test_decisions_match_the_hard_coded_drop(data_v1):
    table, keep = feature_selection.selection_table(data_v1, n_refits=feature_selection.MIN_REFITS, seed=0)
    assert set(keep) == set(feature_selection.SELECTION_CANDIDATES)
    assert {f for f, k in keep.items() if not k} == set(V2_DROPPED)
    assert set(table.feature) == set(feature_selection.SELECTION_CANDIDATES)
    # absent sessions are carried by session_missing, so ip_type has only its dc level
    assert table.loc[table.feature == "ip_type", "level"].tolist() == ["ip_type=dc"]


def test_needs_enough_refits(data_v1):
    with pytest.raises(ValueError, match="at least 200"):
        feature_selection.selection_table(data_v1, n_refits=50)
