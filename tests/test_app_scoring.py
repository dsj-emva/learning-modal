"""app.scoring and app.auth: the form adapter scores through emva.scoring; the password gate."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import scoring
from app.auth import check_password, expected_password
from emva.io import read_leads
from emva.persist import load_bundle
from emva.scoring import UnknownLevelError, lead_from_form, points_breakdown, score_leads

# ADR 0018: a lead scored alone differs from the batch run by at most this many ulp (BLAS kernel per batch shape).
MAX_ULP = 64


def _ulp_close(a: float, b: float) -> bool:
    """``a`` within ``MAX_ULP`` units in the last place of ``b``."""
    return abs(a - b) <= MAX_ULP * np.spacing(abs(b))


@pytest.fixture(scope="module")
def trained(app_trained):
    """``(bundle, dataset dir, scores.csv)`` of the shared trained run."""
    _, run = app_trained
    return load_bundle(Path(run.out_dir) / "model.joblib"), Path(run.dataset_path), \
        pd.read_csv(Path(run.out_dir) / "scores.csv", index_col="lead_id")


def test_form_lead_matches_score_leads(trained) -> None:
    bundle, data, _ = trained
    values = scoring.defaults_for(data)
    got = scoring.score_form(bundle, values, data)
    ref = score_leads(bundle, lead_from_form(scoring.clean_values(values)), data).iloc[0]
    assert _ulp_close(got.p, ref.p_formula) and _ulp_close(got.value_at_submit, ref.value_at_submit)
    assert not got.is_bot and not got.is_duplicate
    logit = np.log(got.p / (1 - got.p))
    assert got.points.log_odds.sum() == pytest.approx(logit, abs=1e-9)
    assert got.points.label.iloc[0] == "Starting score"


def test_historical_lead_matches_batch_scores(trained) -> None:
    bundle, data, scores = trained
    raw = read_leads(data / "historical_leads.csv")
    for lead_id in scores.index[:25]:
        got = scoring.score_form(bundle, scoring.values_from_lead(raw.loc[lead_id]), data)
        want = scores.loc[lead_id]
        assert _ulp_close(got.p, want.p_formula), lead_id
        assert _ulp_close(got.deal_value, want.deal_value_hat), lead_id
        assert _ulp_close(got.value_at_submit, want.value_at_submit), lead_id


def test_every_scoring_field_is_on_the_form() -> None:
    from emva.scoring import submit_time_fields
    on_form = [f.name for fields in scoring.form_sections().values() for f in fields]
    assert sorted(on_form) == sorted(f.name for f in submit_time_fields())
    assert set(scoring.DEFAULTS) == set(on_form)


def test_defaults_fill_every_session_field(trained) -> None:
    assert scoring.blank_session_fields(scoring.DEFAULTS) == []
    assert scoring.blank_session_fields({**scoring.DEFAULTS, "hesitation_ms": None}) == ["hesitation_ms"]
    bundle, data, _ = trained
    blank = scoring.score_form(bundle, {**scoring.defaults_for(data), "hesitation_ms": None}, data)
    assert "session_missing" in set(blank.points.feature)


def test_unknown_level_and_bad_input(trained) -> None:
    bundle, data, _ = trained
    with pytest.raises(ValueError, match="allowed"):
        scoring.score_form(bundle, {**scoring.defaults_for(data), "form_variant": "Z"}, data)
    with pytest.raises(ValueError, match="required"):
        scoring.score_form(bundle, {**scoring.defaults_for(data), "email": "  "}, data)
    assert issubclass(UnknownLevelError, ValueError)


def test_bot_flag_is_reported_not_dropped(trained) -> None:
    bundle, data, _ = trained
    got = scoring.score_form(bundle, {**scoring.defaults_for(data), "time_on_page_s": 3.0}, data)
    assert got.is_bot and 0 < got.p < 1


def test_describe_transform(trained) -> None:
    text = scoring.describe_transform(trained[0])
    assert text.startswith("p × expected deal value, capped at £") and "floored at £25" in text


def test_check_password() -> None:
    assert check_password("s3cret-é", "s3cret-é")
    assert not check_password("wrong", "s3cret")
    assert not check_password("", "s3cret") and not check_password(None, "s3cret")
    assert not check_password("", "") and not check_password("x", None)


def test_unset_password_refuses() -> None:
    assert expected_password({}) is None
    assert expected_password({"APP_PASSWORD": "   "}) is None
    assert expected_password({"APP_PASSWORD": "pw"}) == "pw"
