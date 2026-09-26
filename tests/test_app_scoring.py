"""app.scoring: the form adapter scores through emva.scoring, matching the batch run (R17, ADR 0018)."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import results, scoring
from emva.io import read_leads
from emva.persist import load_bundle
from emva.scoring import UnknownLevelError, lead_from_form, points_breakdown, score_leads

# ADR 0018 / R17: a lead scored alone differs from the batch run by at most this many ulp (BLAS kernel per batch
# shape). Scores are read with float_precision="round_trip" (results.load_scores); the default parser alone added
# ~110 ulp on top. A form lead vs score_leads on the same row is exact (same path, same shape).
MAX_ULP = 64


def _ulp_close(a: float, b: float) -> bool:
    """``a`` within ``MAX_ULP`` units in the last place of ``b``."""
    return abs(a - b) <= MAX_ULP * np.spacing(abs(b))


@pytest.fixture(scope="module")
def trained(app_trained):
    """``(bundle, dataset dir, scores.csv)`` of the shared trained run."""
    _, run = app_trained
    return load_bundle(Path(run.out_dir) / "model.joblib"), Path(run.dataset_path), \
        results.load_scores(run.out_dir)


def test_form_lead_matches_score_leads(trained) -> None:
    bundle, data, _ = trained
    values = scoring.defaults_for(data)
    got = scoring.score_form(bundle, values, data)
    ref = score_leads(bundle, lead_from_form(scoring.clean_values(values)), data).iloc[0]
    assert got.p == ref.p_formula and got.value_at_submit == ref.value_at_submit  # same code path, same batch shape
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


# --- source format (Phase 9): leads as the source sends them, converted with the dataset's mapping -----------------

@pytest.fixture(scope="module")
def converted(app_converted):
    """``(bundle, dataset dir, mapping)`` of the run trained on the converted Olist-shaped fixture."""
    _, run = app_converted
    return load_bundle(Path(run.out_dir) / "model.joblib"), Path(run.dataset_path), \
        scoring.run_mapping(run.dataset_path)


def _mql_csv() -> bytes:
    from conftest import INGEST_FIXTURES, OLIST_MQL

    return (INGEST_FIXTURES / "olist_funnel" / OLIST_MQL).read_bytes()


def test_run_mapping(converted, app_trained) -> None:
    _, _, mapping = converted
    assert mapping.name == "olist_funnel" and mapping.outcome_confirmed
    assert scoring.run_mapping(app_trained[1].dataset_path) is None  # the v1 format has no mapping


def test_source_fields_are_the_submit_time_columns(converted) -> None:
    _, _, mapping = converted
    fields = scoring.source_fields(mapping)
    assert [f.column for f in fields] == ["mql_id", "first_contact_date", "origin", "landing_page_id"]
    by = {f.column: f for f in fields}
    assert by["origin"].options[:2] == ("paid_search", "social") and by["origin"].feeds == "utm_source, utm_medium"
    assert by["landing_page_id"].options is None and by["mql_id"].feeds.startswith("lead lead_id")
    assert "won_date" not in by and "declared_monthly_revenue" not in by  # outcome and deal value: never asked


def test_source_form_lead_scores_like_convert_leads_then_score_leads(converted) -> None:
    from emva.ingest.convert import convert_leads

    bundle, data, mapping = converted
    keys = {f.column: f.key for f in scoring.source_fields(mapping)}
    values = {keys["mql_id"]: "", keys["first_contact_date"]: "2018-04-01", keys["origin"]: "paid_search",
              keys["landing_page_id"]: "lp-1"}
    frames = scoring.frames_from_source_form(mapping, values)
    assert "mql_id" not in next(iter(frames.values())).columns  # a blank id: convert_leads numbers the lead
    res = scoring.score_source(bundle, mapping, frames, data)
    t = res.table()
    assert list(t.lead_id) == ["new-000001"] and 0 < t.p.iloc[0] < 1
    direct = score_leads(bundle, convert_leads(frames, mapping), data)
    assert t.p.iloc[0] == direct.p_formula.iloc[0] and t.value_at_submit.iloc[0] == direct.value_at_submit.iloc[0]
    one = scoring.source_lead_score(bundle, res, "new-000001", data)
    logit = np.log(one.p / (1 - one.p))
    assert abs(one.points.log_odds.sum() - logit) < 1e-9 and "session_missing" in set(one.points.feature)
    assert "time_on_page_s" in one.blank_session  # the source has no on-site session


def test_source_upload_scores_every_row_and_ignores_outcome_columns(converted) -> None:
    bundle, data, mapping = converted
    frames = scoring.frames_from_source_uploads(mapping, {"new_mqls.csv": _mql_csv()})  # any name when alone
    res = scoring.score_source(bundle, mapping, frames, data)
    raw = pd.read_csv(io.BytesIO(_mql_csv()), dtype=str)
    assert list(res.table().lead_id) == list(raw.mql_id) and res.table().p.between(0, 1).all()
    with_outcome = raw.assign(won_date="2018-05-01", deal_stage="Won")  # extra columns are not read
    frames = scoring.frames_from_source_uploads(mapping, {"x.csv": with_outcome.to_csv(index=False).encode()})
    assert scoring.score_source(bundle, mapping, frames, data).table().p.tolist() == res.table().p.tolist()


def test_source_refusals_are_clear(converted) -> None:
    bundle, data, mapping = converted
    keys = {f.column: f.key for f in scoring.source_fields(mapping)}
    frames = scoring.frames_from_source_form(mapping, {keys["origin"]: "carrier_pigeon"})
    with pytest.raises(ValueError, match="not in its value_map"):
        scoring.score_source(bundle, mapping, frames, data)
    raw = pd.read_csv(io.BytesIO(_mql_csv()), dtype=str)
    dup = pd.concat([raw.head(2), raw.head(1)]).to_csv(index=False).encode()
    with pytest.raises(ValueError, match="unique"):
        scoring.score_source(bundle, mapping, scoring.frames_from_source_uploads(mapping, {"x.csv": dup}), data)
    with pytest.raises(ValueError, match="missing source file"):
        scoring.frames_from_source_uploads(mapping, {"a.csv": b"x\n1\n", "b.csv": b"y\n2\n"})


def test_joined_file_gets_the_primary_join_value() -> None:
    from emva.ingest.mapping import load_mapping

    from conftest import REPO

    m = load_mapping((REPO / "mappings" / "crm_opportunities.toml").read_text(encoding="utf-8"))
    fields = scoring.source_fields(m)
    assert [(f.file, f.column) for f in fields] == [("sales_pipeline.csv", "opportunity_id"),
                                                    ("sales_pipeline.csv", "engage_date"),
                                                    ("sales_pipeline.csv", "account"),
                                                    ("accounts.csv", "office_location")]
    values = {f.key: v for f, v in zip(fields, ["", "2017-03-01", "Acme", "United States"])}
    frames = scoring.frames_from_source_form(m, values)
    assert frames["accounts.csv"].account.iloc[0] == "Acme" == frames["sales_pipeline.csv"].account.iloc[0]
