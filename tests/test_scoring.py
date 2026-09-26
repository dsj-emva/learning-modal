"""emva.scoring: scoring new leads with a saved bundle must reproduce the batch pipeline's scores exactly."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from emva.constants import ANSWER_KEYS, FREE, MISSING, V2_LEVELS
from emva.feature_spec import feature_spec
from emva.constants import CHANNEL_SOURCES
from emva.features import SESSION_INPUTS, FeatureSet, session_absent
from emva.io import read_leads
from emva.model import INTERCEPT
from emva.persist import ModelBundle, load_bundle, save_bundle
from emva.pipeline import PipelineResult
from emva.scoring import (
    FieldSpec,
    UnknownLevelError,
    lead_from_form,
    points_breakdown,
    prepare,
    score_leads,
    submit_time_fields,
)

from conftest import DATA_V1, DATA_V2

CASES = [(d, fs) for d in (DATA_V1, DATA_V2) for fs in FeatureSet]
SCORE_COLUMNS = ["p_formula", "deal_value_hat", "value_at_submit"]
# Scores of the same lead from batches of different sizes differ in the last bits: sklearn's predict_proba and
# Ridge.predict are BLAS matrix products (Accelerate here) whose kernel, and so summation order, depends on the
# number of rows. Measured over 3,000 leads in batches of 10 and 600 single leads per (data, feature set): at most
# 14 ulp on p_formula, 16 on deal_value_hat and value_formula, 62 on value_at_submit (the log compression). Scoring
# all cleaned leads in one batch (the training matrix's shape) is byte-equal (test_full_batch_is_byte_equal).
MAX_ULP = 64


@pytest.fixture(scope="module")
def trained(trained_run, tmp_path_factory) -> dict[tuple[Path, FeatureSet], tuple[PipelineResult, ModelBundle]]:
    """Each (data, feature set): the batch run and its bundle after a save/load round trip."""
    out = {}
    for data, fs in CASES:
        result = trained_run(data, fs)
        path = tmp_path_factory.mktemp("bundle") / "model.joblib"
        save_bundle(result, path, data, margin=1.0)
        out[data, fs] = result, load_bundle(path)
    return out


def _raw(data: Path, ids: list[str]) -> pd.DataFrame:
    """The ``historical_leads.csv`` rows of ``ids``, with ``lead_id`` as a column (what the app passes)."""
    return read_leads(data / "historical_leads.csv").loc[ids].reset_index()


def _pick(result: PipelineResult, data: Path) -> list[str]:
    """~10 cleaned leads covering the awkward cases, then a few random ones."""
    X = result.X
    raw = read_leads(data / "historical_leads.csv").loc[X.index]
    groups = {
        "no enrichment": X.en_employee_band.isna(),
        "name-matched enrichment": X.enrichment_source.eq("name"),
        "no session": session_absent(raw),
        "lead ads": X.channel.eq("meta_leadads"),
        "free email": X.email_l.str.split("@").str[1].isin(FREE),
        "boilerplate text": X.text.eq("copy_paste"),
        "no company size, no enrichment": X.en_employee_band.isna() & X.a_company_size.isna(),
    }
    ids: list[str] = []
    for name, mask in groups.items():
        assert mask.any(), name
        ids.append(next(i for i in X.index[mask] if i not in ids))
    rest = X.index.difference(ids).to_series().sample(4, random_state=0).tolist()
    return ids + rest


@pytest.mark.parametrize("data,fs", CASES, ids=[f"{d.name}-{fs.value}" for d, fs in CASES])
def test_model_inputs_are_byte_equal_to_the_batch_run(trained, data, fs):
    result, bundle = trained[data, fs]
    ids = _pick(result, data)
    X, D, M = prepare(bundle, _raw(data, ids), data)
    assert D.index.tolist() == ids and list(D.columns) == list(result.design.columns)
    assert np.array_equal(D.to_numpy(), result.design.loc[ids].to_numpy())
    batch_M = feature_spec(fs).value_design(result.X).loc[ids]
    assert np.array_equal(M.to_numpy(), batch_M.reindex(columns=M.columns).to_numpy())
    for c in bundle.levels:
        assert X[c].astype(str).tolist() == result.X.loc[ids, c].astype(str).tolist(), c


@pytest.mark.parametrize("data,fs", CASES, ids=[f"{d.name}-{fs.value}" for d, fs in CASES])
def test_full_batch_is_byte_equal(trained, data, fs):
    # every cleaned lead in one call: same matrix shape as the batch run, so bit-for-bit the same scores
    result, bundle = trained[data, fs]
    got = score_leads(bundle, _raw(data, result.X.index.tolist()), data)
    want = result.scores()
    assert got.index.equals(want.index)
    for c in SCORE_COLUMNS + ["value_formula"]:
        assert np.array_equal(got[c].to_numpy(), want[c].to_numpy()), c
    assert not got.is_bot.any() and not got.is_duplicate.any()


@pytest.mark.parametrize("data,fs", CASES, ids=[f"{d.name}-{fs.value}" for d, fs in CASES])
def test_scores_equal_the_batch_run_to_the_last_bits(trained, data, fs):
    # not np.array_equal: see MAX_ULP (BLAS summation order depends on the batch size)
    result, bundle = trained[data, fs]
    ids = _pick(result, data)
    got = score_leads(bundle, _raw(data, ids), data)
    want = result.scores().loc[ids]
    assert got.index.tolist() == ids and got.index.name == "lead_id"
    for c in SCORE_COLUMNS + ["value_formula"]:
        np.testing.assert_array_max_ulp(got[c].to_numpy(), want[c].to_numpy(), maxulp=MAX_ULP)
    assert not got.is_bot.any() and not got.is_duplicate.any()


@pytest.mark.parametrize("data,fs", CASES, ids=[f"{d.name}-{fs.value}" for d, fs in CASES])
def test_scores_are_byte_equal_to_the_batch_computation_on_the_same_rows(trained, data, fs):
    # the same arithmetic on the same matrix is bit-for-bit reproducible: score_leads == the pipeline's own
    # functions applied to the batch run's design rows for these leads
    from emva.model import predict
    from emva.value import expected_value, predict_deal_value
    result, bundle = trained[data, fs]
    ids = _pick(result, data)
    got = score_leads(bundle, _raw(data, ids), data)
    p = predict(result.model, result.design.loc[ids])
    dv = predict_deal_value(result.deal_value, feature_spec(fs).value_design(result.X).loc[ids])
    value = expected_value(pd.Series(p), pd.Series(dv), 1.0)
    assert np.array_equal(got.p_formula.to_numpy(), p) and np.array_equal(got.deal_value_hat.to_numpy(), dv)
    assert np.array_equal(got.value_formula.to_numpy(), value.to_numpy())
    assert np.array_equal(got.value_at_submit.to_numpy(), result.value_transform.apply(value))


def test_legacy_labels_have_no_value_at_submit(tmp_path, v1_result):
    save_bundle(v1_result, tmp_path / "m.joblib", DATA_V1, margin=1.0)
    bundle = load_bundle(tmp_path / "m.joblib")
    ids = v1_result.X.index[:5].tolist()
    got = score_leads(bundle, _raw(DATA_V1, ids), DATA_V1)
    assert got.value_at_submit.isna().all()
    np.testing.assert_array_max_ulp(got.p_formula.to_numpy(), v1_result.X.p_formula.loc[ids].to_numpy(), MAX_ULP)


def test_form_fields_are_all_the_model_reads(trained):
    # keep only the submit-time fields (+ lead_id, created_at, answers): the scores must not move
    result, bundle = trained[DATA_V2, FeatureSet.V2]
    ids = _pick(result, DATA_V2)
    raw = _raw(DATA_V2, ids)
    keep = ["lead_id", "created_at", "answers"] + [f.name for f in submit_time_fields() if f.source == "column"]
    full, slim = score_leads(bundle, raw, DATA_V2), score_leads(bundle, raw[keep], DATA_V2)
    for c in SCORE_COLUMNS:
        assert np.array_equal(full[c].to_numpy(), slim[c].to_numpy()), c


def test_absent_session_columns_mean_no_session(trained):
    result, bundle = trained[DATA_V2, FeatureSet.V2]
    lead = result.X.index[~result.X.session_missing.eq("yes")][0]
    raw = _raw(DATA_V2, [lead])
    blank = raw.copy()
    blank[list(SESSION_INPUTS)] = np.nan
    dropped = score_leads(bundle, raw.drop(columns=list(SESSION_INPUTS)), DATA_V2)
    assert np.array_equal(dropped.p_formula.to_numpy(), score_leads(bundle, blank, DATA_V2).p_formula.to_numpy())
    assert dropped.p_formula.iloc[0] != score_leads(bundle, raw, DATA_V2).p_formula.iloc[0]


def test_bots_and_duplicates_are_flagged_within_the_batch_not_dropped(trained):
    result, bundle = trained[DATA_V1, FeatureSet.V2]
    raw = _raw(DATA_V1, result.X.index[:3].tolist())
    raw.loc[1, "email"] = raw.loc[0, "email"].upper()
    raw.loc[1, "created_at"] = raw.loc[0, "created_at"] + pd.Timedelta(hours=1)
    raw.loc[2, "time_on_page_s"] = 3.0
    got = score_leads(bundle, raw, DATA_V1)
    assert got.is_duplicate.tolist() == [False, True, False] and got.is_bot.tolist() == [False, False, True]
    assert got.p_formula.notna().all()


@pytest.mark.parametrize("fs", list(FeatureSet))
def test_unknown_level_raises_listing_the_allowed_levels(trained, fs):
    result, bundle = trained[DATA_V1, fs]
    raw = _raw(DATA_V1, result.X.index[:2].tolist())
    raw.loc[1, "form_variant"] = "Z"
    with pytest.raises(UnknownLevelError, match=r"form_variant.*'Z'.*allowed: \['A', 'B', 'C', 'D'\]") as e:
        score_leads(bundle, raw, DATA_V1)
    assert raw.lead_id[1] in str(e.value) and raw.lead_id[0] not in str(e.value)


def test_unknown_company_size_answer_raises(trained):
    result, bundle = trained[DATA_V2, FeatureSet.V2]
    lead = result.X.index[result.X.en_employee_band.isna()][0]
    raw = _raw(DATA_V2, [lead])
    raw.loc[0, "answers"] = json.dumps({"company_size": "5000+"})
    with pytest.raises(UnknownLevelError, match="band"):
        score_leads(bundle, raw, DATA_V2)


def test_input_errors(trained):
    result, bundle = trained[DATA_V1, FeatureSet.V2]
    raw = _raw(DATA_V1, result.X.index[:1].tolist())
    with pytest.raises(ValueError, match="lead_id"):
        score_leads(bundle, raw.drop(columns="lead_id"), DATA_V1)
    with pytest.raises(ValueError, match="unknown columns.*utm_sauce"):
        score_leads(bundle, raw.rename(columns={"utm_source": "utm_sauce"}), DATA_V1)
    with pytest.raises(ValueError, match="email"):
        score_leads(bundle, raw.drop(columns="email"), DATA_V1)


@pytest.mark.parametrize("data,fs", CASES, ids=[f"{d.name}-{fs.value}" for d, fs in CASES])
def test_points_breakdown_sums_to_the_logit(trained, data, fs):
    result, bundle = trained[data, fs]
    ids = _pick(result, data)
    raw = _raw(data, ids)
    pb = points_breakdown(bundle, raw, data)
    p = score_leads(bundle, raw, data).p_formula
    assert list(pb.columns) == ["lead_id", "feature", "level", "log_odds", "odds_multiplier", "points"]
    total = pb.groupby("lead_id").log_odds.sum()
    assert np.allclose(total.loc[ids].to_numpy(), np.log(p / (1 - p)).loc[ids].to_numpy(), rtol=0, atol=1e-9)
    first = pb[pb.lead_id == ids[0]]
    assert first.feature.iloc[0] == INTERCEPT and first.log_odds.iloc[0] == bundle.model.intercept_[0]
    for _, row in first.iloc[1:].iterrows():
        assert f"{row.feature}={row.level}" in bundle.design_columns
    assert np.allclose(pb.points, pb.log_odds * 20 / np.log(2)) and np.allclose(pb.odds_multiplier, np.exp(pb.log_odds))


def _form_fields(raw_row: pd.Series) -> dict[str, object]:
    """The form-field dict an app would collect for this historical lead."""
    answers = json.loads(raw_row.answers)
    out: dict[str, object] = {}
    for f in submit_time_fields():
        v = answers.get(f.name) if f.source == "answers" else raw_row[f.name]
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            out[f.name] = v.item() if hasattr(v, "item") else v
    return out


@pytest.mark.parametrize("data", [DATA_V1, DATA_V2], ids=["v1", "v2"])
def test_lead_from_form_round_trips_through_score_leads(trained, data):
    # byte-equal to scoring the historical row alone (same one-row matrix, same arithmetic)
    result, bundle = trained[data, FeatureSet.V2]
    ids = _pick(result, data)
    raw = _raw(data, ids)
    for i, lead in enumerate(ids):
        form = lead_from_form(_form_fields(raw.iloc[i]))
        assert len(form) == 1 and form.lead_id[0] != lead and form.created_at[0].tz is not None
        got, want = score_leads(bundle, form, data), score_leads(bundle, raw.iloc[[i]], data)
        for c in SCORE_COLUMNS:
            assert np.array_equal(got[c].to_numpy(), want[c].to_numpy()), (lead, c)


def test_lead_from_form_validates():
    ok = {"email": "a@b.example", "form_variant": "A"}
    assert lead_from_form(ok).email[0] == "a@b.example"
    with pytest.raises(ValueError, match="unknown form field.*colour"):
        lead_from_form({**ok, "colour": "red"})
    with pytest.raises(ValueError, match="email"):
        lead_from_form({"form_variant": "A"})
    with pytest.raises(ValueError, match="form_variant.*'Z'"):
        lead_from_form({**ok, "form_variant": "Z"})
    with pytest.raises(ValueError, match="viewed_pricing.*bool"):
        lead_from_form({**ok, "viewed_pricing": "yes"})
    with pytest.raises(ValueError, match="time_on_page_s.*number"):
        lead_from_form({**ok, "time_on_page_s": "12"})
    form = lead_from_form({**ok, "what_to_solve": "£5k budget", "company_size": "", "utm_source": ""})
    assert json.loads(form.answers[0]) == {"what_to_solve": "£5k budget", "company_size": ""}
    assert "utm_source" not in form
    assert "time_on_page_s" not in lead_from_form({**ok, "time_on_page_s": float("nan")})


def test_submit_time_fields(trained):
    fields = submit_time_fields()
    assert all(isinstance(f, FieldSpec) for f in fields)
    names = [f.name for f in fields]
    assert len(names) == len(set(names))
    assert [f.name for f in fields if f.source == "answers"] == list(ANSWER_KEYS)
    schema = trained[DATA_V1, FeatureSet.V2][1].lead_schema
    columns = {f.name for f in fields if f.source == "column"}
    assert columns <= set(schema.columns) and set(SESSION_INPUTS) <= columns
    assert {f.name for f in fields if f.required} == {"email"}
    spec = {f.name: f for f in fields}
    assert spec["company_size"].allowed == tuple(b for b in V2_LEVELS["band"] if b != MISSING)
    assert spec["form_variant"].allowed == V2_LEVELS["form_variant"]
    assert set(spec["utm_source"].allowed) == {s for sources in CHANNEL_SOURCES.values() for s in sources}
    assert spec["viewed_pricing"].dtype == "bool" and spec["time_on_page_s"].dtype == "float"


@pytest.mark.parametrize("value,blank", [(None, True), (float("nan"), True), (np.float64("nan"), True),
                                         (pd.NA, True), (pd.NaT, True), ("", True), (" ", False), ("x", False),
                                         (0, False), (0.0, False), (False, False), (np.int64(3), False)])
def test_is_blank(value: object, blank: bool) -> None:
    from emva.scoring import is_blank
    assert is_blank(value) is blank


def test_field_types_check_and_coerce_in_one_place() -> None:
    from emva.scoring import FieldType

    assert FieldType.INT.coerce(3.0) == 3 and isinstance(FieldType.INT.coerce(np.float64(2.0)), int)
    assert FieldType.FLOAT.coerce(" ") is None and FieldType.FLOAT.coerce(np.nan) is None
    assert FieldType.BOOL.coerce(np.bool_(True)) is True and FieldType.STR.coerce(5) == "5"
    assert FieldType.STR.coerce(" as typed ") == " as typed "
    assert FieldType.INT.accepts(4.0) and not FieldType.INT.accepts(4.5) and not FieldType.FLOAT.accepts(True)
    spec = {f.name: f for f in submit_time_fields()}
    assert spec["email"].required and not spec["company"].required
    with pytest.raises(ValueError, match="must be an integer"):
        spec["local_submit_hour"].check(3.5)
