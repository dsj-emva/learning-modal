"""emva.persist: the model bundle written next to scores.csv and loaded by the scoring seam."""
from __future__ import annotations

import subprocess
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest

from emva.features import FeatureSet
from emva.io import read_leads
from emva.persist import FORMAT_VERSION, ModelBundle, library_versions, load_bundle, model_levels, save_bundle
from emva.scoring import score_leads

from conftest import DATA_V1, DATA_V2, REPO


@pytest.mark.parametrize("feature_set", [fs for fs in FeatureSet if fs is not FeatureSet.GENERIC])  # no extras here
def test_round_trip(tmp_path, trained_run, feature_set):
    result = trained_run(DATA_V2, feature_set)
    save_bundle(result, tmp_path / "model.joblib", DATA_V2, margin=0.8)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # same libraries: no version warning
        b = load_bundle(tmp_path / "model.joblib")
    assert isinstance(b, ModelBundle) and b.format_version == FORMAT_VERSION
    assert b.feature_set is feature_set and b.labels == result.labels and b.margin == 0.8
    assert b.design_columns == tuple(result.design.columns)
    assert np.array_equal(b.model.coef_, result.model.coef_) and b.model.intercept_ == result.model.intercept_
    assert np.array_equal(b.deal_value.ridge.coef_, result.deal_value.ridge.coef_)
    assert b.deal_value.log_residual_sd == result.deal_value.log_residual_sd
    assert b.value_transform == result.value_transform
    pd.testing.assert_frame_equal(b.weights, result.weights)
    assert b.data_name == "v2" and b.created_at.endswith("+00:00")
    assert set(b.versions) == {"numpy", "pandas", "scikit-learn"}
    # the raw lead schema (for coercing form input) and every model feature's allowed levels
    assert "created_at" in b.lead_schema and len(b.lead_schema) == 0 and b.lead_schema.index.name == "lead_id"
    assert set(b.levels) >= {"channel", "band", "text", "sector"}
    for col in b.design_columns:
        feature, level = col.split("=", 1)
        assert level in b.levels[feature]


def test_levels_include_the_reference_and_every_v2_level(tmp_path, trained_run):
    save_bundle(trained_run(DATA_V1, FeatureSet.V2), tmp_path / "m.joblib", DATA_V1, margin=1.0)
    b = load_bundle(tmp_path / "m.joblib")
    assert b.levels["channel"] == ("google", "meta", "meta_leadads", "linkedin", "chatgpt", "organic_direct")
    assert "missing" in b.levels["sector"] and "missing" in b.levels["band"]


def test_legacy_labels_have_no_value_transform(tmp_path, v1_result):
    save_bundle(v1_result, tmp_path / "m.joblib", DATA_V1, margin=1.0)
    assert load_bundle(tmp_path / "m.joblib").value_transform is None


def _rewrite(path, **changes) -> None:
    payload = joblib.load(path)
    payload.update(changes)
    joblib.dump(payload, path)


def test_format_version_mismatch_raises(tmp_path, trained_run):
    path = tmp_path / "m.joblib"
    save_bundle(trained_run(DATA_V1, FeatureSet.V2), path, DATA_V1, margin=1.0)
    _rewrite(path, format_version=FORMAT_VERSION + 1)
    with pytest.raises(ValueError, match=f"format_version {FORMAT_VERSION + 1}.*expects {FORMAT_VERSION}"):
        load_bundle(path)


def _save_format_1(result, path, data, margin: float) -> None:
    """Write a bundle exactly as the Phase 8-9 ``save_bundle`` did (format_version 1, no ``extras`` key)."""
    design_columns = tuple(result.design.columns)
    value_columns = tuple(result.deal_value.ridge.feature_names_in_)
    joblib.dump({
        "format_version": 1, "feature_set": result.features, "labels": result.labels, "margin": float(margin),
        "design_columns": design_columns, "value_columns": value_columns,
        "levels": model_levels(result.features, design_columns, value_columns),
        "model": result.model, "deal_value": result.deal_value, "value_transform": result.value_transform,
        "weights": result.weights, "lead_schema": read_leads(data / "historical_leads.csv").iloc[:0],
        "created_at": "2026-09-20T12:00:00+00:00", "data_name": data.name, "versions": library_versions()}, path)


@pytest.mark.parametrize("feature_set", [FeatureSet.V2, FeatureSet.LEGACY])
def test_format_1_bundle_loads_without_extras_and_scores_as_before(tmp_path, trained_run, feature_set):
    """Phase 10 (b) ruling: runs trained before the generic feature set (e.g. on Keel) keep scoring."""
    result = trained_run(DATA_V1, feature_set)
    _save_format_1(result, tmp_path / "old.joblib", DATA_V1, margin=1.0)
    save_bundle(result, tmp_path / "new.joblib", DATA_V1, margin=1.0)
    old, new = load_bundle(tmp_path / "old.joblib"), load_bundle(tmp_path / "new.joblib")
    assert old.format_version == 1 and old.extras is None and old.feature_set is feature_set
    leads = read_leads(DATA_V1 / "historical_leads.csv").iloc[:50].reset_index()
    pd.testing.assert_frame_equal(score_leads(old, leads, DATA_V1), score_leads(new, leads, DATA_V1))


def test_format_1_with_an_extras_field_is_refused(tmp_path, trained_run):
    path = tmp_path / "m.joblib"
    save_bundle(trained_run(DATA_V1, FeatureSet.V2), path, DATA_V1, margin=1.0)
    _rewrite(path, format_version=1)  # keeps the extras key: not what a format-1 writer produced
    with pytest.raises(ValueError, match="format_version 1 bundle with an extras field"):
        load_bundle(path)


def test_format_0_is_refused(tmp_path, trained_run):
    path = tmp_path / "m.joblib"
    save_bundle(trained_run(DATA_V1, FeatureSet.V2), path, DATA_V1, margin=1.0)
    _rewrite(path, format_version=0)
    with pytest.raises(ValueError, match="format_version 0"):
        load_bundle(path)


def test_not_a_bundle_raises(tmp_path):
    joblib.dump([1, 2], tmp_path / "m.joblib")
    with pytest.raises(ValueError, match="not an EMVA model bundle"):
        load_bundle(tmp_path / "m.joblib")


def test_library_version_mismatch_warns(tmp_path, trained_run):
    path = tmp_path / "m.joblib"
    save_bundle(trained_run(DATA_V1, FeatureSet.V2), path, DATA_V1, margin=1.0)
    versions = joblib.load(path)["versions"]
    _rewrite(path, versions={**versions, "scikit-learn": "0.1.0"})
    with pytest.warns(UserWarning, match="scikit-learn 0.1.0"):
        load_bundle(path)


def test_cli_writes_a_loadable_bundle(tmp_path):
    proc = subprocess.run([sys.executable, "-m", "emva", "--data", str(DATA_V1), "--out", str(tmp_path)],
                          cwd=REPO, capture_output=True, text=True, check=True)
    assert "model.joblib" not in proc.stdout
    b = load_bundle(tmp_path / "model.joblib")
    assert b.feature_set is FeatureSet.V2 and b.value_transform is not None and b.data_name == "v1"
