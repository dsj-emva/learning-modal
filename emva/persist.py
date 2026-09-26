"""Persist a trained pipeline as a ``ModelBundle`` (``model.joblib``) so new leads can be scored without retraining.

A bundle holds everything ``emva.scoring`` needs to score a new lead exactly as the training run scored its
own leads: the feature set, label config and margin; the fitted formula model and its design columns; the
deal-value model and its design columns; the fitted value transform (None in legacy label mode); the
scorecard; the raw ``historical_leads.csv`` schema (column dtypes, used to coerce form input); each model
feature's allowed levels; the extras' frozen encoder (generic feature set, else None); and provenance (UTC
creation time, data directory name, library versions).

The file is a joblib pickle of a plain dict with a ``format_version`` key, checked before the bundle is
built, so a bundle from an incompatible release fails with a clear ``ValueError`` instead of an unpickling
error deep inside a dataclass. Format 1 (Phase 8-9, before the extras) still loads, as a bundle with
``extras = None``: its fields are format 2's without ``extras``, so it scores exactly as its v2 / legacy run did
(ADR 0024, Phase 10 (b)). Unpickling runs code: load only bundles this repo wrote.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression

from emva.constants import DEAL_VALUE_FEATURES
from emva.feature_spec import feature_spec
from emva.features import FeatureSet
from emva.generic import GenericEncoder
from emva.io import read_leads
from emva.labels import LabelConfig
from emva.pipeline import PipelineResult
from emva.value import DealValueModel
from emva.value_transform import FittedValueTransform

# Bump when a field is added, removed or changes meaning; load_bundle refuses any version not in READABLE_FORMATS.
# 2 (Phase 10): the ``extras`` field (the generic feature set's encoder).
FORMAT_VERSION: int = 2
# Versions load_bundle reads: 1 (Phase 8-9 runs, e.g. Keel's, no ``extras`` field) and the current one.
READABLE_FORMATS: frozenset[int] = frozenset({1, FORMAT_VERSION})
# File name ``python -m emva`` writes into ``--out``.
BUNDLE_FILE: str = "model.joblib"


def library_versions() -> dict[str, str]:
    """The versions of the libraries whose objects a bundle pickles or whose numerics it depends on."""
    return {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__}


@dataclass(frozen=True, eq=False)
class ModelBundle:
    """A trained pipeline, ready to score new leads (``emva.scoring``).

    ``design_columns`` / ``value_columns`` are the formula / deal-value design columns in fit order;
    ``levels`` maps every categorical model feature (formula and deal-value) to the levels the fitted models
    can score (a lead with any other level is refused); ``lead_schema`` is an empty frame with the dtypes of
    the training ``historical_leads.csv`` (index ``lead_id``); ``weights`` is the ``weights.csv`` scorecard;
    ``value_transform`` is None in legacy label mode (no ``value_at_submit``); ``extras`` is the generic feature set's
    encoder of the declared extra columns, fitted on the training leads (None for the other feature sets; its
    ``x_`` columns are part of ``design_columns``, and an unseen categorical value is ``other``, not refused).
    Provenance: ``created_at`` (UTC ISO 8601), ``data_name`` (the training data directory's name), ``versions`` (``library_versions``).
    """

    format_version: int
    feature_set: FeatureSet
    labels: LabelConfig
    margin: float
    design_columns: tuple[str, ...]
    value_columns: tuple[str, ...]
    levels: dict[str, tuple[str, ...]]
    model: LogisticRegression
    deal_value: DealValueModel
    value_transform: FittedValueTransform | None
    weights: pd.DataFrame
    lead_schema: pd.DataFrame
    created_at: str
    data_name: str
    versions: dict[str, str]
    extras: GenericEncoder | None = None


def _column_levels(columns: tuple[str, ...], feature: str) -> list[str]:
    """Levels of ``feature`` that have a column: ``<feature>=<level>`` (both formula designs and the v2 value
    design) or ``<feature>_<level>`` (the legacy value design, ``pd.get_dummies`` default separator)."""
    return [c[len(feature) + 1:] for c in columns if c.startswith((f"{feature}=", f"{feature}_"))]


def model_levels(feature_set: FeatureSet, design_columns: tuple[str, ...],
                 value_columns: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """The levels each model feature can take for the fitted models (``ModelBundle.levels``).

    A formula feature's levels are its reference level plus every level with a design column; a deal-value
    feature's levels are those with a value-design column (the ridge keeps every level); a feature in both
    models gets the levels both accept, in formula order. For v2 this is every declared level
    (``V2_LEVELS``, ``DEAL_VALUE_LEVELS``); for the legacy feature set, the levels seen in training.
    """
    cats = feature_spec(feature_set).cats
    levels = {f: [ref, *_column_levels(design_columns, f)] for f, ref in cats.items()}
    for f in DEAL_VALUE_FEATURES:
        value = _column_levels(value_columns, f)
        levels[f] = [lvl for lvl in levels[f] if lvl in value] if f in levels else value
    return {f: tuple(v) for f, v in levels.items()}


def save_bundle(result: PipelineResult, path: str | Path, data: str | Path, margin: float) -> None:
    """Write ``result``'s fitted models as a ``ModelBundle`` to ``path`` (joblib).

    ``data`` is the dataset directory ``result`` was trained on (its ``historical_leads.csv`` gives the lead
    schema, its name the provenance); ``margin`` the margin ``emva.pipeline.run`` was called with (the
    result does not record it). Only the formula model is saved: context-mode models (``--context``) are not.
    """
    design_columns = tuple(result.design.columns)
    value_columns = tuple(result.deal_value.ridge.feature_names_in_)
    bundle = ModelBundle(
        format_version=FORMAT_VERSION, feature_set=result.features, labels=result.labels, margin=float(margin),
        design_columns=design_columns, value_columns=value_columns,
        levels=model_levels(result.features, design_columns, value_columns),
        model=result.model, deal_value=result.deal_value, value_transform=result.value_transform,
        weights=result.weights, lead_schema=read_leads(Path(data) / "historical_leads.csv").iloc[:0],
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), data_name=Path(data).resolve().name,
        versions=library_versions(), extras=result.extras)
    joblib.dump({f.name: getattr(bundle, f.name) for f in fields(bundle)}, path)  # shallow: nested objects stay objects


def load_bundle(path: str | Path) -> ModelBundle:
    """Read a bundle written by ``save_bundle``.

    Raises ``ValueError`` if the file is not a bundle or its ``format_version`` is not in ``READABLE_FORMATS``;
    a format-1 bundle (no ``extras`` field) is returned with ``extras = None`` and its ``format_version`` 1. Warns (``UserWarning``) when it was written with other numpy / pandas / scikit-learn versions, since
    scores may then differ from the training run's.
    """
    payload = joblib.load(path)
    if not isinstance(payload, dict) or "format_version" not in payload:
        raise ValueError(f"{path} is not an EMVA model bundle")
    if payload["format_version"] not in READABLE_FORMATS:
        raise ValueError(f"{path} has bundle format_version {payload['format_version']}; this code expects "
                         f"{FORMAT_VERSION} (or reads {sorted(READABLE_FORMATS)}). Retrain with python -m emva "
                         "--out DIR to write a current bundle.")
    if payload["format_version"] == 1:
        if "extras" in payload:
            raise ValueError(f"{path} is a format_version 1 bundle with an extras field; format 1 has none")
        payload = {**payload, "extras": None}
    now = library_versions()
    differ = [f"{lib} {v} (installed {now.get(lib)})" for lib, v in payload["versions"].items() if now.get(lib) != v]
    if differ:
        warnings.warn(f"{path} was written with other library versions: {', '.join(differ)}; scores may differ "
                      "from the training run's", UserWarning, stacklevel=2)
    return ModelBundle(**payload)


__all__ = ["BUNDLE_FILE", "FORMAT_VERSION", "READABLE_FORMATS", "ModelBundle", "library_versions", "load_bundle",
           "model_levels", "save_bundle"]
