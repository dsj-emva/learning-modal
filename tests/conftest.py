"""Shared fixtures: the v1 data directory and pipeline runs on it (session-scoped, ~3 s each)."""
from __future__ import annotations

from pathlib import Path

import pytest

from emva.features import FeatureSet
from emva.labels import HORIZON, LEGACY
from emva.pipeline import PipelineResult, run

REPO = Path(__file__).resolve().parents[1]
DATA_V1 = REPO / "data" / "v1"


@pytest.fixture(scope="session")
def data_v1() -> Path:
    return DATA_V1


@pytest.fixture(scope="session")
def v1_result() -> PipelineResult:
    """Legacy labels and legacy features: must equal the frozen baseline."""
    return run(DATA_V1, labels=LEGACY, features=FeatureSet.LEGACY)


@pytest.fixture(scope="session")
def v1_horizon() -> PipelineResult:
    """The defaults: horizon labels (plan Phase 1) and v2 features (plan Phase 2)."""
    return run(DATA_V1, labels=HORIZON)


@pytest.fixture(scope="session")
def v1_v2_legacy_labels() -> PipelineResult:
    """v2 features with legacy labels (isolates the Phase 2 feature change)."""
    return run(DATA_V1, labels=LEGACY, features=FeatureSet.V2)
