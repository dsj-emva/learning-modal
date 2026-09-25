"""Shared fixtures: the v1 data directory and pipeline runs on it (session-scoped, ~3 s each)."""
from __future__ import annotations

from pathlib import Path

import pytest

from emva.labels import HORIZON, LEGACY
from emva.pipeline import PipelineResult, run

REPO = Path(__file__).resolve().parents[1]
DATA_V1 = REPO / "data" / "v1"


@pytest.fixture(scope="session")
def data_v1() -> Path:
    return DATA_V1


@pytest.fixture(scope="session")
def v1_result() -> PipelineResult:
    """Legacy labels: must equal the frozen baseline."""
    return run(DATA_V1, labels=LEGACY)


@pytest.fixture(scope="session")
def v1_horizon() -> PipelineResult:
    """Default horizon labels (plan Phase 1)."""
    return run(DATA_V1, labels=HORIZON)
