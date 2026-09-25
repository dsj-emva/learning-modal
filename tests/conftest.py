"""Shared fixtures: the v1 data directory and one pipeline run on it (session-scoped, ~3 s)."""
from __future__ import annotations

from pathlib import Path

import pytest

from emva.pipeline import PipelineResult, run

REPO = Path(__file__).resolve().parents[1]
DATA_V1 = REPO / "data" / "v1"


@pytest.fixture(scope="session")
def data_v1() -> Path:
    return DATA_V1


@pytest.fixture(scope="session")
def v1_result() -> PipelineResult:
    return run(DATA_V1)
