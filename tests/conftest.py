"""Shared fixtures: the v1 data directory and pipeline runs on it (session-scoped, ~3 s each)."""
from __future__ import annotations

from collections.abc import Callable
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


DATA_V2 = REPO / "data" / "v2"


@pytest.fixture(scope="session")
def trained_run() -> Callable[[Path, FeatureSet], PipelineResult]:
    """``trained_run(data, feature_set)``: the default horizon-label pipeline run, computed once per pair (~1 s)."""
    cache: dict[tuple[Path, FeatureSet], PipelineResult] = {}

    def get(data: Path, feature_set: FeatureSet) -> PipelineResult:
        if (data, feature_set) not in cache:
            cache[data, feature_set] = run(data, labels=HORIZON, features=feature_set)
        return cache[data, feature_set]

    return get


# --- app (Phase 8): one trained run shared by the app tests ------------------------------------------------------
APP_SUBSAMPLE_LEADS = 1000
APP_REPORT_RESAMPLES = 200


def _app_subsample(out: Path, n: int = APP_SUBSAMPLE_LEADS) -> dict[str, bytes]:
    """v1 training files cut to ``n`` random leads (seed 0) and their CRM rows; never touches ground truth."""
    import pandas as pd

    L = pd.read_csv(DATA_V1 / "historical_leads.csv", dtype=str).sample(n=n, random_state=0).sort_values("created_at")
    C = pd.read_csv(DATA_V1 / "crm_history.csv", dtype=str)
    C = C[C.lead_id.isin(L.lead_id)]
    files = {n_: (DATA_V1 / n_).read_bytes() for n_ in ("companies.csv", "people.csv", "status_quo_rules.json")}
    return {**files, "historical_leads.csv": L.to_csv(index=False).encode(),
            "crm_history.csv": C.to_csv(index=False).encode()}


@pytest.fixture(scope="session")
def app_trained(tmp_path_factory):
    """``(root, run)``: an app data root with a 1,000-lead v1 subsample trained end to end by
    ``app.training.start_training`` (horizon + v2, report with 200 resamples). Shared by the app tests."""
    import sys

    from app import storage, training

    root = storage.init_root(tmp_path_factory.mktemp("app-data"))
    ds = storage.save_dataset(root, "v1-sample-1000", _app_subsample(root))
    run = storage.register_run(root, storage.new_run(root, ds, training.TrainingConfig().as_dict()))
    proc = training.start_training(root, run, python=sys.executable, report_resamples=APP_REPORT_RESAMPLES)
    assert proc.wait(timeout=180) == 0, Path(run.log_path).read_text()
    return root, storage.get_run(root, run.run_id)
