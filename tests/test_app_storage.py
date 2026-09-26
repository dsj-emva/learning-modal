"""app.storage: dataset store, run registry, atomic writes, name and file-name checks."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import storage
from app.storage import StorageError

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "v1"


def _files(*names: str) -> dict[str, bytes]:
    """Tiny placeholder contents for the given training files."""
    return {n: (b"{}" if n.endswith(".json") else b"a,b\n1,2\n3,4\n") for n in names}


def test_init_root_creates_layout(tmp_path: Path) -> None:
    storage.init_root(tmp_path / "d")
    assert (tmp_path / "d" / "datasets").is_dir() and (tmp_path / "d" / "runs").is_dir()
    assert json.loads((tmp_path / "d" / "registry.json").read_text()) == {"runs": []}


def test_save_and_list_datasets(tmp_path: Path) -> None:
    ds = storage.save_dataset(tmp_path, "crm-2026", _files(*storage.REQUIRED_FILES, storage.RULES_FILE))
    assert ds.rows[storage.LEADS_FILE] == 2 and ds.rows[storage.RULES_FILE] == 1 and ds.has_rules
    assert (tmp_path / "datasets" / "crm-2026" / storage.LEADS_FILE).read_bytes() == b"a,b\n1,2\n3,4\n"
    names = [d.name for d in storage.list_datasets(tmp_path)]
    assert names[0] == "crm-2026" and names[-1] == storage.SAMPLE_DATASET_NAME
    assert storage.get_dataset(tmp_path, "crm-2026").path == str(tmp_path / "datasets" / "crm-2026")
    assert not list((tmp_path / "datasets").glob(".*"))  # no staging leftovers


def test_sample_dataset_is_read_only_and_counts_rows() -> None:
    s = storage.sample_dataset()
    assert s.read_only and s.n_leads == 10000 and s.has_rules
    assert set(s.rows) == set(storage.TRAINING_FILES)


def test_save_dataset_refuses_bad_input(tmp_path: Path) -> None:
    with pytest.raises(StorageError, match="missing required"):
        storage.save_dataset(tmp_path, "x", _files(storage.LEADS_FILE))
    with pytest.raises(StorageError, match="ground-truth"):
        storage.save_dataset(tmp_path, "x", {**_files(*storage.REQUIRED_FILES), "ground_truth_labels.csv": b"x"})
    with pytest.raises(StorageError, match="not a training file"):
        storage.save_dataset(tmp_path, "x", {**_files(*storage.REQUIRED_FILES), "notes.txt": b"x"})
    with pytest.raises(StorageError, match="reserved"):
        storage.save_dataset(tmp_path, storage.SAMPLE_DATASET_NAME, _files(*storage.REQUIRED_FILES))
    storage.save_dataset(tmp_path, "x", _files(*storage.REQUIRED_FILES))
    with pytest.raises(StorageError, match="already exists"):
        storage.save_dataset(tmp_path, "x", _files(*storage.REQUIRED_FILES))


@pytest.mark.parametrize("name", ["../x", "a/b", "..", "", "UPPER", "-lead", "x" * 64, "a b"])
def test_slug_validation_rejects(name: str, tmp_path: Path) -> None:
    with pytest.raises(StorageError, match="not a valid name"):
        storage.validate_name(name)
    with pytest.raises(StorageError):
        storage.save_dataset(tmp_path, name, _files(*storage.REQUIRED_FILES))
    assert not (tmp_path.parent / "x").exists()


@pytest.mark.parametrize("name", ["ground_truth_labels.csv", "Ground_Truth.md", "ground_truth_companies.csv"])
def test_ground_truth_names_rejected_without_reading(name: str) -> None:
    with pytest.raises(StorageError, match="ground-truth"):
        storage.check_file_name(name)


def test_register_update_list_runs(tmp_path: Path) -> None:
    ds = storage.sample_dataset()
    a = storage.register_run(tmp_path, storage.new_run(tmp_path, ds, {"label_mode": "horizon", "feature_set": "v2"}))
    b = storage.register_run(tmp_path, storage.new_run(tmp_path, ds, {"label_mode": "legacy", "feature_set": "v2"}))
    assert Path(a.out_dir).is_dir() and a.status == "queued"
    assert [r.run_id for r in storage.list_runs(tmp_path)] == [b.run_id, a.run_id]
    done = storage.update_run(tmp_path, a.run_id, status="succeeded", metrics={"auc": 0.8})
    assert done.metrics == {"auc": 0.8} and storage.get_run(tmp_path, a.run_id).status == "succeeded"
    assert [r.run_id for r in storage.list_runs(tmp_path, lambda r: r.is_done)] == [a.run_id]
    with pytest.raises(StorageError, match="already registered"):
        storage.register_run(tmp_path, a)
    with pytest.raises(StorageError, match="unknown run status"):
        storage.update_run(tmp_path, a.run_id, status="exploded")
    with pytest.raises(StorageError, match="cannot update"):
        storage.update_run(tmp_path, a.run_id, colour="red")
    with pytest.raises(StorageError, match="no run"):
        storage.update_run(tmp_path, "nope", status="failed")
    with pytest.raises(StorageError):
        storage.run_dir(tmp_path, "../../etc")


def test_atomic_write_leaves_old_file_on_failure(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    storage.atomic_write_json(path, {"v": 1})
    with pytest.raises(TypeError):
        storage.atomic_write_json(path, {"v": object()})  # not serialisable: fails before the rename
    assert json.loads(path.read_text()) == {"v": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["r.json"]


def _running(tmp_path: Path, pid: int) -> storage.Run:
    """A registered run marked running with ``pid``."""
    run = storage.register_run(tmp_path, storage.new_run(tmp_path, storage.sample_dataset(), {"label_mode": "horizon"}))
    return storage.update_run(tmp_path, run.run_id, status="running", pid=pid)


def test_list_runs_fails_a_run_whose_job_is_gone(tmp_path: Path) -> None:
    import subprocess
    import sys

    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    run = _running(tmp_path, dead.pid)
    got = storage.list_runs(tmp_path)[0]
    assert got.run_id == run.run_id and got.status == "failed" and got.error == storage.INTERRUPTED
    assert got.pid is None


def test_list_runs_fails_a_run_whose_pid_is_another_process(tmp_path: Path) -> None:
    import os

    run = _running(tmp_path, os.getpid())  # alive, but pytest, not this run's app.job
    assert storage.list_runs(tmp_path)[0].status == "failed"


def test_list_runs_keeps_a_live_job_running(tmp_path: Path) -> None:
    import subprocess
    import sys

    run = storage.register_run(tmp_path, storage.new_run(tmp_path, storage.sample_dataset(), {}))
    fake = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "app.job", "--run-id", run.run_id])
    try:
        storage.update_run(tmp_path, run.run_id, status="running", pid=fake.pid)
        assert storage.list_runs(tmp_path)[0].status == "running"
    finally:
        fake.kill()
        fake.wait()


MAPPING = b'source_url = "https://www.kaggle.com/datasets/example/leads"\n[columns]\nlead_id = "id"\n'
DATES = b'{"as_of": "2026-09-24", "test_from": "2026-05-01"}\n'


def test_save_dataset_keeps_metadata_files(tmp_path: Path) -> None:
    files = {**_files(*storage.REQUIRED_FILES), storage.MAPPING_FILE: MAPPING, storage.DATASET_META_FILE: DATES}
    ds = storage.save_dataset(tmp_path, "kaggle-leads", files)
    d = Path(ds.path)
    assert (d / storage.MAPPING_FILE).read_bytes() == MAPPING and (d / storage.DATASET_META_FILE).read_bytes() == DATES
    assert set(ds.rows) == set(storage.REQUIRED_FILES)  # metadata files are not training files
    assert (d / storage.STORE_META).exists() and ds.has_mapping and ds.converted
    got = storage.get_dataset(tmp_path, "kaggle-leads")
    assert got == ds and got.converted and storage.is_converted(got.path)
    plain = storage.save_dataset(tmp_path, "plain", _files(*storage.REQUIRED_FILES))
    assert not plain.has_mapping and not plain.converted
    assert not storage.sample_dataset().converted and not storage.sample_dataset().has_mapping


def test_metadata_names_accepted_only_when_allowed() -> None:
    for name in storage.METADATA_FILES:
        storage.check_file_name(name, allow_metadata=True)
        with pytest.raises(StorageError, match="not a training file"):
            storage.check_file_name(name)
    with pytest.raises(StorageError, match="ground-truth"):
        storage.check_file_name("ground_truth.md", allow_metadata=True)
    with pytest.raises(StorageError, match="not a training file"):
        storage.check_file_name("sub/mapping.toml", allow_metadata=True)


def test_list_and_read_mappings(tmp_path: Path) -> None:
    import time

    base = _files(*storage.REQUIRED_FILES)
    storage.save_dataset(tmp_path, "older", {**base, storage.MAPPING_FILE: b'[columns]\nlead_id = "id"\n'})
    storage.save_dataset(tmp_path, "no-mapping", base)
    time.sleep(1.1)  # created_at has one-second resolution
    storage.save_dataset(tmp_path, "newer", {**base, storage.MAPPING_FILE: MAPPING, storage.DATASET_META_FILE: DATES})
    got = storage.list_mappings(tmp_path)
    assert [m.name for m in got] == ["newer", "older"]
    assert got[0].source_url == "https://www.kaggle.com/datasets/example/leads" and got[1].source_url is None
    assert got[0].path == str(tmp_path / "datasets" / "newer" / storage.MAPPING_FILE)
    assert got[0].created_at == storage.get_dataset(tmp_path, "newer").created_at
    assert storage.read_mapping(tmp_path, "newer") == MAPPING.decode()
    with pytest.raises(StorageError, match="has no mapping.toml"):
        storage.read_mapping(tmp_path, "no-mapping")
    with pytest.raises(StorageError, match="no dataset"):
        storage.read_mapping(tmp_path, "missing")
    with pytest.raises(StorageError, match="not a valid name"):
        storage.read_mapping(tmp_path, "../older")


def test_unparsable_mapping_is_listed_without_source_url(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    storage.save_dataset(tmp_path, "broken", {**_files(*storage.REQUIRED_FILES), storage.MAPPING_FILE: b"= not toml"})
    assert [(m.name, m.source_url) for m in storage.list_mappings(tmp_path)] == [("broken", None)]
    assert "cannot read source_url" in caplog.text
    assert storage.read_mapping(tmp_path, "broken") == "= not toml"


def test_init_root_migrates_legacy_store_records(tmp_path: Path) -> None:
    """A pre-Phase 9 dataset kept its store record in dataset.json; init_root renames it, and leaves a converter's
    dataset.json (with its store record already in keel_meta.json) alone."""
    old = storage.save_dataset(tmp_path, "old", _files(*storage.REQUIRED_FILES))
    d = Path(old.path)
    (d / storage.STORE_META).rename(d / storage.DATASET_META_FILE)
    new = storage.save_dataset(tmp_path, "new", {**_files(*storage.REQUIRED_FILES), storage.DATASET_META_FILE: DATES})
    storage.init_root(tmp_path)
    assert (d / storage.STORE_META).exists() and not (d / storage.DATASET_META_FILE).exists()
    assert storage.get_dataset(tmp_path, "old") == old and not old.converted
    assert (Path(new.path) / storage.DATASET_META_FILE).read_bytes() == DATES
    orphan = tmp_path / "datasets" / "orphan"  # a converter file with no store record is not a Dataset: untouched
    orphan.mkdir()
    (orphan / storage.DATASET_META_FILE).write_bytes(DATES)
    storage.init_root(tmp_path)
    assert (orphan / storage.DATASET_META_FILE).read_bytes() == DATES and not (orphan / storage.STORE_META).exists()
