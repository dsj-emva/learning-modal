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
