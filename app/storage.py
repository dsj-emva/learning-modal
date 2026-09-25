"""Dataset store and run registry under the app's data root (``DATA_DIR``). Pure Python, no Streamlit.

Layout, created by ``init_root``::

    DATA_DIR/
      registry.json            {"runs": [Run, ...]}  (atomic writes, guarded by registry.lock)
      datasets/<name>/         uploaded training files + dataset.json (a Dataset)
      runs/<run_id>/           python -m emva --out (scores.csv, weights.csv, model.joblib), train.log, report.txt

Only training-input files are accepted (``TRAINING_FILES``). A file named like evaluation-only ground truth is
refused by name before anything reads it: the app never opens such a file (CLAUDE.md ground rule 2).
Invalid user input raises ``StorageError`` (a ``ValueError``) with a message fit to show in the UI.
"""
from __future__ import annotations

import csv
import fcntl
import json
import os
import re
import secrets
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
# The bundled synthetic sample, offered read-only in the training panel.
SAMPLE_DATASET_NAME: str = "sample-v1"
SAMPLE_DATASET_PATH: Path = REPO_ROOT / "data" / "v1"

LEADS_FILE, CRM_FILE, COMPANIES_FILE, PEOPLE_FILE, RULES_FILE = (
    "historical_leads.csv", "crm_history.csv", "companies.csv", "people.csv", "status_quo_rules.json")
# Every file a dataset may hold; the pipeline needs the first three, people.csv is never read by the model and
# the rules only feed the status-quo benchmark.
TRAINING_FILES: tuple[str, ...] = (LEADS_FILE, CRM_FILE, COMPANIES_FILE, PEOPLE_FILE, RULES_FILE)
REQUIRED_FILES: tuple[str, ...] = (LEADS_FILE, CRM_FILE, COMPANIES_FILE)
# Evaluation-only file names (ground truth) are refused on sight, never opened.
FORBIDDEN_PREFIX: str = "ground_truth"

REGISTRY_FILE: str = "registry.json"
DATASET_META: str = "dataset.json"
LOG_FILE: str = "train.log"
REPORT_FILE: str = "report.txt"
RUN_STATUSES: tuple[str, ...] = ("queued", "running", "succeeded", "failed")

_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")


class StorageError(ValueError):
    """A request the store refuses: bad name, unknown or forbidden file, missing required file, unknown run."""


def utc_now() -> str:
    """Current UTC time as ISO 8601 with seconds, e.g. ``2026-09-25T10:15:00+00:00``."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_name(name: str) -> str:
    """Return ``name`` if it is a slug (lower-case letters, digits, ``-``, ``_``; starts alphanumeric; <= 63
    chars), else raise ``StorageError``. Slugs cannot contain ``/`` or ``..``, so no path traversal."""
    if not isinstance(name, str) or not _SLUG.fullmatch(name):
        raise StorageError(f"'{name}' is not a valid name: use lower-case letters, digits, '-' and '_' "
                           "(start with a letter or digit, at most 63 characters)")
    return name


def check_file_name(name: str) -> None:
    """Raise ``StorageError`` unless ``name`` is one of ``TRAINING_FILES``.

    A ground-truth file name gets its own message: those files are evaluation-only and must not be uploaded.
    Only the name is checked; the file is never opened.
    """
    base = Path(name).name
    if base.lower().startswith(FORBIDDEN_PREFIX):
        raise StorageError(f"'{base}' looks like an evaluation-only ground-truth file. The app never accepts "
                           "ground truth: upload only CRM exports (leads, CRM history, companies, people, rules).")
    if base != name or base not in TRAINING_FILES:
        raise StorageError(f"'{name}' is not a training file; expected one of: {', '.join(TRAINING_FILES)}")


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically: a temp file in the same directory, fsync, then ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, obj: object) -> None:
    """``atomic_write_text`` of ``obj`` as indented JSON."""
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


def init_root(root: str | Path) -> Path:
    """Create the data root layout (``datasets/``, ``runs/``, an empty ``registry.json``) if missing; return it."""
    root = Path(root)
    (root / "datasets").mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(parents=True, exist_ok=True)
    if not (root / REGISTRY_FILE).exists():
        atomic_write_json(root / REGISTRY_FILE, {"runs": []})
    return root


@dataclass(frozen=True)
class Dataset:
    """A set of training files: ``name`` (slug), ``path`` (directory), ``created_at`` (UTC ISO), ``rows`` (data
    rows per CSV; the rules JSON counts 1), ``read_only`` (the bundled sample)."""

    name: str
    path: str
    created_at: str
    rows: dict[str, int]
    read_only: bool = False

    @property
    def n_leads(self) -> int:
        """Rows in ``historical_leads.csv``."""
        return self.rows.get(LEADS_FILE, 0)

    @property
    def has_rules(self) -> bool:
        """True when the dataset carries ``status_quo_rules.json`` (needed for the status-quo benchmark)."""
        return RULES_FILE in self.rows


def _count_rows(path: Path) -> int:
    """Data rows of a CSV (quoted newlines respected) or 1 for a JSON file."""
    if path.suffix == ".json":
        return 1
    with open(path, newline="", encoding="utf-8") as f:
        return max(sum(1 for _ in csv.reader(f)) - 1, 0)


def _dataset_from_dir(name: str, path: Path, created_at: str, read_only: bool) -> Dataset:
    """A ``Dataset`` for the training files present in ``path`` (other files are ignored, never opened)."""
    rows = {f: _count_rows(path / f) for f in TRAINING_FILES if (path / f).exists()}
    return Dataset(name=name, path=str(path), created_at=created_at, rows=rows, read_only=read_only)


def save_dataset(root: str | Path, name: str, files: dict[str, bytes]) -> Dataset:
    """Store uploaded ``files`` (file name -> content) as dataset ``name`` under ``root/datasets/`` and return it.

    Raises ``StorageError`` for an invalid or taken name (including the sample's), an unknown or ground-truth
    file name, or a missing required file. The directory appears atomically (written aside, then renamed).
    Content is not schema-checked here: run ``app.validation.validate_files`` first.
    """
    validate_name(name)
    if name == SAMPLE_DATASET_NAME:
        raise StorageError(f"'{name}' is reserved for the bundled sample dataset")
    for f in files:
        check_file_name(f)
    missing = [f for f in REQUIRED_FILES if f not in files]
    if missing:
        raise StorageError(f"missing required file(s): {', '.join(missing)}")
    datasets = init_root(root) / "datasets"
    target = datasets / name
    if target.exists():
        raise StorageError(f"a dataset named '{name}' already exists; choose another name")
    staging = Path(tempfile.mkdtemp(dir=datasets, prefix=f".{name}."))
    try:
        for f, content in files.items():
            (staging / f).write_bytes(content)
        ds = Dataset(name=name, path=str(target), created_at=utc_now(),
                     rows={f: _count_rows(staging / f) for f in TRAINING_FILES if f in files})
        atomic_write_json(staging / DATASET_META, asdict(ds))
        os.rename(staging, target)
    except BaseException:
        for p in staging.glob("*"):
            p.unlink()
        staging.rmdir()
        raise
    return ds


def sample_dataset() -> Dataset:
    """The bundled synthetic sample (``data/v1``) as a read-only ``Dataset``; only training files are read."""
    return _dataset_from_dir(SAMPLE_DATASET_NAME, SAMPLE_DATASET_PATH, "2026-09-24T00:00:00+00:00", read_only=True)


def list_datasets(root: str | Path, include_sample: bool = True) -> list[Dataset]:
    """Stored datasets, newest first, then (with ``include_sample``) the bundled sample if it exists."""
    base = Path(root) / "datasets"
    out = []
    for meta in base.glob(f"*/{DATASET_META}") if base.exists() else []:
        d = json.loads(meta.read_text(encoding="utf-8"))
        out.append(Dataset(**{**d, "path": str(meta.parent)}))
    out.sort(key=lambda d: d.created_at, reverse=True)
    if include_sample and (SAMPLE_DATASET_PATH / LEADS_FILE).exists():
        out.append(sample_dataset())
    return out


def get_dataset(root: str | Path, name: str) -> Dataset:
    """The dataset called ``name`` (stored or the sample); ``StorageError`` if there is none."""
    for d in list_datasets(root):
        if d.name == name:
            return d
    raise StorageError(f"no dataset named '{name}'")


@dataclass(frozen=True)
class Run:
    """One training run. ``args`` are the CLI options (``label_mode``, ``feature_set``); ``status`` is one of
    ``RUN_STATUSES``; ``metrics`` the pipeline's printed summary row (auc, brier, top20_wins, top20_revenue) once
    finished; ``has_report`` whether ``report.txt`` was written; ``pid`` the job process while it runs."""

    run_id: str
    dataset: str
    dataset_path: str
    created_at: str
    args: dict[str, str]
    status: str
    log_path: str
    out_dir: str
    metrics: dict[str, float] = field(default_factory=dict)
    finished_at: str | None = None
    has_report: bool = False
    pid: int | None = None
    error: str | None = None

    @property
    def report_path(self) -> Path:
        """``<out_dir>/report.txt``."""
        return Path(self.out_dir) / REPORT_FILE

    @property
    def is_done(self) -> bool:
        """True once the run has succeeded or failed."""
        return self.status in ("succeeded", "failed")


def new_run_id() -> str:
    """A sortable, unique run id: UTC timestamp plus 4 random hex digits, e.g. ``20260925-101500-a1b2``."""
    return f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"


def run_dir(root: str | Path, run_id: str) -> Path:
    """``root/runs/<run_id>`` (not created); ``StorageError`` for an id that is not a slug."""
    return Path(root) / "runs" / validate_name(run_id)


@contextmanager
def _locked_registry(root: Path) -> Iterator[dict]:
    """Hold an exclusive lock on the registry, yield its parsed content and write it back atomically."""
    init_root(root)
    with open(root / "registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            reg = json.loads((root / REGISTRY_FILE).read_text(encoding="utf-8"))
            yield reg
            atomic_write_json(root / REGISTRY_FILE, reg)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def new_run(root: str | Path, dataset: Dataset, args: dict[str, str]) -> Run:
    """A queued ``Run`` for ``dataset`` with a fresh id and its directory created (not yet registered)."""
    run_id = new_run_id()
    out = run_dir(Path(root).resolve(), run_id)  # absolute, so the job and the UI agree whatever their cwd
    out.mkdir(parents=True)
    return Run(run_id=run_id, dataset=dataset.name, dataset_path=dataset.path, created_at=utc_now(), args=dict(args),
               status="queued", log_path=str(out / LOG_FILE), out_dir=str(out))


def register_run(root: str | Path, run: Run) -> Run:
    """Add ``run`` to the registry; ``StorageError`` if its id is already there. Returns ``run``."""
    with _locked_registry(Path(root)) as reg:
        if any(r["run_id"] == run.run_id for r in reg["runs"]):
            raise StorageError(f"run '{run.run_id}' is already registered")
        reg["runs"].append(asdict(run))
    return run


def update_run(root: str | Path, run_id: str, **changes: object) -> Run:
    """Apply ``changes`` (``Run`` field values) to the registered run ``run_id``; return the updated run.

    Raises ``StorageError`` for an unknown run, field or status.
    """
    unknown = set(changes) - set(Run.__dataclass_fields__) | ({"run_id"} & set(changes))
    if unknown:
        raise StorageError(f"cannot update run field(s): {sorted(unknown)}")
    if "status" in changes and changes["status"] not in RUN_STATUSES:
        raise StorageError(f"unknown run status {changes['status']!r}")
    with _locked_registry(Path(root)) as reg:
        for i, r in enumerate(reg["runs"]):
            if r["run_id"] == run_id:
                reg["runs"][i] = {**r, **changes}
                return Run(**reg["runs"][i])
    raise StorageError(f"no run '{run_id}'")


def list_runs(root: str | Path, where: Callable[[Run], bool] | None = None) -> list[Run]:
    """Registered runs, newest first, optionally filtered by ``where``."""
    path = Path(root) / REGISTRY_FILE
    if not path.exists():
        return []
    runs = [Run(**r) for r in json.loads(path.read_text(encoding="utf-8"))["runs"]][::-1]
    runs.sort(key=lambda r: r.created_at, reverse=True)  # stable: same-second runs stay newest-registered first
    return [r for r in runs if where is None or where(r)]


def get_run(root: str | Path, run_id: str) -> Run:
    """The registered run ``run_id``; ``StorageError`` if there is none."""
    for r in list_runs(root):
        if r.run_id == run_id:
            return r
    raise StorageError(f"no run '{run_id}'")


__all__ = ["COMPANIES_FILE", "CRM_FILE", "Dataset", "LEADS_FILE", "PEOPLE_FILE", "REQUIRED_FILES", "RULES_FILE",
           "Run", "SAMPLE_DATASET_NAME", "StorageError", "TRAINING_FILES", "atomic_write_json", "check_file_name",
           "get_dataset", "get_run", "init_root", "list_datasets", "list_runs", "new_run", "register_run",
           "run_dir", "sample_dataset", "save_dataset", "update_run", "validate_name"]
