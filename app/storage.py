"""Dataset store and run registry under the app's data root (``DATA_DIR``). Pure Python, no Streamlit.

Layout, created by ``init_root``::

    DATA_DIR/
      registry.json            {"runs": [Run, ...]}  (atomic writes, guarded by registry.lock)
      datasets/<name>/         training files + keel_meta.json (the store's Dataset record); a converted dataset
                               also holds mapping.toml (the confirmed mapping) and dataset.json (dates, coverage),
                               and, when its mapping declares extras, extra_features.csv (a training file)
      runs/<run_id>/           python -m emva --out (scores.csv, weights.csv, model.joblib), train.log, report.txt

Only training-input files are accepted (``TRAINING_FILES``), plus the two metadata files a converted dataset carries
(``METADATA_FILES``). A file named like evaluation-only ground truth is refused by name before anything reads it: the
app never opens such a file (CLAUDE.md ground rule 2).
Invalid user input raises ``StorageError`` (a ``ValueError``) with a message fit to show in the UI.
"""
from __future__ import annotations

import csv
import fcntl
import json
import logging
import os
import re
import secrets
import subprocess
import tempfile
import tomllib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from emva.generic import EXTRA_FEATURES_FILE

log = logging.getLogger(__name__)

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
# The bundled synthetic sample, offered read-only in the training panel.
SAMPLE_DATASET_NAME: str = "sample-v1"
SAMPLE_DATASET_PATH: Path = REPO_ROOT / "data" / "v1"

LEADS_FILE, CRM_FILE, COMPANIES_FILE, PEOPLE_FILE, RULES_FILE = (
    "historical_leads.csv", "crm_history.csv", "companies.csv", "people.csv", "status_quo_rules.json")
# The generic feature set's raw extras (``EXTRA_FEATURES_FILE``, imported from emva.generic, Phase 10), written by the
# converter when the mapping declares [[features]]. It is training data (the generic feature set learns from it), so it is a training
# file, counted in ``Dataset.rows``; its columns are declared in the converter's dataset.json, so only a converted
# dataset carries it (the EMVA-format upload has no slot for it) and ``app.validation`` refuses it without that
# declaration (Phase 10 (b)).
# Every file a dataset may hold; the pipeline needs the first three, people.csv is never read by the model, the rules
# only feed the status-quo benchmark and the extras only the generic feature set.
TRAINING_FILES: tuple[str, ...] = (LEADS_FILE, CRM_FILE, COMPANIES_FILE, PEOPLE_FILE, RULES_FILE, EXTRA_FEATURES_FILE)
REQUIRED_FILES: tuple[str, ...] = (LEADS_FILE, CRM_FILE, COMPANIES_FILE)
# A dataset converted from a foreign CSV also keeps the mapping it was converted with and the converter's metadata
# (the dataset's own as_of / test_from and a coverage table). The store writes them as given and reads only the
# mapping's ``source_url``; a dataset that carries ``dataset.json`` is "converted".
MAPPING_FILE, DATASET_META_FILE = "mapping.toml", "dataset.json"
METADATA_FILES: tuple[str, ...] = (MAPPING_FILE, DATASET_META_FILE)
# Evaluation-only file names (ground truth) are refused on sight, never opened.
FORBIDDEN_PREFIX: str = "ground_truth"

REGISTRY_FILE: str = "registry.json"
# The store's own record of a dataset (a ``Dataset`` as JSON). Before Phase 9 it was called dataset.json, which is
# now the converter's file; ``init_root`` renames such legacy records (``_migrate_store_meta``).
STORE_META: str = "keel_meta.json"
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


def check_file_name(name: str, allow_metadata: bool = False) -> None:
    """Raise ``StorageError`` unless ``name`` is one of ``TRAINING_FILES`` (or, with ``allow_metadata``, one of
    ``METADATA_FILES``).

    A ground-truth file name gets its own message: those files are evaluation-only and must not be uploaded.
    Only the name is checked; the file is never opened.
    """
    allowed = TRAINING_FILES + METADATA_FILES if allow_metadata else TRAINING_FILES
    base = Path(name).name
    if base.lower().startswith(FORBIDDEN_PREFIX):
        raise StorageError(f"'{base}' looks like an evaluation-only ground-truth file. The app never accepts "
                           "ground truth: upload only CRM exports (leads, CRM history, companies, people, rules).")
    if base != name or base not in allowed:
        raise StorageError(f"'{name}' is not a training file; expected one of: {', '.join(allowed)}")


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


def _is_store_record(obj: object) -> bool:
    """True when ``obj`` (parsed JSON) is a ``Dataset`` record as the store writes it, not the converter's file."""
    return (isinstance(obj, dict) and {"name", "created_at", "rows"} <= obj.keys()
            and obj.keys() <= Dataset.__dataclass_fields__.keys())


def _migrate_store_meta(datasets: Path) -> None:
    """Rename pre-Phase 9 store records (``dataset.json`` holding a ``Dataset``) to ``STORE_META``, so that
    ``dataset.json`` only ever means the converter's metadata. A converter ``dataset.json`` is left alone, and so is
    a corrupt one (logged as a warning; the dataset's validation refuses it loudly later)."""
    for legacy in datasets.glob(f"*/{DATASET_META_FILE}"):
        if (legacy.parent / STORE_META).exists():
            continue
        try:
            obj = json.loads(legacy.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            log.warning("not migrating %s: it is not valid JSON (%s); validation refuses it when the dataset is used",
                        legacy, e)
            continue
        if _is_store_record(obj):
            os.replace(legacy, legacy.parent / STORE_META)
            log.info("renamed the store record of dataset %s to %s", legacy.parent.name, STORE_META)


def init_root(root: str | Path) -> Path:
    """Create the data root layout (``datasets/``, ``runs/``, an empty ``registry.json``) if missing, migrate
    legacy dataset records (``_migrate_store_meta``) and return the root."""
    root = Path(root)
    (root / "datasets").mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(parents=True, exist_ok=True)
    if not (root / REGISTRY_FILE).exists():
        atomic_write_json(root / REGISTRY_FILE, {"runs": []})
    _migrate_store_meta(root / "datasets")
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

    @property
    def has_extras(self) -> bool:
        """True when the dataset carries ``extra_features.csv`` (the generic feature set can be trained on it). Read
        from disk, like ``has_mapping``, so a dataset stored while the file was still metadata counts too."""
        return (Path(self.path) / EXTRA_FEATURES_FILE).is_file()

    @property
    def has_mapping(self) -> bool:
        """True when the dataset keeps the ``mapping.toml`` it was converted with."""
        return (Path(self.path) / MAPPING_FILE).is_file()

    @property
    def converted(self) -> bool:
        """True when the dataset carries the converter's ``dataset.json`` (its own dates and coverage)."""
        return is_converted(self.path)


def is_converted(dataset_path: str | Path) -> bool:
    """True when the dataset directory holds the converter's ``dataset.json``: its data come from another period
    and format than the frozen baseline script assumes (ADR 0022)."""
    return (Path(dataset_path) / DATASET_META_FILE).is_file()


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

    ``files`` may also hold ``METADATA_FILES`` (a converted dataset's ``mapping.toml`` and ``dataset.json``),
    written as given. Raises ``StorageError`` for an invalid or taken name (including the sample's), an unknown or
    ground-truth file name, or a missing required file. The directory appears atomically (written aside, then renamed).
    Content is not schema-checked here: run ``app.validation.validate_files`` first.
    """
    validate_name(name)
    if name == SAMPLE_DATASET_NAME:
        raise StorageError(f"'{name}' is reserved for the bundled sample dataset")
    for f in files:
        check_file_name(f, allow_metadata=True)
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
        atomic_write_json(staging / STORE_META, asdict(ds))
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


def list_datasets(root: str | Path) -> list[Dataset]:
    """Stored datasets, newest first, then the bundled sample if it exists."""
    base = Path(root) / "datasets"
    out = []
    for meta in base.glob(f"*/{STORE_META}") if base.exists() else []:
        d = json.loads(meta.read_text(encoding="utf-8"))
        out.append(Dataset(**{**d, "path": str(meta.parent)}))
    out.sort(key=lambda d: d.created_at, reverse=True)
    if (SAMPLE_DATASET_PATH / LEADS_FILE).exists():
        out.append(sample_dataset())
    return out


def get_dataset(root: str | Path, name: str) -> Dataset:
    """The dataset called ``name`` (stored or the sample); ``StorageError`` if there is none."""
    for d in list_datasets(root):
        if d.name == name:
            return d
    raise StorageError(f"no dataset named '{name}'")


@dataclass(frozen=True)
class SavedMapping:
    """A stored dataset's ``mapping.toml``: ``name`` (the dataset's), ``path`` (the file), ``created_at`` (the
    dataset's, UTC ISO) and ``source_url`` (the mapping's top-level ``source_url``; None when absent or unreadable)."""

    name: str
    path: str
    created_at: str
    source_url: str | None = None


def _source_url(path: Path) -> str | None:
    """The top-level ``source_url`` string of the TOML file ``path``; None (logged) when it cannot be parsed."""
    try:
        doc = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        log.warning("cannot read source_url from %s: %s", path, e)
        return None
    url = doc.get("source_url")
    return url if isinstance(url, str) and url else None


def list_mappings(root: str | Path) -> list[SavedMapping]:
    """The ``mapping.toml`` of every stored dataset that keeps one, newest dataset first."""
    return [SavedMapping(name=d.name, path=str(Path(d.path) / MAPPING_FILE), created_at=d.created_at,
                         source_url=_source_url(Path(d.path) / MAPPING_FILE))
            for d in list_datasets(root) if d.has_mapping]


def read_mapping(root: str | Path, name: str) -> str:
    """The text of dataset ``name``'s ``mapping.toml``; ``StorageError`` when there is no such dataset or it keeps
    no mapping."""
    ds = get_dataset(root, validate_name(name))
    if not ds.has_mapping:
        raise StorageError(f"dataset '{name}' has no {MAPPING_FILE}")
    return (Path(ds.path) / MAPPING_FILE).read_text(encoding="utf-8")


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


INTERRUPTED: str = "interrupted (server restarted or the training job was stopped); see the log"


def process_command(pid: int) -> str | None:
    """The command line of process ``pid``, or None when there is no such process. Linux: ``/proc/<pid>/cmdline``
    (the slim image has no ``ps``); elsewhere (macOS): ``ps -o command= -p``."""
    proc = Path(f"/proc/{pid}/cmdline")
    if Path("/proc/self/cmdline").exists():
        try:
            return proc.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip() or None
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            return None
    out = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True)
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def job_alive(run: Run) -> bool:
    """True when ``run``'s job process still exists and is that run's job: its command line contains
    ``app.job`` and the run id (PIDs are reused, and restart low in a new container)."""
    if not run.pid:
        return False
    cmd = process_command(run.pid)
    return cmd is not None and "app.job" in cmd and run.run_id in cmd


def list_runs(root: str | Path, where: Callable[[Run], bool] | None = None) -> list[Run]:
    """Registered runs, newest first, optionally filtered by ``where``.

    A run registered as ``running`` whose job process is gone (``job_alive``) is marked ``failed`` with
    ``INTERRUPTED`` first, so a restart or redeploy never leaves runs spinning forever.
    """
    path = Path(root) / REGISTRY_FILE
    if not path.exists():
        return []
    runs = [Run(**r) for r in json.loads(path.read_text(encoding="utf-8"))["runs"]][::-1]
    runs = [update_run(root, r.run_id, status="failed", pid=None, finished_at=utc_now(), error=INTERRUPTED)
            if r.status == "running" and not job_alive(r) else r for r in runs]
    runs.sort(key=lambda r: r.created_at, reverse=True)  # stable: same-second runs stay newest-registered first
    return [r for r in runs if where is None or where(r)]


def get_run(root: str | Path, run_id: str) -> Run:
    """The registered run ``run_id``; ``StorageError`` if there is none."""
    for r in list_runs(root):
        if r.run_id == run_id:
            return r
    raise StorageError(f"no run '{run_id}'")


__all__ = ["COMPANIES_FILE", "CRM_FILE", "DATASET_META_FILE", "Dataset", "EXTRA_FEATURES_FILE", "LEADS_FILE",
           "MAPPING_FILE", "METADATA_FILES", "PEOPLE_FILE", "REQUIRED_FILES", "RULES_FILE", "Run",
           "SAMPLE_DATASET_NAME", "STORE_META", "SavedMapping", "StorageError", "TRAINING_FILES", "atomic_write_json",
           "check_file_name", "get_dataset", "get_run", "init_root", "is_converted", "list_datasets", "list_mappings",
           "list_runs", "new_run", "read_mapping", "register_run", "run_dir", "sample_dataset", "save_dataset",
           "update_run", "validate_name"]
