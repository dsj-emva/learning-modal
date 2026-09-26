"""Per-dataset snapshot date and test boundary (ADR 0020): ``<data>/dataset.json`` if present, else the constants.

A dataset converted by ``emva.ingest`` comes from another period than the synthetic data, so it carries its own
``as_of`` (the snapshot date its outcomes were read at) and ``test_from`` (the frozen train/test boundary, set in
its mapping) in ``dataset.json``. data/v1, data/v2 and every dataset in the v1 format have no such file and keep
``emva.constants.AS_OF`` / ``TEST_FROM``, so their outputs do not change.

``dataset.json`` is written by ``emva.ingest.convert`` (with ``"emva_dataset_format": 1``, its dates and a coverage
table). The name is reserved for it: a ``dataset.json`` without ``as_of`` / ``test_from`` is refused, never read as
"no dates" (before Phase 9 the app kept its own dataset record under that name; the app now uses another file).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from emva.constants import AS_OF, TEST_FROM

# File name of the converter's metadata inside a dataset directory, and the marker key and version it carries.
DATASET_META_FILE: str = "dataset.json"
FORMAT_KEY: str = "emva_dataset_format"
FORMAT_VERSION: int = 1
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_as_of(value: object) -> pd.Timestamp:
    """``value`` (an ISO date or datetime string) as a UTC timestamp; a naive value is read as UTC.

    Raises ``ValueError`` for anything that is not a non-empty ISO string.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"as_of must be an ISO date or datetime string, got {value!r}")
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def check_test_from(value: object) -> str:
    """``value`` if it is a ``YYYY-MM-DD`` date string that parses; ``ValueError`` otherwise."""
    if not isinstance(value, str) or not _DATE.match(value):
        raise ValueError(f"test_from must be a date string like 2026-05-01, got {value!r}")
    pd.Timestamp(value)  # ValueError for an impossible date such as 2026-02-30
    return value


def read_dataset_meta(data: str | Path) -> dict | None:
    """The converter's ``dataset.json`` of the dataset directory ``data`` as a dict (dates, counts, ``coverage``;
    see ``emva.ingest.convert``), or None when the directory has no such file.

    Raises ``ValueError`` naming the file when it is not a JSON object, lacks ``as_of`` or ``test_from``, carries
    another ``FORMAT_KEY`` version, has a malformed date (``parse_as_of``, ``check_test_from``) or a ``test_from``
    after ``as_of``.
    """
    path = Path(data) / DATASET_META_FILE
    if not path.is_file():
        return None
    meta = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict) or not {"as_of", "test_from"} <= meta.keys():
        raise ValueError(f"{path} must be a JSON object with 'as_of' and 'test_from' (ADR 0020); the name "
                         "dataset.json is reserved for the converter's metadata")
    if meta.get(FORMAT_KEY, FORMAT_VERSION) != FORMAT_VERSION:
        raise ValueError(f"{path}: {FORMAT_KEY} {meta[FORMAT_KEY]!r}, expected {FORMAT_VERSION}")
    try:
        as_of, test_from = parse_as_of(meta["as_of"]), check_test_from(meta["test_from"])
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from e
    if pd.Timestamp(test_from, tz="UTC") > as_of:
        raise ValueError(f"{path}: test_from {test_from} is after as_of {as_of.isoformat()}")
    return meta


def dataset_dates(data: str | Path) -> tuple[pd.Timestamp, str]:
    """``(as_of, test_from)`` for the dataset directory ``data``: from its ``dataset.json`` (``read_dataset_meta``,
    which raises ``ValueError`` for a malformed file) when there is one, else ``(AS_OF, TEST_FROM)``."""
    meta = read_dataset_meta(data)
    if meta is None:
        return AS_OF, TEST_FROM
    return parse_as_of(meta["as_of"]), meta["test_from"]


def has_dataset_meta(data: str | Path) -> bool:
    """True when ``data`` carries the converter's ``dataset.json`` (its dates are its own, not the constants)."""
    return read_dataset_meta(data) is not None


__all__ = ["DATASET_META_FILE", "FORMAT_KEY", "FORMAT_VERSION", "check_test_from", "dataset_dates",
           "has_dataset_meta", "parse_as_of", "read_dataset_meta"]
