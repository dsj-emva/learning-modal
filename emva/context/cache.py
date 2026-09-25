"""On-disk cache of context-agent replies, keyed by (card hash, brief hash, prompt version, prompt fingerprint,
model id).

The cache is a JSON file committed next to the outputs (``data/v2/context/cache.json``) so a rerun
with the same brief, prompt and model reproduces every judgment without the API (ADR 0010). Any
change to the brief, the prompt version or the model id changes the key, so old replies are never
reused silently. The prompt fingerprint (``emva.context.agent.prompt_fingerprint``: system template,
response schema, max tokens) catches a prompt edit made without a version bump. Successful replies and
parse errors are cached (both are what the model said); transport errors never are. Writes are atomic (temp file ``<name>.tmp`` + rename; ``data/**/*.tmp`` is
gitignored) and deterministic (sorted keys). Format 2 added the fingerprint to the key; format 1 files are
refused (the committed Phase 6 cache was re-keyed in place, see ``reports/phase6.md``).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

CACHE_FORMAT = 2


def cache_key(card_hash: str, brief_hash: str, prompt_version: str, prompt_fingerprint: str, model_id: str) -> str:
    """sha256 hex of the five identifiers, JSON-encoded as a list."""
    ids = [card_hash, brief_hash, prompt_version, prompt_fingerprint, model_id]
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


class ReplyCache:
    """Thread-safe dict of cache entries backed by a JSON file.

    Each entry is ``{"card_hash", "brief_hash", "prompt_version", "prompt_fingerprint", "model_id", "status", "reply",
    "raw", "error"}``:
    ``status`` is ``ok`` or ``parse_error``, ``reply`` the validated judgments (``ok``) or None, ``raw``
    the model's text and ``error`` the contract violation (parse errors; None and "" otherwise).
    """

    def __init__(self, path: str | Path) -> None:
        """Load ``path`` if it exists (empty otherwise); raise ``ValueError`` on an unknown format."""
        self.path = Path(path)
        self._lock = threading.Lock()
        self.entries: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("format") != CACHE_FORMAT:
                raise ValueError(f"{self.path}: cache format {data.get('format')!r}, expected {CACHE_FORMAT}")
            self.entries = data["entries"]

    def get(self, key: str) -> dict[str, Any] | None:
        """The entry for ``key`` or None."""
        with self._lock:
            return self.entries.get(key)

    def put(self, key: str, entry: dict[str, Any]) -> None:
        """Store ``entry`` under ``key`` and rewrite the file (so an interrupted run loses nothing)."""
        with self._lock:
            self.entries[key] = entry
            self._save()

    def brief_hashes(self) -> set[str]:
        """Every brief hash that has at least one entry."""
        with self._lock:
            return {e["brief_hash"] for e in self.entries.values()}

    def _save(self) -> None:
        """Write the file atomically; the caller holds the lock."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"format": CACHE_FORMAT, "entries": self.entries}, f, indent=1, ensure_ascii=False, sort_keys=True)
            f.write("\n")
        os.replace(tmp, self.path)
