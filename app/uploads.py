"""Uploaded files held in the session across page switches (Keel QA M5). No Streamlit.

Streamlit drops a file uploader's file when the person opens another page (the state of a widget that is not rendered
is deleted), so the upload page keeps each slot's file here instead: ``hold`` when a file arrives (replacing the slot's
previous file), ``release`` on Replace, when the files are saved or when the page no longer needs them. Exactly one
copy per slot, never a history. A ground-truth-looking upload is held by its name only, never its bytes, so it can be
refused by name without keeping its content.

``state`` is the Streamlit session state (any mutable mapping); the held files live under ``HELD_KEY``.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, MutableMapping
from dataclasses import dataclass

from app.storage import FORBIDDEN_PREFIX

HELD_KEY: str = "held_uploads"


@dataclass(frozen=True)
class HeldFile:
    """One slot's uploaded file: its name, its bytes (None when refused by name) and their SHA-256 ("" if refused)."""

    name: str
    data: bytes | None
    sha256: str

    @property
    def refused(self) -> bool:
        """True for a ground-truth-looking file (``app.storage.FORBIDDEN_PREFIX``), held without its bytes."""
        return self.data is None

    @property
    def size(self) -> str:
        """The size as short text (``"812 B"``, ``"1.4 MB"``); "" when refused."""
        if self.data is None:
            return ""
        n = len(self.data)
        if n < 1024:
            return f"{n} B"
        return f"{n / 1024:.1f} KB" if n < 1024 ** 2 else f"{n / 1024 ** 2:.1f} MB"


def is_refused_name(name: str) -> bool:
    """True when an upload named ``name`` is refused by name (evaluation-only ground-truth files)."""
    return name.lower().startswith(FORBIDDEN_PREFIX)


def hold(state: MutableMapping, slot: str, name: str, data: bytes) -> HeldFile:
    """Hold ``data`` (uploaded as ``name``) for ``slot``, replacing what the slot held; returns the held file."""
    f = HeldFile(name, None, "") if is_refused_name(name) else HeldFile(name, data, hashlib.sha256(data).hexdigest())
    state.setdefault(HELD_KEY, {})[slot] = f
    return f


def held(state: MutableMapping, slot: str) -> HeldFile | None:
    """The file ``slot`` holds, or None."""
    return state.get(HELD_KEY, {}).get(slot)


def release(state: MutableMapping, slots: Iterable[str]) -> None:
    """Drop the files of ``slots`` (a slot holding nothing is fine)."""
    files = state.get(HELD_KEY, {})
    for s in slots:
        files.pop(s, None)


__all__ = ["HELD_KEY", "HeldFile", "hold", "held", "is_refused_name", "release"]
