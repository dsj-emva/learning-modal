"""app.uploads: the uploaded files the upload page holds across page switches (Keel QA M5). Plain dicts stand in for
the Streamlit session state."""
from __future__ import annotations

import hashlib

from app import uploads


def test_hold_keeps_one_copy_per_slot_and_replaces_it() -> None:
    state: dict = {}
    first = uploads.hold(state, "raw_primary", "leads.csv", b"a,b\n1,2\n")
    assert uploads.held(state, "raw_primary") == first and not first.refused
    assert first.sha256 == hashlib.sha256(b"a,b\n1,2\n").hexdigest() and first.size == "8 B"
    uploads.hold(state, "raw_primary", "other.csv", b"x\n")
    assert list(state[uploads.HELD_KEY]) == ["raw_primary"]  # replaced, not a history
    assert uploads.held(state, "raw_primary").name == "other.csv"
    assert uploads.held(state, "raw_join_1") is None


def test_a_ground_truth_upload_is_held_by_name_only() -> None:
    state: dict = {}
    f = uploads.hold(state, "up_crm_history.csv", "Ground_Truth_labels.csv", b"lead_id,y\nL1,1\n")
    assert f.refused and f.data is None and f.sha256 == "" and f.size == ""
    assert uploads.is_refused_name("ground_truth.md") and not uploads.is_refused_name("leads.csv")


def test_release_drops_the_named_slots_only() -> None:
    state: dict = {}
    for slot in ("raw_primary", "raw_join_1", "mc_rules"):
        uploads.hold(state, slot, f"{slot}.csv", b"x\n")
    uploads.release(state, ["raw_primary", "mc_rules", "never_held"])
    assert list(state[uploads.HELD_KEY]) == ["raw_join_1"]
    uploads.release({}, ["raw_primary"])  # nothing held yet: fine


def test_sizes_read_as_bytes_kilobytes_and_megabytes() -> None:
    assert uploads.HeldFile("a", b"x" * 1023, "").size == "1023 B"
    assert uploads.HeldFile("a", b"x" * 1536, "").size == "1.5 KB"
    assert uploads.HeldFile("a", b"x" * (3 * 1024 ** 2), "").size == "3.0 MB"
