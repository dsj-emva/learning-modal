"""Keel's page state in Streamlit's AppTest (the second Keel QA fix round): Map & convert work and the training
choices survive a page switch (M5), stale check results are withdrawn (m1, m10), a test-set switch never mixes two
test sets' numbers (m2), and a new starting lead clears the Score page's result (m7)."""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import uploads

from test_app_smoke import _app, _confirm_and_convert, _no_model_call, _sign_in, _text, _upload_raw


def _converted(monkeypatch: pytest.MonkeyPatch, root: Path) -> AppTest:
    """The upload page with the Olist-shaped raw files, the olist_funnel mapping loaded, its name edited, the
    status-quo rules uploaded, and the conversion done."""
    from conftest import REPO, olist_raw

    _no_model_call(monkeypatch)
    at = _sign_in(_app(monkeypatch, root), "letmein")
    at.switch_page("views/upload.py").run()
    at = _upload_raw(at, olist_raw())
    at.selectbox(key="mc_choice").select("builtin:olist_funnel").run()
    at.button(key="mc_load").click().run()
    at.text_input(key=f"mc_name_{at.session_state['mc_version']}").input("olist_edited").run()
    rules = (REPO / "data" / "v1" / "status_quo_rules.json").read_bytes()
    at.file_uploader(key="mc_rules").upload("status_quo_rules.json", rules, "application/json")
    at = _confirm_and_convert(at.run())
    assert not at.exception, at.exception
    assert "Check results" in _text(at) and "Save as a dataset" in _text(at)
    return at


def test_map_and_convert_and_training_choices_survive_a_page_switch(monkeypatch: pytest.MonkeyPatch,
                                                                     tmp_path: Path) -> None:
    """Keel QA M5: raw files, the edited mapping, its conversion and check results, the confirmations and the
    training options are all there after a visit to another page; saving drops every held file."""
    from conftest import OLIST_MQL

    at = _converted(monkeypatch, tmp_path)
    held = {"raw_primary", "raw_join_1", "mc_rules"}
    assert set(at.session_state[uploads.HELD_KEY]) == held
    assert not held & {u.key for u in at.file_uploader}  # held files replace their uploaders: no second copy
    at.segmented_control(key="label_mode").set_value("legacy").run()
    feature = next(s for s in at.segmented_control if s.key.startswith("feature_set_"))
    feature.set_value("legacy").run()
    at.switch_page("views/score.py").run()
    assert not at.exception, at.exception
    v = at.session_state["mc_version"]
    assert f"mc_name_{v}" not in at.session_state and "label_mode" not in at.session_state  # Streamlit dropped them
    at.switch_page("views/upload.py").run()
    assert not at.exception, at.exception
    page = _text(at)
    assert f"Primary file: using {OLIST_MQL}" in page and "Status-quo rules (optional)" in page
    assert "Check results" in page and "of 39 signals" in page and "Convert again" not in page
    assert at.text_input(key=f"mc_name_{at.session_state['mc_version']}").value == "olist_edited"
    assert all(c.value for c in at.checkbox if c.key and c.key.startswith(("mc_confirm_", "mc_features_confirm_")))
    assert at.segmented_control(key="label_mode").value == "legacy"
    assert next(s for s in at.segmented_control if s.key.startswith("feature_set_")).value == "legacy"
    at.text_input(key="dataset_name").input("olist-kept")
    next(b for b in at.button if b.label == "Save dataset").click()
    at.run()
    assert not at.exception, at.exception
    assert at.session_state[uploads.HELD_KEY] == {}  # saved: nothing held any more
    assert {u.key for u in at.file_uploader} >= {"raw_primary", "raw_join_1", "raw_join_2"}
    assert "Review the mapping" not in _text(at)


def test_replace_empties_a_held_slot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    at = _converted(monkeypatch, tmp_path)
    at.button(key="raw_primary_replace").click().run()
    assert not at.exception, at.exception
    assert uploads.held(at.session_state, "raw_primary") is None
    assert at.file_uploader(key="raw_primary") is not None
    assert "Review the mapping" not in _text(at) and "Check results" not in _text(at)  # the raw files changed


def test_any_mapping_change_withdraws_the_held_conversion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keel QA m10: a changed mapping (or an edit that cannot be applied) hides the conversion's check results until
    it is converted again; the mapping it was made from brings them back."""
    at = _converted(monkeypatch, tmp_path)
    v = at.session_state["mc_version"]
    at.text_input(key=f"mc_name_{v}").input("olist_other").run()
    page = _text(at)
    assert "Convert again before saving" in page and "Check results" not in page and "Save as a dataset" not in page
    at.text_input(key=f"mc_name_{v}").input("olist_edited").run()
    assert "Convert again" not in _text(at) and "Check results" in _text(at)
    at.session_state[f"mc_fields_{v}"] = {"edited_rows": {"0": {"target": "(value map)"}}, "added_rows": [],
                                          "deleted_rows": []}
    at.run()
    assert not at.exception, at.exception
    page = _text(at)
    assert "only a column with a value map" in page and "The mapping is not complete" in page
    assert "Convert again before saving" in page and "Check results" not in page
    assert not [b for b in at.button if b.key == "mc_convert"]


def test_removing_a_refused_upload_withdraws_its_check_results(monkeypatch: pytest.MonkeyPatch,
                                                               tmp_path: Path) -> None:
    """Keel QA m1: the check results of files that included a refused ground-truth upload are withdrawn when it is
    removed ("Validate again"), and cleared once no file is left."""
    from conftest import DATA_V1

    at = _sign_in(_app(monkeypatch, tmp_path), "letmein")
    at.switch_page("views/upload.py").run()
    at.file_uploader(key="up_historical_leads.csv").upload("historical_leads.csv",
                                                             (DATA_V1 / "historical_leads.csv").read_bytes(), "text/csv")
    at.file_uploader(key="up_crm_history.csv").upload("ground_truth_labels.csv", b"lead_id,y\nL1,1\n", "text/csv")
    at.run()
    assert "Refused ground_truth_labels.csv" in _text(at)
    at.button(key="validate").click().run()
    assert "problem(s) to fix" in _text(at) and "Refused upload" in _text(at)
    at.button(key="up_crm_history.csv_replace").click().run()
    page = _text(at)
    assert "Validate again before saving" in page and "problem(s) to fix" not in page and "Refused upload" not in page
    at.button(key="up_historical_leads.csv_replace").click().run()
    page = _text(at)
    assert "Validate again" not in page and "Check results" not in page


def _top_level(at: AppTest, needle: str) -> int:
    """The index of the main area's top-level element whose subtree holds a markdown containing ``needle``."""
    def texts(node) -> list[str]:
        own = [node.value] if getattr(node, "type", None) == "markdown" else []
        return own + [t for c in getattr(node, "children", {}).values() for t in texts(c)]
    return next(i for i, node in at.main.children.items() if any(needle in t for t in texts(node)))


def test_a_test_set_switch_replaces_every_test_set_section_at_once(monkeypatch: pytest.MonkeyPatch,
                                                                    app_generic) -> None:
    """Keel QA m2: the headline, the standard table, Generic vs v2 and the charts of a test set render into one slot,
    which is emptied before the new test set's numbers are computed; the scorecard (the same on both) is outside."""
    root, run = app_generic
    at = _sign_in(_app(monkeypatch, root), "letmein")
    at.segmented_control(key=f"test_set_{run.run_id}").set_value("legacy").run()
    assert not at.exception, at.exception
    slot = _top_level(at, "Headline")
    assert _top_level(at, "Generic vs v2") == _top_level(at, "Stability by month") == slot
    assert _top_level(at, "Scorecard") != slot
    assert "Legacy labels (frozen POC)" in _text(at) and "Mature leads (horizon labels)" not in _text(at)


def test_a_new_starting_lead_clears_the_score(monkeypatch: pytest.MonkeyPatch, app_trained) -> None:
    """Keel QA m7: after a score, an unknown "Start from a lead" id shows the example form without the old result,
    and going back to the example does not bring it back."""
    root, run = app_trained
    at = _sign_in(_app(monkeypatch, root), "letmein")
    at.switch_page("views/score.py").run()
    next(b for b in at.button if b.label == "Score this lead").click()
    at.run()
    assert "Chance this lead closes" in _text(at)
    at.text_input(key=f"prefill_{run.run_id}").input("NO-SUCH-LEAD").run()
    assert not at.exception, at.exception
    assert "No lead NO-SUCH-LEAD in this dataset." in _text(at) and "Chance this lead closes" not in _text(at)
    at.text_input(key=f"prefill_{run.run_id}").input("").run()
    assert "Chance this lead closes" not in _text(at) and "Ready when you are" in _text(at)
