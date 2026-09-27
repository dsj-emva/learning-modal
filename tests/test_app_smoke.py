"""Keel end to end in Streamlit's AppTest: the gate refuses without APP_PASSWORD, blocks a wrong password, and
every page renders without an exception once signed in (on a data root holding the shared trained run)."""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import components as C
from app import scoring
from app.validation import Issue
from emva.persist import load_bundle

MAIN = str(Path(__file__).resolve().parents[1] / "app" / "main.py")
PAGES = ("views/results.py", "views/upload.py", "views/score.py")


def _app(monkeypatch: pytest.MonkeyPatch, root: Path, password: str | None = "letmein") -> AppTest:
    """An AppTest of app/main.py with ``DATA_DIR`` = ``root`` and ``APP_PASSWORD`` set (or unset)."""
    monkeypatch.setenv("DATA_DIR", str(root))
    if password is None:
        monkeypatch.delenv("APP_PASSWORD", raising=False)
    else:
        monkeypatch.setenv("APP_PASSWORD", password)
    return AppTest.from_file(MAIN, default_timeout=90)


def _text(at: AppTest) -> str:
    """All markdown on the page, joined."""
    return "\n".join(m.value for m in at.markdown)


def _sign_in(at: AppTest, password: str) -> AppTest:
    at.run()
    at.text_input(key="password").input(password)
    at.button[0].click()
    return at.run()


def test_unset_password_refuses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    at = _app(monkeypatch, tmp_path, password=None).run()
    assert not at.exception
    assert "APP_PASSWORD is not set" in _text(at)
    assert not at.text_input  # no way in


def test_wrong_password_blocked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    at = _sign_in(_app(monkeypatch, tmp_path), "nope")
    assert not at.exception
    assert "not right" in _text(at) and "authenticated" not in at.session_state
    assert "How well the model ranks leads" not in _text(at)


def test_empty_root_shows_empty_states(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    at = _sign_in(_app(monkeypatch, tmp_path / "fresh"), "letmein")
    assert not at.exception and "Nothing trained yet" in _text(at)
    assert (tmp_path / "fresh" / "registry.json").exists()  # layout created on first start


def test_every_page_renders_with_a_trained_run(monkeypatch: pytest.MonkeyPatch, app_trained) -> None:
    root, run = app_trained
    at = _sign_in(_app(monkeypatch, root), "letmein")
    assert not at.exception, at.exception
    assert "How well the model ranks leads" in _text(at) and "Standard table" in _text(at)
    for page in PAGES:
        at.switch_page(page).run()
        assert not at.exception, (page, at.exception)
    at.switch_page("views/score.py").run()
    next(b for b in at.button if b.label == "Score this lead").click()
    at.run()
    assert not at.exception, at.exception
    assert "Chance this lead closes" in _text(at) and "Why this score" in _text(at)


def _stats(at: AppTest) -> str:
    """The value-per-lead stats rows on the results page."""
    return "\n".join(m.value for m in at.markdown if "k-stats" in m.value)


def _captions(at: AppTest) -> str:
    return "\n".join(c.value for c in at.caption)


def test_results_page_amounts_and_constant_signals(monkeypatch: pytest.MonkeyPatch, app_trained,
                                                    app_converted) -> None:
    """The sample shows £ and no constant-signal note; the converted Olist fixture (currency BRL) shows BRL, never £,
    and says which signals are the same on every training lead (M3, M4)."""
    import numpy as np

    from app import charts

    at = _sign_in(_app(monkeypatch, app_trained[0]), "letmein")
    assert not at.exception and "£" in _stats(at) and "same on every training lead" not in _captions(at)
    at = _sign_in(_app(monkeypatch, app_converted[0]), "letmein")
    assert not at.exception, at.exception
    assert "BRL " in _stats(at) and "£" not in _stats(at)
    assert "same on every training lead" in _captions(at) and "source's currency" not in _captions(at)
    ticks = charts.value_histogram({"v": np.array([5.0, 500.0])}, "BRL").layout.xaxis.ticktext
    assert ticks[0] == "BRL 1" and all("£" not in t for t in ticks)
    assert charts.value_histogram({"v": np.array([5.0])}, None).layout.xaxis.ticktext[0] == "1"


def test_sign_out(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    at = _sign_in(_app(monkeypatch, tmp_path), "letmein")
    at.button(key="sign_out").click()
    at.run()
    assert "Sign in" in _text(at)


def test_components_escape_user_text() -> None:
    html = C.file_card("<x>.csv", "Leads", [Issue("f", "<col>", "<script>alert(1)</script>", (1, 2))], [], True)
    assert "<script>" not in html and "&lt;script&gt;" in html and "&lt;col&gt;" in html
    assert C.points(-7.4) == "−7" and C.points(3) == "+3" and C.pct(0.2345) == "23.4%"


def test_logo_is_the_glyph_then_the_escaped_name() -> None:
    """st.logo's image (glyph square in the accent, then the name) and its collapsed-sidebar icon (the square)."""
    from app import theme

    logo, icon = C.logo_svg(theme.PALETTE, "<Keel>"), C.logo_svg(theme.PALETTE)
    assert logo.startswith("<svg") and theme.PALETTE["accent"] in logo and "&lt;Keel&gt;</text>" in logo
    assert icon.startswith("<svg") and theme.PALETTE["accent"] in icon and "<text" not in icon


def test_money_formats_by_the_dataset_currency() -> None:
    """GBP (the sample, any dataset without dataset.json) as £; another code before the amount; unknown as a plain
    number with a note for the page."""
    assert C.money(12345.6) == C.money(12345.6, "GBP") == "£12,346" and C.money(None) == "–"
    assert C.money(41106, "BRL") == "BRL 41,106" and C.money(41106, None) == "41,106"
    assert (C.money_format("GBP"), C.money_format("BRL"), C.money_format(None)) == ("£%.0f", "BRL %.0f", "%.0f")
    assert C.currency_note("GBP") == C.currency_note("BRL") == "" and "source's currency" in C.currency_note(None)
    card = C.result_card(0.2, 0.1, 41106, 900, 800, "t", "", "BRL")
    assert "BRL 41,106" in card and "£" not in card


def test_run_error_and_names_render_escaped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A failed run whose error (and header detail) carries markup is shown as text on the results page."""
    from app import storage

    evil = "<script>alert(1)</script>"
    run = storage.register_run(tmp_path, storage.new_run(tmp_path, storage.sample_dataset(), {"label_mode": evil}))
    storage.update_run(tmp_path, run.run_id, status="failed", error=evil)
    at = _sign_in(_app(monkeypatch, tmp_path), "letmein")
    assert not at.exception
    page = _text(at)
    assert "<script>" not in page and page.count("&lt;script&gt;alert(1)&lt;/script&gt;") >= 2
    assert "<script>" not in C.run_header("failed", evil, f"dataset {evil}") + C.callout(evil, "bad", lead=evil)


def test_score_page_prefills_a_dataset_lead(monkeypatch: pytest.MonkeyPatch, app_trained) -> None:
    """The prefill (``scoring.values_from_lead``) fills the form with a real lead, and it scores."""
    root, run = app_trained
    import pandas as pd

    lead_id = pd.read_csv(Path(run.out_dir) / "scores.csv", usecols=["lead_id"]).lead_id.iloc[0]
    at = _sign_in(_app(monkeypatch, root), "letmein")
    at.switch_page("views/score.py").run()
    at.text_input(key=f"prefill_{run.run_id}").input(lead_id).run()
    assert not at.exception
    next(b for b in at.button if b.label == "Score this lead").click()
    at.run()
    assert not at.exception and "Chance this lead closes" in _text(at)


def test_score_page_legacy_run_on_converted_data_starts_from_a_training_lead(monkeypatch: pytest.MonkeyPatch,
                                                                             app_converted_legacy) -> None:
    """Keel QA M1: the legacy design has not seen the built-in example's levels, so the EMVA form starts from a
    training lead, says so, and that lead scores."""
    root, run = app_converted_legacy
    at = _sign_in(_app(monkeypatch, root), "letmein")
    at.switch_page("views/score.py").run()
    assert not at.exception, at.exception
    assert "Example taken from training lead " in _text(at)
    next(b for b in at.button if b.label == "Score this lead").click()
    at.run()
    assert not at.exception, at.exception
    assert "Chance this lead closes" in _text(at) and "not seen" not in _text(at)


# --- Map & convert and the source-format score page (Phase 9) --------------------------------------------------------

def _upload_raw(at: AppTest, files: dict[str, bytes]) -> AppTest:
    """Put ``files`` (primary first) into the Map & convert slots and rerun."""
    for (name, content), key in zip(files.items(), ("raw_primary", "raw_join_1", "raw_join_2")):
        at.file_uploader(key=key).upload(name, content, "text/csv")
    return at.run()


def _no_model_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if the page tries to build an Anthropic client (every path here must work without one)."""
    from app import ingest

    def refuse() -> None:
        raise AssertionError("the app must not call the model in this test")
    monkeypatch.setattr(ingest, "make_client", refuse)


def _confirm_and_convert(at: AppTest) -> AppTest:
    """Tick the outcome confirmation and, when the mapping declares extras, the features one; then convert."""
    next(c for c in at.checkbox if c.key and c.key.startswith("mc_confirm_")).check().run()
    for c in [c for c in at.checkbox if c.key and c.key.startswith("mc_features_confirm_")]:
        c.check().run()
    at.button(key="mc_convert").click()
    return at.run()


def test_map_convert_validate_save_with_a_saved_mapping(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Raw Olist-shaped CSVs -> the committed hand-written mapping -> confirm -> convert -> check -> save, no model."""
    from app import ingest, storage
    from conftest import REPO, olist_raw

    _no_model_call(monkeypatch)
    at = _sign_in(_app(monkeypatch, tmp_path), "letmein")
    at.switch_page("views/upload.py").run()
    assert at.selectbox(key="train_ds").value == "sample-v1"  # the picker holds its own state from here on
    at = _upload_raw(at, olist_raw())
    assert not at.exception, at.exception
    assert "Column profile" in _text(at) and "Start the mapping" in _text(at)
    at.selectbox(key="mc_choice").select("builtin:olist_funnel").run()
    at.button(key="mc_load").click().run()
    assert not at.exception, at.exception
    assert "Loaded olist_funnel" in _text(at) and "Review the mapping" in _text(at)
    assert at.button(key="mc_convert").disabled  # not before the outcome mapping is confirmed
    rules = (REPO / "data" / "v1" / "status_quo_rules.json").read_bytes()
    at.file_uploader(key="mc_rules").upload("status_quo_rules.json", rules, "application/json")
    at = _confirm_and_convert(at.run())
    assert not at.exception, at.exception
    page = _text(at)
    assert "fills " in page and " of 39 signals" in page and "Check results" in page and "Save as a dataset" in page
    assert "Derived during conversion" in page and "Lost leads without close_at" in "\n".join(t.value for t in at.text)
    assert "Filled for won leads" not in page  # the committed mapping ignores every won-only column
    at.text_input(key="dataset_name").input("olist-smoke")
    next(b for b in at.button if b.label == "Save dataset").click()
    at.run()
    assert not at.exception, at.exception
    saved = Path(storage.get_dataset(tmp_path, "olist-smoke").path)
    assert {"mapping.toml", "dataset.json", "historical_leads.csv"} <= {p.name for p in saved.iterdir()}
    assert (saved / "status_quo_rules.json").read_bytes() == rules
    assert storage.get_dataset(tmp_path, "olist-smoke").has_rules
    assert ingest.mapping_choices(tmp_path)[0].key == "saved:olist-smoke"  # offered for the next export
    assert at.selectbox(key="train_ds").value == "olist-smoke"  # the train picker moves to the saved dataset


def test_fill_by_hand_then_apply_a_hand_written_toml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from conftest import REPO, olist_raw

    _no_model_call(monkeypatch)
    at = _sign_in(_app(monkeypatch, tmp_path), "letmein")
    at.switch_page("views/upload.py").run()
    at = _upload_raw(at, olist_raw())
    at.button(key="mc_blank").click().run()
    assert not at.exception, at.exception
    assert "Empty mapping" in _text(at) and "The mapping is not complete" in _text(at)
    at.text_area[0].input("name = ").run()
    at.button(key="mc_toml_apply").click().run()
    assert "The TOML was not applied" in _text(at) and "not valid TOML" in _text(at)
    at.text_area[0].input((REPO / "mappings" / "olist_funnel.toml").read_text(encoding="utf-8")).run()
    at.button(key="mc_toml_apply").click().run()
    assert not at.exception, at.exception
    assert "TOML applied" in _text(at)
    at = _confirm_and_convert(at)
    assert not at.exception and "of 39 signals" in _text(at)


def test_a_won_only_column_mapped_to_a_field_is_flagged_after_convert(monkeypatch: pytest.MonkeyPatch,
                                                                      tmp_path: Path) -> None:
    """Keel fix round (M6): business_segment (closed deals only) mapped to answers.what_to_solve converts, with an
    amber warning in the coverage card and a note on historical_leads.csv in Check results."""
    from conftest import REPO, olist_raw

    _no_model_call(monkeypatch)
    at = _sign_in(_app(monkeypatch, tmp_path), "letmein")
    at.switch_page("views/upload.py").run()
    at = _upload_raw(at, olist_raw())
    at.button(key="mc_blank").click().run()
    text = (REPO / "mappings" / "olist_funnel.toml").read_text(encoding="utf-8")
    leaky = text.replace('source = "deal.business_segment"\ntarget = "ignore"',
                         'source = "deal.business_segment"\ntarget = "answers.what_to_solve"')
    assert leaky != text
    at.text_area[0].input(leaky).run()
    at.button(key="mc_toml_apply").click().run()
    at = _confirm_and_convert(at)
    assert not at.exception, at.exception
    page = _text(at)
    assert page.count("Filled for won leads but not for lost ones:") == 2  # the coverage card and the file note
    assert "deal.business_segment -&gt; answers.what_to_solve (won 100%, lost 0%)" in page
    assert "Save as a dataset" in page  # a warning, not a block


def test_draft_button_uses_a_cached_draft(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A cached draft (put there by a fake client with a hand-built reply) loads with no client at all."""
    from app import ingest
    from conftest import olist_raw
    from test_app_ingest import FakeClient, hand_built_reply, reply_message

    files = olist_raw()
    profiles = ingest.profile(ingest.raw_frames(files))
    root = tmp_path / "root"
    assert ingest.draft(profiles, root, "seed", lambda: FakeClient(reply_message(hand_built_reply()))).mapping
    _no_model_call(monkeypatch)
    at = _sign_in(_app(monkeypatch, root), "letmein")
    at.switch_page("views/upload.py").run()
    at = _upload_raw(at, files)
    assert "A cached draft exists" in "\n".join(c.value for c in at.caption)
    at.button(key="mc_draft").click().run()
    assert not at.exception, at.exception
    page = _text(at)
    assert "from the draft cache" in page and "Low confidence, check these rows" in page
    assert "high confidence · hand-built test reply" in [t.value for t in at.text]  # plain text, not markdown


def test_raw_ground_truth_upload_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    at = _sign_in(_app(monkeypatch, tmp_path), "letmein")
    at.switch_page("views/upload.py").run()
    at = _upload_raw(at, {"ground_truth_labels.csv": b"lead_id,y\nL1,1\n"})
    assert not at.exception and "Refused ground_truth_labels.csv" in _text(at)
    assert "Column profile" not in _text(at)


def test_score_page_source_format(monkeypatch: pytest.MonkeyPatch, app_converted) -> None:
    """Score one lead in the source format, then a CSV of them, then a refused value (shown escaped)."""
    from conftest import INGEST_FIXTURES, OLIST_MQL

    root, run = app_converted
    at = _sign_in(_app(monkeypatch, root), "letmein")
    at.switch_page("views/score.py").run()
    at.segmented_control(key=f"score_mode_{run.run_id}").set_value("source").run()
    assert not at.exception, at.exception
    assert [t.label for t in at.text_input if t.key.startswith("src_")] == ["mql_id", "first_contact_date *",
                                                                            "landing_page_id"]
    origin = next(s for s in at.selectbox if s.key.endswith(":origin"))
    origin.select("paid_search")
    next(b for b in at.button if b.label == "Score this lead").click()
    at.run()
    assert not at.exception and "created_at is blank" in _text(at)  # the date is required (Keel QA M2)
    assert "Chance this lead closes" not in _text(at)
    next(t for t in at.text_input if t.key.endswith(":first_contact_date")).input("2018-04-01")
    next(b for b in at.button if b.label == "Score this lead").click()
    at.run()
    assert not at.exception, at.exception
    assert "Chance this lead closes" in _text(at) and "Why this score" in _text(at)
    card = next(m.value for m in at.markdown if m.value.startswith("<div class=\"k-result\""))
    assert "BRL " in card and "£" not in card  # the mapping declares BRL (M4)
    assert "same on every training lead" in _text(at)  # the constant signals are in the starting score (M3)
    raw = (INGEST_FIXTURES / "olist_funnel" / OLIST_MQL).read_bytes()
    at.file_uploader(key=f"src_upload_{run.run_id}").upload("new.csv", raw, "text/csv").run()
    at.button(key=f"src_score_file_{run.run_id}").click().run()
    assert not at.exception, at.exception
    assert at.selectbox(key=f"src_pick_{run.run_id}").options and "Chance this lead closes" in _text(at)
    bad = raw.replace(b"paid_search", b"<b>pigeon</b>", 1)
    at.file_uploader(key=f"src_upload_{run.run_id}").clear().upload("new.csv", bad, "text/csv").run()
    at.button(key=f"src_score_file_{run.run_id}").click().run()
    page = _text(at)
    assert not at.exception and "These leads cannot be scored" in page
    assert "<b>pigeon</b>" not in page and "&lt;b&gt;pigeon&lt;/b&gt;" in page


# --- the generic feature set in Keel (Phase 10 (b)) ------------------------------------------------------------------

def test_map_with_extras_confirm_both_convert_save_train_generic_and_see_the_comparison(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Raw Olist-shaped CSVs -> the committed mapping (one extra, landing_page) -> both confirmations -> convert ->
    check -> save -> train with generic -> the results page shows Generic vs v2. No model call."""
    import time

    from app import storage
    from app.views.upload import ACTIVE_KEY, PROCS_KEY
    from conftest import olist_raw

    _no_model_call(monkeypatch)
    at = _sign_in(_app(monkeypatch, tmp_path), "letmein")
    at.switch_page("views/upload.py").run()
    at = _upload_raw(at, olist_raw())
    at.selectbox(key="mc_choice").select("builtin:olist_funnel").run()
    at.button(key="mc_load").click().run()
    assert not at.exception, at.exception
    assert "Fields and extra features" in _text(at)
    feat = [c for c in at.checkbox if c.key and c.key.startswith("mc_features_confirm_")]
    assert len(feat) == 1 and not feat[0].value  # never pre-ticked, even for a mapping confirmed on disk
    assert feat[0].label == "Extra features are known when the lead is submitted"
    next(c for c in at.checkbox if c.key and c.key.startswith("mc_confirm_")).check().run()
    assert at.button(key="mc_convert").disabled  # the outcome alone is not enough with declared extras
    next(c for c in at.checkbox if c.key and c.key.startswith("mc_features_confirm_")).check().run()
    assert not at.button(key="mc_convert").disabled
    at.button(key="mc_convert").click()
    at.run()
    assert not at.exception, at.exception
    page = _text(at)
    assert "Extra features" in page and "landing_page. How many design columns" in page
    assert "Extra features (generic feature set)" in page  # its own file card in the check results
    at.text_input(key="dataset_name").input("olist-generic")
    next(b for b in at.button if b.label == "Save dataset").click()
    at.run()
    assert not at.exception, at.exception
    ds = storage.get_dataset(tmp_path, "olist-generic")
    assert ds.has_extras and storage.EXTRA_FEATURES_FILE in ds.rows
    control = at.segmented_control(key="feature_set_3")
    assert control.options == ["v2", "Generic (v2 + extras)", "Legacy (POC)"] and control.value == "v2"  # not default
    control.set_value("generic").run()
    assert not at.exception, at.exception
    assert "Extra features" in _text(at) and "1 (0 numeric)" in _text(at)  # the pre-training summary
    at.button(key="train").click().run()
    run_id = at.session_state[ACTIVE_KEY]
    assert at.session_state[PROCS_KEY][run_id].wait(timeout=240) == 0
    for _ in range(40):  # the job records the outcome right after the report
        if storage.get_run(tmp_path, run_id).is_done:
            break
        time.sleep(0.5)
    run = storage.get_run(tmp_path, run_id)
    assert run.status == "succeeded" and run.args["feature_set"] == "generic" and run.has_report, run.error
    at.switch_page("views/results.py").run()
    assert not at.exception, at.exception
    page = _text(at)
    assert "Generic vs v2" in page and "Leakage screen" in page and "landing_page" in page
    assert "Missing: closed" in page and "Missingness flag" in page
    assert "Collinearity of the generic design" in page and "extra columns" in page


def test_generic_run_score_page_offers_the_extras(monkeypatch: pytest.MonkeyPatch, app_generic) -> None:
    """Source format: the extra's training levels are offered; EMVA form: a note that extras are blank, and a warning
    per extra that no training lead had missing (it scores as the reference level)."""
    root, run = app_generic
    at = _sign_in(_app(monkeypatch, root), "letmein")
    at.switch_page("views/score.py").run()
    assert not at.exception, at.exception
    assert "Extra features are left blank." in _text(at) and "landing_page" in _text(at)
    assert "reference level wherever the model never saw a missing value" in _text(at)
    unseen = scoring.missing_unseen(load_bundle(Path(run.out_dir) / "model.joblib"))
    assert all(f"Blank {name} scores as {ref}." in _text(at) for name, ref in unseen.items())
    at.segmented_control(key=f"score_mode_{run.run_id}").set_value("source").run()
    assert not at.exception, at.exception
    page = next(s for s in at.selectbox if s.key.endswith(":landing_page_id"))
    assert page.options[0] == "— blank —" and page.options[-1] == "other (any value not listed)"
    page.select_index(1)
    next(t for t in at.text_input if t.key.endswith(":first_contact_date")).input("2018-04-01")
    next(b for b in at.button if b.label == "Score this lead").click()
    at.run()
    assert not at.exception, at.exception
    assert "Chance this lead closes" in _text(at)
