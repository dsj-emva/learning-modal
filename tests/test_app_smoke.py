"""Keel end to end in Streamlit's AppTest: the gate refuses without APP_PASSWORD, blocks a wrong password, and
every page renders without an exception once signed in (on a data root holding the shared trained run)."""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import components as C
from app.validation import Issue

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


def test_sign_out(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    at = _sign_in(_app(monkeypatch, tmp_path), "letmein")
    at.button(key="sign_out").click()
    at.run()
    assert "Sign in" in _text(at)


def test_components_escape_user_text() -> None:
    html = C.file_card("<x>.csv", "Leads", [Issue("f", "<col>", "<script>alert(1)</script>", (1, 2))], [], True)
    assert "<script>" not in html and "&lt;script&gt;" in html and "&lt;col&gt;" in html
    assert C.points(-7.4) == "−7" and C.points(3) == "+3" and C.gbp(12345.6) == "£12,346" and C.pct(0.2345) == "23.4%"


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
