"""Keel entrypoint: ``streamlit run app/main.py``. Password gate, sidebar identity and page navigation.

Environment: ``APP_PASSWORD`` (required; without it the app shows a refusal screen and never opens) and
``DATA_DIR`` (data root, default ``runs/app``; created on first start with ``datasets/``, ``runs/`` and
``registry.json``). Pages live in ``app/views/``.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # streamlit puts app/ on sys.path, not the repo root that holds app/ and emva/
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402

from app import auth, theme, ui  # noqa: E402
from app import components as C  # noqa: E402

AUTH_KEY = "authenticated"
# Seconds to wait after a wrong password (slows guessing; the gate is a shared password, not per-user auth).
FAILED_LOGIN_DELAY_S = 1.0


def _unset_screen() -> None:
    """Refuse to open: the password is not configured."""
    ui.html('<div class="k-login">' + C.wordmark(theme.APP_NAME, theme.APP_TAGLINE)
            + C.page_header("Not configured", "APP_PASSWORD is not set",
                            "This tool only opens behind a password. Set the APP_PASSWORD environment variable "
                            "on the service (Railway: Variables) or in your shell, then restart the app.")
            + "</div>")


def _login(expected: str) -> None:
    """The sign-in screen; sets the session flag on the right password."""
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        ui.html('<div class="k-login" style="margin-bottom:8px">' + C.wordmark(theme.APP_NAME, theme.APP_TAGLINE)
                + C.page_header("Internal tool", "Sign in",
                                "Model results, training on your CRM exports and single-lead scoring.") + "</div>")
        with st.form("login", border=False):
            given = st.text_input("Password", type="password", key="password")
            ok = st.form_submit_button("Sign in", type="primary", width="stretch")
        if ok:
            if auth.check_password(given, expected):
                st.session_state[AUTH_KEY] = True
                st.rerun()
            time.sleep(FAILED_LOGIN_DELAY_S)
            ui.html(C.callout("That password is not right. Ask the EMVA team for the current one.", "bad"))
        ui.html('<div class="k-footnote">Shared password; every model here is trained on the data you upload. '
                "Numbers from the bundled sample are on simulated data.</div>")


def main() -> None:
    """Configure the page, run the gate, then the navigation."""
    st.set_page_config(page_title=f"{theme.APP_NAME} · EMVA", page_icon=":material/sailing:", layout="wide")
    theme.inject_css()
    expected = auth.expected_password()
    if expected is None:
        _unset_screen()
        st.stop()
    if not st.session_state.get(AUTH_KEY):
        _login(expected)
        st.stop()

    ui.data_root()
    pages = [
        st.Page("views/results.py", title="Model results", icon=":material/insights:", url_path="results",
                default=True),
        st.Page("views/upload.py", title="Upload & train", icon=":material/upload_file:", url_path="train"),
        st.Page("views/score.py", title="Score a lead", icon=":material/target:", url_path="score"),
    ]
    nav = st.navigation(pages, position="hidden")
    with st.sidebar:
        ui.html(C.wordmark(theme.APP_NAME, theme.APP_TAGLINE))
        for page in pages:
            st.page_link(page, label=page.title, icon=page.icon)
        st.space("medium")
        if st.button("Sign out", icon=":material/logout:", key="sign_out", type="tertiary"):
            st.session_state.clear()
            st.rerun()
        ui.html('<div class="k-footnote">Proof of concept. Figures from the bundled sample are on simulated data '
                "and are not to be quoted externally.</div>")
    nav.run()


main()
