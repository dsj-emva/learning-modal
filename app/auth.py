"""The shared-password gate as pure functions (``app.main`` renders it). No Streamlit.

The password comes from the environment variable ``APP_PASSWORD``. When it is unset or blank the app refuses to
open at all (it never falls back to "no password"). Comparison is constant-time (``hmac.compare_digest``).
"""
from __future__ import annotations

import hmac
import os
from collections.abc import Mapping

PASSWORD_ENV: str = "APP_PASSWORD"


def expected_password(env: Mapping[str, str] | None = None) -> str | None:
    """``APP_PASSWORD`` from ``env`` (default ``os.environ``); None when unset or blank."""
    value = (os.environ if env is None else env).get(PASSWORD_ENV, "")
    return value if value.strip() else None


def check_password(given: str | None, expected: str | None) -> bool:
    """True only when ``expected`` is set and ``given`` equals it (constant-time, UTF-8 bytes)."""
    if not expected or given is None:
        return False
    return hmac.compare_digest(given.encode("utf-8"), expected.encode("utf-8"))


__all__ = ["PASSWORD_ENV", "check_password", "expected_password"]
