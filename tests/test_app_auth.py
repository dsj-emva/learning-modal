"""app.auth: the shared-password check and the refusal when APP_PASSWORD is unset."""
from __future__ import annotations

from app.auth import check_password, expected_password


def test_check_password() -> None:
    assert check_password("s3cret-é", "s3cret-é")
    assert not check_password("wrong", "s3cret")
    assert not check_password("", "s3cret") and not check_password(None, "s3cret")
    assert not check_password("", "") and not check_password("x", None)


def test_unset_password_refuses() -> None:
    assert expected_password({}) is None
    assert expected_password({"APP_PASSWORD": "   "}) is None
    assert expected_password({"APP_PASSWORD": "pw"}) == "pw"
