"""Unit tests for auth.py's error-handling around Zerodha's login/2FA steps.

Mocks requests.Session.post directly rather than hitting the real endpoints -
this module's whole job is turning Zerodha's raw HTTP responses into clear
AuthErrors, so what matters is verifying that translation, not real network
behavior (already exercised live, by definition, every time login() runs)."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from auth import AuthError, _fetch_request_token

FAKE_TOTP_SECRET = "JBSWY3DPEHPK3PXP"  # valid base32, not a real account secret


def _mock_response(status_code: int, json_body: dict):
    resp = Mock()
    resp.status_code = status_code
    resp.json.return_value = json_body
    return resp


def _login_success_response():
    return _mock_response(200, {"status": "success", "data": {"request_id": "req123"}})


def test_twofa_failure_raises_clear_autherror_not_a_raw_httperror():
    # Confirmed live 2026-08-14: Zerodha's /api/twofa returned a real 400 with
    # a JSON body, but the old code called raise_for_status() blindly first,
    # turning it into a generic, unhelpful HTTPError that crashed the bot at
    # startup. Must raise AuthError with the real message instead.
    twofa_failure = _mock_response(400, {"status": "error", "message": "Invalid TOTP. Please try again."})

    with patch("requests.Session.post", side_effect=[_login_success_response(), twofa_failure]):
        with pytest.raises(AuthError, match="Invalid TOTP"):
            _fetch_request_token("api_key", "user_id", "password", FAKE_TOTP_SECRET)


def test_twofa_failure_with_no_message_field_still_raises_clearly():
    twofa_failure = _mock_response(400, {"status": "error"})

    with patch("requests.Session.post", side_effect=[_login_success_response(), twofa_failure]):
        with pytest.raises(AuthError, match="2FA step failed"):
            _fetch_request_token("api_key", "user_id", "password", FAKE_TOTP_SECRET)


def test_login_step_failure_raises_clear_autherror():
    login_failure = _mock_response(400, {"status": "error", "message": "Invalid password"})

    with patch("requests.Session.post", side_effect=[login_failure]):
        with pytest.raises(AuthError, match="Invalid password"):
            _fetch_request_token("api_key", "user_id", "password", FAKE_TOTP_SECRET)


def test_login_step_captcha_failure_points_to_manual_login():
    captcha_failure = _mock_response(
        400, {"status": "error", "message": "Invalid CAPTCHA values.", "data": {"captcha": True}}
    )

    with patch("requests.Session.post", side_effect=[captcha_failure]):
        with pytest.raises(AuthError, match="manual_login.py"):
            _fetch_request_token("api_key", "user_id", "password", FAKE_TOTP_SECRET)
