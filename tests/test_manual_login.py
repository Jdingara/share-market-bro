"""Unit tests for manual_login.py's pure request_token extraction logic."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from auth import AuthError
from manual_login import _extract_request_token


def test_extract_request_token_from_full_redirect_url():
    url = "http://127.0.0.1/redirect?action=login&type=login&status=success&request_token=abc123XYZ"
    assert _extract_request_token(url) == "abc123XYZ"


def test_extract_request_token_from_bare_token():
    # A user might copy just the token instead of the whole URL - both must work.
    assert _extract_request_token("abc123XYZ") == "abc123XYZ"


def test_extract_request_token_strips_surrounding_whitespace():
    assert _extract_request_token("  abc123XYZ  ") == "abc123XYZ"


def test_extract_request_token_raises_clearly_when_url_has_no_token():
    with pytest.raises(AuthError):
        _extract_request_token("http://127.0.0.1/redirect?action=login&status=success")
