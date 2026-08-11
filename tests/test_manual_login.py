"""Unit tests for manual_login.py's request_token extraction and local-server
redirect-capture logic."""

import sys
import threading
from http.server import HTTPServer
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from auth import AuthError
from manual_login import _extract_request_token, _make_capture_handler


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


def _run_capture_server(port: int):
    captured: dict = {}
    server = HTTPServer(("127.0.0.1", port), _make_capture_handler(captured))
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    return server, thread, captured


def test_capture_handler_extracts_token_from_a_real_redirect_request():
    # Simulates exactly what Zerodha's browser redirect sends - a real HTTP
    # GET request, not a mocked one, so this exercises the actual server code.
    server, thread, captured = _run_capture_server(8766)
    try:
        resp = requests.get(
            "http://127.0.0.1:8766/?action=login&type=login&status=success&request_token=abc123XYZ",
            timeout=5,
        )
        assert resp.status_code == 200
        assert b"Logged in" in resp.content
        thread.join(timeout=5)
        assert captured.get("token") == "abc123XYZ"
    finally:
        server.server_close()


def test_capture_handler_shows_failure_page_when_redirect_has_no_token():
    # e.g. login failed, or Zerodha's redirect shape changes - must not crash,
    # and must not silently "capture" a missing token as if it were real.
    server, thread, captured = _run_capture_server(8767)
    try:
        resp = requests.get("http://127.0.0.1:8767/?action=login&status=error", timeout=5)
        assert resp.status_code == 200
        assert b"No request_token" in resp.content
        thread.join(timeout=5)
        assert "token" not in captured
    finally:
        server.server_close()
