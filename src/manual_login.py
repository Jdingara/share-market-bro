"""
One-time-per-day manual login, for when Zerodha's login page requires a
CAPTCHA (confirmed happening for this account 2026-07-30 - see auth.py's
module docstring for the full story). The automated password+TOTP flow in
auth.py cannot solve a CAPTCHA; a human has to, once, through a real browser.

Run this directly whenever auth.py reports a CAPTCHA error:

    py src/manual_login.py

It opens your browser to the login page automatically, runs a tiny local
server to catch Zerodha's redirect (which has consistently landed on
127.0.0.1:8000 for this app every time so far - confirmed live across
multiple real days), and finishes the login itself the moment you're done -
no copying URLs or switching windows required. The ONLY manual step is
logging in and solving the CAPTCHA yourself in the browser that opens.

If the automatic capture doesn't work for some reason (port already in use,
the redirect goes somewhere unexpected, etc.), it falls back to the old
copy-paste-the-URL approach after a short wait, so there's always a way to
finish even if the fast path fails.

Caches the resulting token to the exact same file auth.py's login() reads,
so any already-running paper_trader.py process picks it up on its next
retry automatically - no restart needed.
"""

from __future__ import annotations

import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from urllib.parse import parse_qs, urlparse

from kiteconnect import KiteConnect

from auth import AuthError, _require_env, _save_cached_token

REDIRECT_HOST = "127.0.0.1"
REDIRECT_PORT = 8000
CAPTURE_TIMEOUT_SECONDS = 180

_SUCCESS_PAGE = b"""<html><body style="font-family: sans-serif; text-align: center; margin-top: 15%;">
<h2>Logged in - you can close this tab.</h2></body></html>"""
_FAILURE_PAGE = b"""<html><body style="font-family: sans-serif; text-align: center; margin-top: 15%;">
<h2>No request_token in this redirect - something went wrong. Close this tab and check the terminal.</h2></body></html>"""


def _extract_request_token(pasted: str) -> str:
    """Accepts either the full redirect URL or just the bare request_token -
    a user copying from the browser address bar will usually grab the whole
    URL, but it's easy to instead copy just the token, so handle both.

    Checks for "://" (looks like a URL), not "request_token" - a URL that's
    missing the token entirely (wrong page, login failed, etc.) must raise
    clearly rather than silently being treated as if the whole URL were
    itself a valid token."""
    pasted = pasted.strip()
    if "://" in pasted:
        query = parse_qs(urlparse(pasted).query)
        token = query.get("request_token", [None])[0]
        if token:
            return token
        raise AuthError("Could not find a request_token in the pasted URL - check you copied the right one.")
    return pasted


def _make_capture_handler(captured: dict):
    class _RedirectHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            query = parse_qs(urlparse(self.path).query)
            token = query.get("request_token", [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            if token:
                captured["token"] = token
                self.wfile.write(_SUCCESS_PAGE)
            else:
                self.wfile.write(_FAILURE_PAGE)

        def log_message(self, format, *args):
            pass  # suppress default per-request console noise

    return _RedirectHandler


def _try_automatic_capture(login_url: str) -> str | None:
    """Starts a local server to catch Zerodha's redirect automatically, opens
    the login page in the browser, and waits. Returns the request_token, or
    None if it didn't arrive in time (caller falls back to manual paste)."""
    captured: dict = {}
    try:
        server = HTTPServer((REDIRECT_HOST, REDIRECT_PORT), _make_capture_handler(captured))
    except OSError as exc:
        print(f"(Couldn't start the local capture server on port {REDIRECT_PORT}: {exc} - falling back to manual paste.)")
        return None

    thread = Thread(target=server.handle_request, daemon=True)  # serves exactly one request, then stops
    thread.start()

    print("Opening your browser to log in - solve any CAPTCHA yourself, the rest happens automatically.")
    print(f"(If it doesn't open, copy this URL manually: {login_url})\n")
    webbrowser.open(login_url)

    print("Waiting for you to finish logging in...")
    thread.join(timeout=CAPTURE_TIMEOUT_SECONDS)
    server.server_close()

    return captured.get("token")


def main() -> None:
    api_key = _require_env("KITE_API_KEY")
    api_secret = _require_env("KITE_API_SECRET")

    login_url = f"https://kite.zerodha.com/connect/login?api_key={api_key}&v=3"

    request_token = _try_automatic_capture(login_url)

    if request_token is None:
        print("\nAutomatic capture didn't complete in time - falling back to the manual method.")
        print("1. Open this URL in a real browser and log in, if you haven't already:")
        print(f"\n   {login_url}\n")
        print("2. After logging in, Zerodha will try to redirect you somewhere that likely won't")
        print("   load - that's expected. Just copy the FULL URL from your browser's address bar")
        print("   at that point (it contains request_token).")
        print()
        pasted = input("3. Paste that URL (or just the request_token) here, then press Enter: ")
        request_token = _extract_request_token(pasted)

    kite = KiteConnect(api_key=api_key)
    session_data = kite.generate_session(request_token, api_secret=api_secret)
    access_token = session_data["access_token"]
    kite.set_access_token(access_token)

    _save_cached_token(access_token)

    profile = kite.profile()
    print(f"\nLogged in and cached a fresh token for: {profile['user_name']} ({profile['user_id']})")
    print("Any running paper_trader.py process will pick this up automatically on its next retry.")


if __name__ == "__main__":
    main()
