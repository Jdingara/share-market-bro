"""
One-time-per-day manual login, for when Zerodha's login page requires a
CAPTCHA (confirmed happening for this account 2026-07-30 - see auth.py's
module docstring for the full story). The automated password+TOTP flow in
auth.py cannot solve a CAPTCHA; a human has to, once, through a real browser.

Run this directly whenever auth.py reports a CAPTCHA error:

    py src/manual_login.py

It caches the resulting token to the exact same file auth.py's login()
reads, so any already-running paper_trader.py process picks it up on its
next retry automatically - no restart needed.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from kiteconnect import KiteConnect

from auth import AuthError, _require_env, _save_cached_token


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


def main() -> None:
    api_key = _require_env("KITE_API_KEY")
    api_secret = _require_env("KITE_API_SECRET")

    login_url = f"https://kite.zerodha.com/connect/login?api_key={api_key}&v=3"
    print("1. Open this URL in a real browser and log in (you'll solve any CAPTCHA yourself):")
    print(f"\n   {login_url}\n")
    print("2. After logging in, Zerodha will try to redirect you somewhere that likely won't")
    print("   load (there's no server running to receive it) - that's expected. Just copy the")
    print("   FULL URL from your browser's address bar at that point (it contains request_token).")
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
