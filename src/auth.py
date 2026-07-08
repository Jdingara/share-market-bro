"""
Daily Kite Connect authentication.

Kite Connect access tokens expire every day (they're invalidated shortly after
midnight IST), so an unattended bot needs a way to log in fresh each morning
without a human clicking through a browser.

Zerodha's *officially documented* login flow is manual: open a browser login
URL, log in, get redirected back with a request_token. There is no official
headless/programmatic login API. The approach below automates the same login
steps (password + TOTP) by calling Kite's web login endpoints directly with
`requests`, which is the widely-used pattern in the retail algo-trading
community for exactly this problem. It is NOT part of Kite Connect's
documented public API, so if Zerodha changes their web login internals, this
will need updating - if `login()` starts failing, that's the first thing to
check.

The resulting access token is cached to `.cache/access_token.json` for the
day, so re-running any script the same day reuses it instead of logging in
again.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pyotp
import requests
from dotenv import load_dotenv
from kiteconnect import KiteConnect

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache"
TOKEN_CACHE_FILE = CACHE_DIR / "access_token.json"

KITE_LOGIN_URL = "https://kite.zerodha.com/api/login"
KITE_TWOFA_URL = "https://kite.zerodha.com/api/twofa"
KITE_CONNECT_LOGIN_URL = "https://kite.zerodha.com/connect/login"


class AuthError(RuntimeError):
    pass


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise AuthError(f"Missing required environment variable: {name} (check your .env file)")
    return value


def _load_cached_token() -> str | None:
    if not TOKEN_CACHE_FILE.exists():
        return None
    try:
        cached = json.loads(TOKEN_CACHE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if cached.get("date") != date.today().isoformat():
        return None
    return cached.get("access_token")


def _save_cached_token(access_token: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_CACHE_FILE.write_text(
        json.dumps({"date": date.today().isoformat(), "access_token": access_token})
    )


def _fetch_request_token(api_key: str, user_id: str, password: str, totp_secret: str) -> str:
    session = requests.Session()

    login_resp = session.post(KITE_LOGIN_URL, data={"user_id": user_id, "password": password})
    login_resp.raise_for_status()
    login_data = login_resp.json()
    if login_data.get("status") != "success":
        raise AuthError(f"Zerodha login step failed: {login_data.get('message', login_data)}")
    request_id = login_data["data"]["request_id"]

    totp = pyotp.TOTP(totp_secret).now()
    twofa_resp = session.post(
        KITE_TWOFA_URL,
        data={
            "user_id": user_id,
            "request_id": request_id,
            "twofa_value": totp,
            "twofa_type": "totp",
        },
    )
    twofa_resp.raise_for_status()
    twofa_data = twofa_resp.json()
    if twofa_data.get("status") != "success":
        raise AuthError(f"Zerodha 2FA step failed: {twofa_data.get('message', twofa_data)}")

    # Walk the connect/login redirect chain manually (without following the
    # final hop) so this works even though no redirect_url server is running -
    # the request_token shows up as a query param on the last Location header.
    url = f"{KITE_CONNECT_LOGIN_URL}?api_key={api_key}&v=3"
    for _ in range(10):
        resp = session.get(url, allow_redirects=False)
        location = resp.headers.get("Location")
        if not location:
            raise AuthError(
                "Login redirect chain ended without a request_token. "
                "Zerodha's login flow may have changed - see the note at the top of auth.py."
            )
        query = parse_qs(urlparse(location).query)
        if "request_token" in query:
            return query["request_token"][0]
        url = location

    raise AuthError("Too many redirects while looking for request_token.")


def login(force: bool = False) -> KiteConnect:
    """Return an authenticated KiteConnect client, logging in only if needed."""
    api_key = _require_env("KITE_API_KEY")

    if not force:
        cached_token = _load_cached_token()
        if cached_token:
            kite = KiteConnect(api_key=api_key)
            kite.set_access_token(cached_token)
            return kite

    api_secret = _require_env("KITE_API_SECRET")
    user_id = _require_env("KITE_USER_ID")
    password = _require_env("KITE_PASSWORD")
    totp_secret = _require_env("KITE_TOTP_SECRET")

    request_token = _fetch_request_token(api_key, user_id, password, totp_secret)

    kite = KiteConnect(api_key=api_key)
    session_data = kite.generate_session(request_token, api_secret=api_secret)
    access_token = session_data["access_token"]
    kite.set_access_token(access_token)

    _save_cached_token(access_token)
    return kite


if __name__ == "__main__":
    client = login()
    profile = client.profile()
    print(f"Logged in as: {profile['user_name']} ({profile['user_id']})")
