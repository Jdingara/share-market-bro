"""
Live paper trading loop: runs a signal engine (rule-based, or one of the ML
model types from ml_signal.py - defaults to Gradient Boosting/XGBoost, the
strongest candidate found in the Phase 6 comparison) during real market
hours against real live data, using REAL quoted option premiums (not
Black-Scholes simulation) for entry/exit.

No real orders are ever placed here (kite.place_order is never called) -
every "trade" is simulated bookkeeping against real market prices. This is
the validation step between backtesting (Phase 3/6, simulated premiums) and
live trading (Phase 5, real orders).

Carries the two pieces of the go-live safety net agreed 2026-08-06, first in
the build order because everything after it (CALL fixes, real order
placement) should sit on top of a working stop mechanism, not be the first
thing to get one: a manual KILL_SWITCH file (see KILL_SWITCH_PATH, managed via
`py src/kill_switch.py`) that halts new entries and force-closes any open
position within one poll interval, and an automatic daily loss circuit
breaker (MAX_DAILY_LOSS_PCT) that halts new entries once today's realized
loss reaches a set share of the day's starting capital. Both are already
active in paper mode so they're proven out before any real money is at risk.

CALL trades are gated PER MODEL, not with one flat switch (since 2026-08-12):
put_only (primary, 15-min) and early_session_put_only (early-session, 5-min)
are independent settings - see _effective_put_only. BOTH default to True
(PUT-only) as of 2026-08-15: the early-session model's CALL was briefly
allowed by default (08-12-08-14) based on a threshold scan that turned out
to be measuring the wrong bracket - backtester.py's STOP_LOSS_PCT had been
stale at -10% since 07-10 (the live bot uses -5%), so every label used to
justify that call was systematically too lenient. Re-run after fixing the
stale constant: the early-session model's CALL precision doesn't actually
climb with confidence under the real bracket, same "collapses" pattern as
the primary model. See PROJECT_STATUS.md for the full incident.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time as time_module
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from functools import partial
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from kiteconnect.exceptions import TokenException

from auth import login
from capital_manager import (
    MAX_CAPITAL_PER_TRADE,
    apply_trade_pnl,
    calculate_affordable_lots,
    deployable_capital,
    load_capital,
    save_capital,
)
from data_fetch import fetch_historical_data, get_instrument_token
from ml_features import attach_vix
from ml_signal import MODEL_TYPES, generate_ml_signal, load_models
from option_lookup import compute_max_pain, find_nearest_valid_expiry, find_option_instrument, get_option_premium
from options_pricing import nearest_strike
from signal_engine import RSI_PERIOD, RSI_TURN_LOOKBACK, generate_signal
from trade_chart import generate_trade_chart
from train_5min_model import MODEL_TYPE_5MIN

SIGNAL_SOURCES = ["rule_based"] + MODEL_TYPES
DEFAULT_SIGNAL_SOURCE = "gradient_boosting"  # strongest candidate in the Phase 6 comparison (77.1% win rate, +5.45% avg)

# generate_ml_signal/generate_signal both need this many candles before RSI (and
# the "did RSI just turn" check) is even computable - a hard floor, not a
# confidence issue. At 15-minute candles that's a 4-hour wait from market open
# (confirmed against every real trading day: no signal has ever fired before
# ~13:15). The early-session 5-minute model (train_5min_model.py) exists to
# shrink that same candle-count floor down to ~1h20m in wall-clock time.
MIN_CANDLES_FOR_SIGNAL = RSI_PERIOD + RSI_TURN_LOOKBACK

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPER_TRADES_DIR = PROJECT_ROOT / "data" / "paper_trades"
LOCK_FILE_PATH = PAPER_TRADES_DIR / "paper_trader.lock"
PAPER_TRADES_CSV_PATH = PAPER_TRADES_DIR / "paper_trades.csv"

# Emergency stop, first piece of the go-live safety net (agreed 2026-08-06, before
# any real order-placement code gets written). Presence of this file means "stop
# now" - checked before every new entry AND while a position is open (see
# _monitor_position_until_exit), so it can't get stuck behind a position that
# hasn't hit its normal exit yet. Deliberately a plain file, not a dashboard-only
# toggle - it works even if the dashboard is closed or the bot was launched from a
# terminal, and survives the bot process being restarted. Use `py src/kill_switch.py
# on/off/status` or the dashboard's Kill Switch button to manage it.
KILL_SWITCH_PATH = PAPER_TRADES_DIR / "KILL_SWITCH"

MARKET_CLOSE_TIME = time(15, 30)
FORCE_CLOSE_TIME = time(15, 25)  # force-close any open paper position before real market close

# Splits the day into two independent trade-count quotas instead of one flat
# max_trades_per_day, per the user's request (2026-07-16): up to N trades from
# market open through 13:15 (the "morning" window, when the early-session 5-min
# model is doing the work), then up to N MORE from 13:15 onward (the "afternoon"
# window, once the primary 15-min model has enough candles) - up to 2N total.
# 13:15 isn't arbitrary here - it's the same ~4h/16-candle floor documented on
# MIN_CANDLES_FOR_SIGNAL, just expressed as a wall-clock cutover for the quota
# logic rather than a candle count.
SESSION_SPLIT_TIME = time(13, 15)

# Earliest wall-clock time a new entry is allowed at all, regardless of which
# model would otherwise be eligible - added 2026-08-13 after analyzing all 116
# real paper trades logged so far. Trading hour turned out to be the single
# cleanest pattern in the data: 10:30-12:30 (entirely the early-session 5-min
# model's window - the primary model can't fire before SESSION_SPLIT_TIME
# regardless) lost -Rs 5,889.65 net across 32 trades, while every window from
# 12:30 onward was net positive. A cutoff scan across candidate times (10:30
# through 13:15, in 30-min steps) found 12:30 maximizes total P&L - moving it
# later than 12:30 starts cutting into the clearly-profitable 12:30-13:00 and
# 13:00-13:30 windows instead. Simulated on the real trade history: skipping
# everything before 12:30 would have turned +Rs 6,353.10 actual into
# +Rs 12,242.75 (kept 84 of 116 trades, win rate 41.3% -> 46.6%). A separate,
# much larger dip around 14:30 (-Rs 8,011.25) was checked and NOT included in
# this decision - traced to specific trades from 2026-07-09/07-10, before
# STOP_LOSS_PCT was tightened from -10% to -5% that same day, not a live
# pattern (excluding those two dates, the 14:30 window is only mildly negative
# across 5 different days - not the clean signal the morning window is).
EARLIEST_ENTRY_TIME = time(12, 30)

SIGNAL_POLL_INTERVAL_SECONDS = 60
POSITION_POLL_INTERVAL_SECONDS = 15

TARGET_PCT = 0.10
STOP_LOSS_PCT = 0.05

# Daily loss circuit breaker, the other half of the 2026-08-06 go-live safety-net
# plan. 10% is 2x the single-trade STOP_LOSS_PCT - enough headroom that one normal
# stop-loss never trips it on its own, but tight enough to actually cap a bad day
# with several trades (max_trades_per_day > 1, or --split-session) from digging a
# much deeper hole than a single trade ever could. Expressed as a share of the
# capital the day STARTED with (see day_start_capital in _run_impl), not the
# shrinking current balance, so the limit doesn't get easier to hit as losses
# compound within the same day.
MAX_DAILY_LOSS_PCT = 0.10

DAILY_HISTORY_DAYS = 100

RETRY_MAX_ATTEMPTS = 5
RETRY_BACKOFF_SECONDS = 5
RECOVERY_SLEEP_SECONDS = 30  # after an unexpected error even retries couldn't fix, pause before trying again


@dataclass
class PaperTrade:
    date: str
    signal_source: str
    direction: str
    zone: str
    tradingsymbol: str
    entry_time: str
    entry_premium: float
    strike: float
    expiry: str
    exit_time: str
    exit_premium: float
    exit_reason: str
    pct_change: float
    lots: int
    invested_amount: float
    pnl_rupees: float
    capital_after: float
    chart_path: str
    max_pain_strike: float
    max_pain_agreed: str


class DuplicateProcessError(RuntimeError):
    pass


def _is_process_alive(pid: int) -> bool:
    """Checks via Windows' built-in tasklist (no extra dependency like psutil needed)."""
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True, timeout=5,
        )
        return str(pid) in result.stdout
    except Exception:
        return False  # can't verify - safer to allow a new start than to block forever


def _acquire_lock(lock_path: Path = LOCK_FILE_PATH) -> None:
    """Refuses to start if another paper_trader.py instance is already running. Found
    necessary the hard way: duplicate processes have silently corrupted the trade log
    and capital file 3 separate times (2026-07-16, and two more discovered together on
    2026-07-21) - the dashboard's Start button alone does not prevent a second launch.
    Takes lock_path as a parameter (rather than hardcoding LOCK_FILE_PATH everywhere)
    so tests can point it at a temp file instead of the real production lock."""
    if lock_path.exists():
        try:
            existing_pid = int(lock_path.read_text().strip())
        except ValueError:
            existing_pid = None
        if existing_pid is not None and existing_pid != os.getpid() and _is_process_alive(existing_pid):
            raise DuplicateProcessError(
                f"Another paper_trader.py instance appears to already be running (PID {existing_pid}). "
                "Stop it first - running two at once corrupts the shared trade log and capital file."
            )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(str(os.getpid()))


def _release_lock(lock_path: Path = LOCK_FILE_PATH) -> None:
    try:
        lock_path.unlink(missing_ok=True)
    except Exception:
        pass


def _call_with_retry(func, *args, **kwargs):
    """Retries a network call a few times with backoff on transient connection errors
    (e.g. a connection reset) - this is meant to run unattended for 6+ hours, and a
    brief network blip shouldn't kill the whole day's monitoring. Confirmed necessary:
    a real ConnectionResetError killed an earlier live run with no retry logic at all."""
    last_exc = None
    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            print(
                f"[{datetime.now().time()}] API call failed (attempt {attempt}/{RETRY_MAX_ATTEMPTS}): "
                f"{exc}. Retrying in {RETRY_BACKOFF_SECONDS}s..."
            )
            time_module.sleep(RETRY_BACKOFF_SECONDS)
    raise last_exc


def _current_session(now: datetime) -> str:
    """Pure decision logic (no I/O), testable without a live feed - see SESSION_SPLIT_TIME."""
    return "morning" if now.time() < SESSION_SPLIT_TIME else "afternoon"


def _effective_put_only(used_early_session_model: bool, put_only: bool, early_session_put_only: bool) -> bool:
    """Pure decision logic (no I/O), testable without a live feed. Which put_only
    setting actually applies depends on which model produced the signal - lets
    the two models be gated independently if the evidence ever again supports
    treating them differently (it briefly did, 2026-08-12 to 08-14, based on a
    threshold scan that turned out to be measuring a stale, too-lenient -10%
    stop-loss in backtester.py instead of the real -5% one - see
    PROJECT_STATUS.md for the full incident). Both default to True (PUT-only)
    as of 2026-08-15, since neither model's CALL precision actually climbs
    with confidence under the corrected, real bracket."""
    return early_session_put_only if used_early_session_model else put_only


def _trade_slot_available(
    now: datetime,
    trades_taken_today: int,
    session_slots: dict,
    max_trades_per_day: int,
    max_trades_per_session: int,
    split_session: bool,
) -> bool:
    """Pure decision logic (no I/O), testable without a live feed."""
    if not split_session:
        return trades_taken_today < max_trades_per_day
    return session_slots[_current_session(now)] < max_trades_per_session


def _before_earliest_entry_time(now: datetime, earliest_entry_time: time) -> bool:
    """Pure decision logic (no I/O), testable without a live feed - see
    EARLIEST_ENTRY_TIME for the data behind this cutoff."""
    return now.time() < earliest_entry_time


def _kill_switch_engaged(kill_switch_path: Path = KILL_SWITCH_PATH) -> bool:
    """True when a human has dropped a KILL_SWITCH file to force an immediate stop.
    Takes kill_switch_path as a parameter (same pattern as _acquire_lock's lock_path)
    so tests can point it at a temp file instead of the real production path."""
    return kill_switch_path.exists()


def _todays_realized_pnl(csv_path: Path, today: date) -> float:
    """Sums pnl_rupees for every trade already logged today. Used to seed the daily
    loss circuit breaker correctly even if the bot restarts mid-day (e.g. after a
    crash or a manual restart) - a plain in-memory counter would silently forget
    losses already taken earlier that day, defeating the point of the breaker on
    exactly the day it matters most."""
    if not csv_path.exists():
        return 0.0
    df = pd.read_csv(csv_path)
    if df.empty or "date" not in df.columns or "pnl_rupees" not in df.columns:
        return 0.0
    todays_rows = df[df["date"] == today.isoformat()]
    return float(todays_rows["pnl_rupees"].sum())


def _circuit_breaker_tripped(daily_pnl: float, day_start_capital: float, max_daily_loss_pct: float) -> bool:
    """Pure decision logic (no I/O), testable without a live feed. See MAX_DAILY_LOSS_PCT
    for why this compares against the day's STARTING capital, not the current balance."""
    if day_start_capital <= 0:
        return False
    return daily_pnl <= -day_start_capital * max_daily_loss_pct


def _is_stale_signal(current_candle_time, last_entry_candle_time) -> bool:
    """Pure decision logic (no I/O), testable without a live feed. True when this
    signal is based on the same (or an older) candle as the last entry already
    acted on - i.e., no genuinely new market information has arrived since then.

    Found live on 2026-07-16: the bot polls every 60s but the early-session model
    only gets new data every 5 minutes, so after a stop-loss it could immediately
    re-see the identical still-unrefreshed signal and re-enter within seconds,
    walking straight back into the same move (5 stop-losses in ~24 minutes on the
    same contract, all logged at the exact same confidence value)."""
    return last_entry_candle_time is not None and current_candle_time <= last_entry_candle_time


def _reauthenticate():
    """Called when a TokenException means the current session is dead.

    Uses force=False (read the cache), NOT force=True (force a fresh automated
    login) - this flipped 2026-07-30. force=True was the fix on 2026-07-27 (see
    git history), back when the automated password+TOTP flow could actually
    produce a fresh token. It can't anymore: Zerodha started requiring a CAPTCHA
    on this account's login page, which that flow can't solve, so force=True
    now just fails every time - and doing that on every retry would keep
    hammering the CAPTCHA-protected endpoint, plausibly making whatever
    triggered it worse, while never recovering even after a human fixes things.

    force=False is safe to call repeatedly here specifically because every
    caller paces its retries at 15-30s minimum (position-monitoring poll,
    the outer per-cycle handler, or a one-off startup call) - never a tight
    loop - so if the cache is still stale this just waits for the next cycle,
    and the moment a human runs `py src/manual_login.py` (which updates the
    same cache file this reads fresh every time), the very next retry picks
    up the fresh token automatically, no restart needed."""
    return _call_with_retry(login, force=False)


def _login_with_retry():
    """Retries the initial login indefinitely (paced by RECOVERY_SLEEP_SECONDS,
    not a tight loop) instead of letting a startup login failure kill the whole
    process outright. Found live 2026-08-14: an unhandled AuthError/HTTPError
    here (that day, a raw /api/twofa failure) crashed the process before the
    main loop's TokenException recovery ever got a chance to run - the in-loop
    recovery this codebase already has (see the try/except around the main
    loop, and _monitor_position_until_exit) was never reachable for a startup
    failure. This makes the same "run manual_login.py, any running process
    picks up the fix automatically" promise (already true mid-loop) actually
    true at startup too, instead of requiring a manual restart after every
    startup auth failure. Gives up (re-raising the original exception) only
    once the market's already closed for the day - no point waiting for a
    login that can't lead to any trade anymore."""
    while True:
        try:
            return _reauthenticate()
        except Exception as exc:
            if datetime.now().time() >= MARKET_CLOSE_TIME:
                print(f"[{datetime.now().time()}] Could not log in before market close ({exc!r}) - giving up for today.")
                raise
            print(
                f"[{datetime.now().time()}] Could not log in yet ({exc!r}) - waiting and retrying. If this "
                "is a CAPTCHA or 2FA issue, run `py src/manual_login.py` to fix it; this process will pick "
                "up the fresh token automatically on its next retry, no restart needed."
            )
            time_module.sleep(RECOVERY_SLEEP_SECONDS)


def _monitor_position_until_exit(kite, tradingsymbol: str, entry_premium: float):
    """Polls an open position's premium until an exit condition fires. Returns
    (kite, exit_reason, exit_premium) - kite is returned because it may get
    reassigned on a mid-monitoring re-login.

    A TokenException here is recovered WITHOUT leaving this loop - it used to
    propagate to the caller's outer handler, which reauthenticates but then
    returns to the top of the main loop, forgetting this open position entirely
    (entry_premium/tradingsymbol were local variables in the caller, lost on
    unwind). Found live 2026-07-28: this let the bot open a SECOND position on
    top of the first, since it no longer knew one was open. Even a failed
    reauth attempt must not escape this loop either - staying here and retrying
    next poll is always safer than falling through to code that forgets this
    position exists.

    Also checks the kill switch (see KILL_SWITCH_PATH) on every poll - an
    emergency stop must not be able to get stuck waiting behind a position's
    normal target/stop-loss/EOD exit, which could otherwise be hours away."""
    exit_reason = None
    exit_premium = entry_premium
    while exit_reason is None:
        time_module.sleep(POSITION_POLL_INTERVAL_SECONDS)
        if _kill_switch_engaged():
            print(
                f"[{datetime.now().time()}] KILL SWITCH engaged - closing {tradingsymbol} now "
                "instead of waiting for its normal target/stop-loss/EOD exit."
            )
            try:
                exit_premium = _call_with_retry(get_option_premium, kite, tradingsymbol)
            except Exception as exc:
                print(f"Could not fetch a live exit price ({exc!r}) - recording the exit at entry price instead.")
                exit_premium = entry_premium
            return kite, "KILL_SWITCH", exit_premium
        try:
            current_premium = _call_with_retry(get_option_premium, kite, tradingsymbol)
        except TokenException:
            print(f"[{datetime.now().time()}] Access token invalid while monitoring open position - re-authenticating...")
            try:
                kite = _reauthenticate()
                print(f"[{datetime.now().time()}] Re-login succeeded, still watching {tradingsymbol}.")
            except Exception as relogin_exc:
                print(f"[{datetime.now().time()}] Re-login failed too ({relogin_exc!r}) - still watching {tradingsymbol}, will retry.")
            continue
        exit_reason = check_exit_condition(entry_premium, current_premium, datetime.now())
        exit_premium = current_premium
    return kite, exit_reason, exit_premium


def _record_trade_slot(now: datetime, trades_taken_today: int, session_slots: dict, split_session: bool) -> int:
    """Pure decision logic (no I/O), testable without a live feed. Mutates session_slots
    in place (dict) and returns the (possibly unchanged) trades_taken_today counter,
    since only one of the two counting schemes is active per run."""
    if split_session:
        session_slots[_current_session(now)] += 1
        return trades_taken_today
    return trades_taken_today + 1


def check_exit_condition(
    entry_premium: float,
    current_premium: float,
    current_time: datetime,
    force_close_time: time = FORCE_CLOSE_TIME,
) -> Optional[str]:
    """Pure decision logic (no I/O), so it's testable without a live feed."""
    pct_change = (current_premium - entry_premium) / entry_premium
    if pct_change >= TARGET_PCT:
        return "TARGET"
    if pct_change <= -STOP_LOSS_PCT:
        return "STOPLOSS"
    if current_time.time() >= force_close_time:
        return "EOD_CLOSE"
    return None


def _log_trade(trade: PaperTrade) -> Path:
    PAPER_TRADES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PAPER_TRADES_CSV_PATH
    pd.DataFrame([asdict(trade)]).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
    return out_path


def _fetch_today_intraday(kite, instrument_token: int) -> pd.DataFrame:
    # Generic despite the historical param name - reused for both NIFTY and,
    # since 2026-08-12, India VIX (see instrument_token callers in _run_impl).
    today = date.today()
    return fetch_historical_data(kite, instrument_token, today, today, "15minute")


def _fetch_today_intraday_5min(kite, instrument_token: int) -> pd.DataFrame:
    today = date.today()
    return fetch_historical_data(kite, instrument_token, today, today, "5minute")


def _fetch_prior_daily(kite, instrument_token: int) -> pd.DataFrame:
    today = date.today()
    from_date = today - timedelta(days=DAILY_HISTORY_DAYS)
    df = fetch_historical_data(kite, instrument_token, from_date, today, "day")
    df["date"] = pd.to_datetime(df["date"])
    return df[df["date"].dt.date < today].reset_index(drop=True)


def _build_signal_fn(signal_source: str):
    """Returns a callable matching generate_signal()'s exact (daily_df, intraday_df) -> Signal
    interface, whichever signal_source is chosen - rule-based or one of the ML model types."""
    if signal_source == "rule_based":
        return generate_signal
    call_model, put_model, call_threshold, put_threshold = load_models(signal_source)
    return partial(
        generate_ml_signal,
        call_model=call_model, put_model=put_model,
        call_threshold=call_threshold, put_threshold=put_threshold,
    )


def _build_early_session_signal_fn(signal_source: str):
    """5-minute-candle model, used only until enough 15-minute candles exist for
    the primary model (see MIN_CANDLES_FOR_SIGNAL). Only trained for the
    gradient_boosting model type so far - returns None for anything else, in
    which case the early-morning wait behaves exactly as it did before."""
    if signal_source != "gradient_boosting":
        return None
    call_model, put_model, call_threshold, put_threshold = load_models(MODEL_TYPE_5MIN)
    return partial(
        generate_ml_signal,
        call_model=call_model, put_model=put_model,
        call_threshold=call_threshold, put_threshold=put_threshold,
    )


def _run_impl(
    max_minutes: Optional[int] = None,
    signal_source: str = DEFAULT_SIGNAL_SOURCE,
    max_trades_per_day: int = 1,
    max_capital_per_trade: float = MAX_CAPITAL_PER_TRADE,
    put_only: bool = True,
    early_session_put_only: bool = True,
    split_session: bool = False,
    max_trades_per_session: int = 6,
    max_daily_loss_pct: float = MAX_DAILY_LOSS_PCT,
    earliest_entry_time: time = EARLIEST_ENTRY_TIME,
) -> None:
    print(f"Signal source for today: {signal_source}")
    print(f"No new entries before {earliest_entry_time} (see EARLIEST_ENTRY_TIME's docstring for the data behind this).")
    if split_session:
        print(
            f"NOTE: split_session=True - up to {max_trades_per_session} trades before {SESSION_SPLIT_TIME} "
            f"(morning session) and up to {max_trades_per_session} more from {SESSION_SPLIT_TIME} onward "
            f"(afternoon session) - up to {max_trades_per_session * 2} trades today. "
            "max_trades_per_day is ignored while this is on."
        )
    elif max_trades_per_day != 1:
        print(
            f"NOTE: max_trades_per_day={max_trades_per_day} (not the default of 1) - this is a "
            "fast-validation setting, not the intended live-trading discipline. Switch back to 1 "
            "once enough data has been gathered."
        )
    if put_only:
        print(
            "NOTE: put_only=True (the default) - the PRIMARY model's CALL signals are being skipped. "
            "Its CALL confidence doesn't reliably climb with confidence (confirmed 2026-08-15, after "
            "fixing a stale backtester bracket) - use --allow-calls to override."
        )
    else:
        print(
            "WARNING: --allow-calls passed - the primary model's CALL signals are being taken despite "
            "still-weak precision there. Only use this deliberately, e.g. to gather more CALL data."
        )
    if early_session_put_only:
        print(
            "NOTE: early_session_put_only=True (the default) - the EARLY-SESSION model's CALL signals "
            "are being skipped too. Briefly allowed by default 08-12 to 08-14 based on a threshold scan "
            "that turned out to be measuring a stale, too-lenient stop-loss - reverted 2026-08-15 once "
            "corrected. Use --allow-calls-early-session to override."
        )
    else:
        print(
            "WARNING: --allow-calls-early-session passed - the early-session model's CALL signals are "
            "being taken despite still-weak precision there under the corrected bracket. Only use this "
            "deliberately, e.g. to gather more CALL data."
        )
    signal_fn = _build_signal_fn(signal_source)
    early_signal_fn = _build_early_session_signal_fn(signal_source)
    if early_signal_fn is not None:
        print("Early-session (5-min candle) model loaded - can signal from ~1h20m after open instead of ~4h.")

    kite = _login_with_retry()
    nifty_token = _call_with_retry(get_instrument_token, kite, "NIFTY 50", "NSE")
    vix_token = _call_with_retry(get_instrument_token, kite, "INDIA VIX", "NSE")

    daily_df = _call_with_retry(_fetch_prior_daily, kite, nifty_token)
    print(f"Loaded {len(daily_df)} prior daily candles.")

    # India VIX features (added 2026-08-12) - merged in once here at startup for
    # the slow-moving daily side; the fast-moving intraday side is re-fetched and
    # re-merged every poll cycle below, alongside NIFTY's own intraday fetch.
    vix_daily_df = _call_with_retry(_fetch_prior_daily, kite, vix_token)
    daily_df = attach_vix(daily_df, vix_daily_df)

    nfo_instruments = _call_with_retry(kite.instruments, "NFO")
    print(f"Loaded {len(nfo_instruments)} NFO instruments.")

    capital = load_capital()
    print(f"Starting capital: Rs {capital:,.2f}")
    print(f"Max capital per trade: Rs {max_capital_per_trade:,.2f}")

    day_start_capital = capital
    daily_pnl = _todays_realized_pnl(PAPER_TRADES_CSV_PATH, date.today())
    if daily_pnl != 0.0:
        print(f"Already have Rs {daily_pnl:,.2f} of realized P&L logged today (resuming mid-day) - circuit breaker starts from there.")
    daily_loss_limit = day_start_capital * max_daily_loss_pct
    print(f"Daily loss circuit breaker: stop new entries if today's P&L reaches -Rs {daily_loss_limit:,.2f} ({max_daily_loss_pct:.0%} of start-of-day capital).")

    start_time = datetime.now()
    trades_taken_today = 0
    session_slots = {"morning": 0, "afternoon": 0}
    last_entry_candle_time = None

    while True:
        now = datetime.now()
        if max_minutes is not None and (now - start_time).total_seconds() > max_minutes * 60:
            print("Max runtime reached, stopping (smoke test mode).")
            break
        if now.time() >= MARKET_CLOSE_TIME:
            print("Market closed, stopping for the day.")
            break
        if _kill_switch_engaged():
            print(
                f"[{now.time()}] KILL SWITCH engaged (data/paper_trades/KILL_SWITCH present) - stopping, "
                "no new entries. Run `py src/kill_switch.py off` (or delete that file) to resume."
            )
            break
        if _circuit_breaker_tripped(daily_pnl, day_start_capital, max_daily_loss_pct):
            print(
                f"[{now.time()}] DAILY LOSS CIRCUIT BREAKER TRIPPED: today's realized P&L is Rs {daily_pnl:,.2f} "
                f"(limit -Rs {daily_loss_limit:,.2f}) - no new entries for the rest of today."
            )
            break
        if _before_earliest_entry_time(now, earliest_entry_time):
            print(
                f"[{now.time()}] Before {earliest_entry_time} - no new entries yet (real trade data shows this "
                f"window underperforms, see EARLIEST_ENTRY_TIME) - waiting."
            )
            time_module.sleep(SIGNAL_POLL_INTERVAL_SECONDS)
            continue
        if not _trade_slot_available(now, trades_taken_today, session_slots, max_trades_per_day, max_trades_per_session, split_session):
            if split_session and _current_session(now) == "morning":
                # Morning quota is full but the afternoon session hasn't opened yet -
                # pause new entries, don't end the day; the afternoon quota opens on
                # its own once the clock crosses SESSION_SPLIT_TIME (checked again
                # next poll - no special-casing needed for the handoff itself).
                print(
                    f"[{now.time()}] Morning session quota complete ({session_slots['morning']}/{max_trades_per_session}) "
                    f"- pausing new entries until the afternoon session opens at {SESSION_SPLIT_TIME}."
                )
                time_module.sleep(SIGNAL_POLL_INTERVAL_SECONDS)
                continue
            limit_desc = f"{max_trades_per_session}/session" if split_session else f"{max_trades_per_day}/day"
            taken_desc = sum(session_slots.values()) if split_session else trades_taken_today
            print(f"Already completed today's {taken_desc} trade(s) (limit {limit_desc}) - done for the day.")
            break
        if now.time() >= FORCE_CLOSE_TIME:
            print(f"Past force-close time ({FORCE_CLOSE_TIME}) - no time left for a new entry to develop, stopping for the day.")
            break

        try:
            intraday_df = _call_with_retry(_fetch_today_intraday, kite, nifty_token)
            if intraday_df.empty:
                print(f"[{now.time()}] No intraday candles yet, waiting...")
                time_module.sleep(SIGNAL_POLL_INTERVAL_SECONDS)
                continue

            if len(intraday_df) >= MIN_CANDLES_FOR_SIGNAL or early_signal_fn is None:
                used_early_session_model = False
                vix_intraday_df = _call_with_retry(_fetch_today_intraday, kite, vix_token)
                intraday_df = attach_vix(intraday_df, vix_intraday_df)
                signal = signal_fn(daily_df, intraday_df)
                current_candle_time = intraday_df.iloc[-1]["date"]
            else:
                # Not enough 15-min candles yet for the primary model - try the
                # 5-min early-session model instead, so a real morning setup
                # isn't missed for hours purely due to the candle-count floor.
                used_early_session_model = True
                intraday_5min_df = _call_with_retry(_fetch_today_intraday_5min, kite, nifty_token)
                if len(intraday_5min_df) < MIN_CANDLES_FOR_SIGNAL:
                    print(f"[{now.time()}] Not enough candles yet for even the early-session model, waiting...")
                    time_module.sleep(SIGNAL_POLL_INTERVAL_SECONDS)
                    continue
                vix_intraday_5min_df = _call_with_retry(_fetch_today_intraday_5min, kite, vix_token)
                intraday_5min_df = attach_vix(intraday_5min_df, vix_intraday_5min_df)
                signal = early_signal_fn(daily_df, intraday_5min_df)
                current_candle_time = intraday_5min_df.iloc[-1]["date"]
                if signal.direction != "NO_TRADE":
                    signal.reasoning = f"[early-session 5-min model] {signal.reasoning}"

            if signal.direction == "NO_TRADE":
                print(f"[{now.time()}] No signal yet. {signal.reasoning}")
                time_module.sleep(SIGNAL_POLL_INTERVAL_SECONDS)
                continue

            if _is_stale_signal(current_candle_time, last_entry_candle_time):
                print(
                    f"[{now.time()}] {signal.direction} signal unchanged since the last entry's candle "
                    f"({last_entry_candle_time}) - waiting for a fresh candle before re-entering."
                )
                time_module.sleep(SIGNAL_POLL_INTERVAL_SECONDS)
                continue

            if _effective_put_only(used_early_session_model, put_only, early_session_put_only) and signal.direction == "CALL":
                model_desc = "early-session model" if used_early_session_model else "primary model"
                print(f"[{now.time()}] CALL signal skipped (put_only mode, {model_desc}): {signal.reasoning}")
                time_module.sleep(SIGNAL_POLL_INTERVAL_SECONDS)
                continue

            print(f"[{now.time()}] SIGNAL: {signal.direction} - {signal.reasoning}")
            last_entry_candle_time = current_candle_time

            option_type = "CE" if signal.direction == "CALL" else "PE"
            strike = nearest_strike(signal.trigger_price)

            try:
                # Real listed expiry, not a weekday guess - robust to NSE changing the
                # weekly expiry day again (it already has once, silently breaking a live
                # trade on 2026-07-08 when this used to call the weekday-guessing function).
                expiry = find_nearest_valid_expiry(nfo_instruments, signal.timestamp.date())
                instrument = find_option_instrument(nfo_instruments, strike, expiry, option_type)
            except Exception as exc:
                print(f"Could not find matching option contract ({exc}) - skipping this trade slot.")
                trades_taken_today = _record_trade_slot(now, trades_taken_today, session_slots, split_session)
                continue

            try:
                # Shadow mode only - logged for later analysis, never filters or blocks
                # a trade. OI data isn't retained after expiry, so this can only ever
                # be computed live, never backtested.
                max_pain_strike = _call_with_retry(compute_max_pain, kite, nfo_instruments, expiry)
                if signal.direction == "PUT":
                    max_pain_agreed = "YES" if signal.trigger_price > max_pain_strike else "NO"
                else:
                    max_pain_agreed = "YES" if signal.trigger_price < max_pain_strike else "NO"
                print(
                    f"Max Pain (shadow mode): strike {max_pain_strike:.0f} vs spot {signal.trigger_price:.0f} "
                    f"- {'agrees' if max_pain_agreed == 'YES' else 'disagrees'} with {signal.direction} signal"
                )
            except Exception as exc:
                print(f"Could not compute Max Pain ({exc}) - continuing without it (shadow mode, non-critical).")
                max_pain_strike = 0.0
                max_pain_agreed = ""

            tradingsymbol = instrument["tradingsymbol"]
            entry_premium = _call_with_retry(get_option_premium, kite, tradingsymbol)

            capped_capital = deployable_capital(capital, max_capital_per_trade)
            lots = calculate_affordable_lots(capped_capital, entry_premium)
            if lots == 0:
                needed = entry_premium * 65
                print(
                    f"Insufficient capital (Rs {capped_capital:,.2f} deployable, capped at Rs {max_capital_per_trade:,.2f}) "
                    f"for even 1 lot at this premium (needs Rs {needed:,.2f}) - skipping this trade slot."
                )
                trades_taken_today = _record_trade_slot(now, trades_taken_today, session_slots, split_session)
                continue

            invested_amount = lots * 65 * entry_premium
            entry_time = datetime.now()
            print(f"PAPER ENTRY: {signal.direction} {tradingsymbol} @ {entry_premium} x {lots} lot(s) (Rs {invested_amount:,.2f})")

            kite, exit_reason, exit_premium = _monitor_position_until_exit(kite, tradingsymbol, entry_premium)

            exit_time = datetime.now()
            pct_change = (exit_premium - entry_premium) / entry_premium
            new_capital = apply_trade_pnl(capital, lots, entry_premium, exit_premium)
            pnl_rupees = new_capital - capital
            daily_pnl += pnl_rupees
            save_capital(new_capital)
            print(
                f"PAPER EXIT: {exit_reason} @ {exit_premium} ({pct_change:+.2%}) | "
                f"P&L Rs {pnl_rupees:,.2f} | capital Rs {capital:,.2f} -> Rs {new_capital:,.2f}"
            )

            try:
                chart_path = generate_trade_chart(
                    kite,
                    instrument["instrument_token"],
                    tradingsymbol,
                    signal.direction,
                    entry_time,
                    exit_time,
                    entry_premium,
                    exit_premium,
                    exit_reason,
                )
                print(f"Saved trade chart -> {chart_path}")
            except Exception as exc:
                # Diagnostic nice-to-have, not core trading logic - never let a charting
                # failure take down the trading loop.
                print(f"Could not generate trade chart ({exc}) - continuing without it.")
                chart_path = ""

            trade = PaperTrade(
                date=entry_time.date().isoformat(),
                signal_source=signal_source,
                direction=signal.direction,
                zone=signal.fib_level,
                tradingsymbol=tradingsymbol,
                entry_time=entry_time.isoformat(),
                entry_premium=entry_premium,
                strike=strike,
                expiry=expiry.isoformat(),
                exit_time=exit_time.isoformat(),
                exit_premium=exit_premium,
                exit_reason=exit_reason,
                pct_change=round(pct_change, 4),
                lots=lots,
                invested_amount=round(invested_amount, 2),
                pnl_rupees=round(pnl_rupees, 2),
                capital_after=round(new_capital, 2),
                chart_path=str(chart_path),
                max_pain_strike=max_pain_strike,
                max_pain_agreed=max_pain_agreed,
            )
            out_path = _log_trade(trade)
            print(f"Logged trade -> {out_path}")
            capital = new_capital  # so the next trade slot (if max_trades_per_day > 1) sizes off the updated balance
            trades_taken_today = _record_trade_slot(now, trades_taken_today, session_slots, split_session)

        except TokenException:
            # Found live 2026-07-23: the access token can go invalid mid-session (not
            # just at the expected daily reset), and blindly retrying the same broken
            # kite object never recovers - it left a real open position unmonitored for
            # ~40 minutes that day. Re-login instead of just sleeping and hoping.
            print(f"[{datetime.now().time()}] Access token invalid - attempting a fresh login...")
            try:
                kite = _reauthenticate()
                print(f"[{datetime.now().time()}] Re-login succeeded, resuming.")
            except Exception as relogin_exc:
                print(f"[{datetime.now().time()}] Re-login failed too ({relogin_exc!r}) - will retry.")
            time_module.sleep(RECOVERY_SLEEP_SECONDS)

        except Exception as exc:
            # Final safety net: even after retries, something unexpected went wrong.
            # This must never take the whole script down mid-day - log it clearly,
            # pause briefly, and keep going rather than silently dying like it did
            # in an earlier live run (a network blip killed the entire day's monitoring).
            print(f"[{datetime.now().time()}] UNEXPECTED ERROR (continuing): {exc!r}")
            time_module.sleep(RECOVERY_SLEEP_SECONDS)


def run(*args, **kwargs) -> None:
    """Thin wrapper around _run_impl() that refuses to start if another instance is
    already running (see _acquire_lock's docstring) - always releases the lock on
    exit, even if _run_impl() crashes."""
    _acquire_lock()
    try:
        _run_impl(*args, **kwargs)
    finally:
        _release_lock()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live paper trading loop (simulated orders, real market data/quotes).")
    parser.add_argument("--max-minutes", type=int, default=None, help="Stop after N minutes (for smoke testing).")
    parser.add_argument("--signal-source", choices=SIGNAL_SOURCES, default=DEFAULT_SIGNAL_SOURCE,
                         help=f"Which signal engine to use (default: {DEFAULT_SIGNAL_SOURCE}).")
    parser.add_argument("--max-trades-per-day", type=int, default=1,
                         help="Trade slots per day (default: 1, the intended live discipline). Set higher "
                              "(e.g. 20) only for fast validation days - switch back to 1 afterward.")
    parser.add_argument("--max-capital-per-trade", type=float, default=MAX_CAPITAL_PER_TRADE,
                         help=f"Never deploy more than this much of the balance on one trade "
                              f"(default: Rs {MAX_CAPITAL_PER_TRADE:,.2f}). Excess balance stays idle.")
    parser.add_argument("--allow-calls", action="store_true",
                         help="Allow the PRIMARY model's CALL signals too (default: off, PUT-only). Its CALL "
                              "confidence doesn't reliably climb with confidence (confirmed 2026-08-15) - "
                              "PUT-only is the standing default for this model. The early-session model's "
                              "CALL signals are handled separately, see --allow-calls-early-session.")
    parser.add_argument("--allow-calls-early-session", action="store_true",
                         help="Allow the EARLY-SESSION (5-min) model's CALL signals too (default: off, "
                              "PUT-only - same as the primary model). Briefly allowed by default 08-12 to "
                              "08-14 based on a threshold scan that turned out to be measuring a stale, "
                              "too-lenient backtester stop-loss (-10%% instead of the live -5%%) - reverted "
                              "2026-08-15 once corrected, since the real precision doesn't climb with "
                              "confidence either. Only use this deliberately, e.g. to gather more CALL data.")
    parser.add_argument("--split-session", action="store_true",
                         help=f"Split the daily cap into two independent windows instead of one flat "
                              f"--max-trades-per-day: up to --max-trades-per-session trades before "
                              f"{SESSION_SPLIT_TIME} (morning), then up to --max-trades-per-session more from "
                              f"{SESSION_SPLIT_TIME} onward (afternoon) - up to 2x --max-trades-per-session total. "
                              "Overrides --max-trades-per-day when set.")
    parser.add_argument("--max-trades-per-session", type=int, default=6,
                         help="Trade cap per session when --split-session is set (default: 6, i.e. up to 12/day).")
    parser.add_argument("--max-daily-loss-pct", type=float, default=MAX_DAILY_LOSS_PCT,
                         # NOTE: literal %% (not %), since argparse's help formatter treats a lone
                         # % as its own substitution syntax and raises otherwise.
                         help=f"Circuit breaker: stop taking new trades for the day once today's realized loss "
                              f"reaches this share of start-of-day capital (default: {MAX_DAILY_LOSS_PCT * 100:.0f}%%)."
                              " Any position already open when it trips still runs to its normal exit.")
    parser.add_argument("--earliest-entry-time", type=lambda s: datetime.strptime(s, "%H:%M").time(),
                         default=EARLIEST_ENTRY_TIME,
                         help=f"No new entries before this wall-clock time, HH:MM 24h (default: "
                              f"{EARLIEST_ENTRY_TIME.strftime('%H:%M')}). Based on analyzing all real paper "
                              "trades so far: 10:30-12:30 (the early-session model's window) lost money net, "
                              "every window from 12:30 onward was net positive - see EARLIEST_ENTRY_TIME's "
                              "docstring for the full numbers. Lower this (e.g. 00:00) to deliberately gather "
                              "more early-morning data instead.")
    args = parser.parse_args()
    run(
        max_minutes=args.max_minutes,
        signal_source=args.signal_source,
        max_trades_per_day=args.max_trades_per_day,
        max_capital_per_trade=args.max_capital_per_trade,
        put_only=not args.allow_calls,
        early_session_put_only=not args.allow_calls_early_session,
        split_session=args.split_session,
        max_trades_per_session=args.max_trades_per_session,
        max_daily_loss_pct=args.max_daily_loss_pct,
        earliest_entry_time=args.earliest_entry_time,
    )
