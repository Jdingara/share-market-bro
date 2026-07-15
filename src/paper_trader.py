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
"""

from __future__ import annotations

import argparse
import time as time_module
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from functools import partial
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

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
from ml_signal import MODEL_TYPES, generate_ml_signal, load_models
from option_lookup import find_nearest_valid_expiry, find_option_instrument, get_option_premium
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

MARKET_CLOSE_TIME = time(15, 30)
FORCE_CLOSE_TIME = time(15, 25)  # force-close any open paper position before real market close

SIGNAL_POLL_INTERVAL_SECONDS = 60
POSITION_POLL_INTERVAL_SECONDS = 15

TARGET_PCT = 0.10
STOP_LOSS_PCT = 0.05

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
    out_path = PAPER_TRADES_DIR / "paper_trades.csv"
    pd.DataFrame([asdict(trade)]).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
    return out_path


def _fetch_today_intraday(kite, nifty_token: int) -> pd.DataFrame:
    today = date.today()
    return fetch_historical_data(kite, nifty_token, today, today, "15minute")


def _fetch_today_intraday_5min(kite, nifty_token: int) -> pd.DataFrame:
    today = date.today()
    return fetch_historical_data(kite, nifty_token, today, today, "5minute")


def _fetch_prior_daily(kite, nifty_token: int) -> pd.DataFrame:
    today = date.today()
    from_date = today - timedelta(days=DAILY_HISTORY_DAYS)
    df = fetch_historical_data(kite, nifty_token, from_date, today, "day")
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


def run(
    max_minutes: Optional[int] = None,
    signal_source: str = DEFAULT_SIGNAL_SOURCE,
    max_trades_per_day: int = 1,
    max_capital_per_trade: float = MAX_CAPITAL_PER_TRADE,
    put_only: bool = False,
) -> None:
    print(f"Signal source for today: {signal_source}")
    if max_trades_per_day != 1:
        print(
            f"NOTE: max_trades_per_day={max_trades_per_day} (not the default of 1) - this is a "
            "fast-validation setting, not the intended live-trading discipline. Switch back to 1 "
            "once enough data has been gathered."
        )
    if put_only:
        print(
            "NOTE: put_only=True - CALL signals are being skipped this run. The CALL side of this model "
            "has shown weaker precision than PUT's in both backtest and live results, but both directions "
            "remain in scope by default - this is an explicit override, not the standing behavior."
        )
    signal_fn = _build_signal_fn(signal_source)
    early_signal_fn = _build_early_session_signal_fn(signal_source)
    if early_signal_fn is not None:
        print("Early-session (5-min candle) model loaded - can signal from ~1h20m after open instead of ~4h.")

    kite = _call_with_retry(login)
    nifty_token = _call_with_retry(get_instrument_token, kite, "NIFTY 50", "NSE")

    daily_df = _call_with_retry(_fetch_prior_daily, kite, nifty_token)
    print(f"Loaded {len(daily_df)} prior daily candles.")

    nfo_instruments = _call_with_retry(kite.instruments, "NFO")
    print(f"Loaded {len(nfo_instruments)} NFO instruments.")

    capital = load_capital()
    print(f"Starting capital: Rs {capital:,.2f}")
    print(f"Max capital per trade: Rs {max_capital_per_trade:,.2f}")

    start_time = datetime.now()
    trades_taken_today = 0

    while True:
        now = datetime.now()
        if max_minutes is not None and (now - start_time).total_seconds() > max_minutes * 60:
            print("Max runtime reached, stopping (smoke test mode).")
            break
        if now.time() >= MARKET_CLOSE_TIME:
            print("Market closed, stopping for the day.")
            break
        if trades_taken_today >= max_trades_per_day:
            print(f"Already completed today's {trades_taken_today} trade(s) (limit {max_trades_per_day}) - done for the day.")
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
                signal = signal_fn(daily_df, intraday_df)
            else:
                # Not enough 15-min candles yet for the primary model - try the
                # 5-min early-session model instead, so a real morning setup
                # isn't missed for hours purely due to the candle-count floor.
                intraday_5min_df = _call_with_retry(_fetch_today_intraday_5min, kite, nifty_token)
                if len(intraday_5min_df) < MIN_CANDLES_FOR_SIGNAL:
                    print(f"[{now.time()}] Not enough candles yet for even the early-session model, waiting...")
                    time_module.sleep(SIGNAL_POLL_INTERVAL_SECONDS)
                    continue
                signal = early_signal_fn(daily_df, intraday_5min_df)
                if signal.direction != "NO_TRADE":
                    signal.reasoning = f"[early-session 5-min model] {signal.reasoning}"

            if signal.direction == "NO_TRADE":
                print(f"[{now.time()}] No signal yet. {signal.reasoning}")
                time_module.sleep(SIGNAL_POLL_INTERVAL_SECONDS)
                continue

            if put_only and signal.direction == "CALL":
                print(f"[{now.time()}] CALL signal skipped (put_only mode): {signal.reasoning}")
                time_module.sleep(SIGNAL_POLL_INTERVAL_SECONDS)
                continue

            print(f"[{now.time()}] SIGNAL: {signal.direction} - {signal.reasoning}")

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
                trades_taken_today += 1
                continue

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
                trades_taken_today += 1
                continue

            invested_amount = lots * 65 * entry_premium
            entry_time = datetime.now()
            print(f"PAPER ENTRY: {signal.direction} {tradingsymbol} @ {entry_premium} x {lots} lot(s) (Rs {invested_amount:,.2f})")

            exit_reason = None
            exit_premium = entry_premium
            while exit_reason is None:
                time_module.sleep(POSITION_POLL_INTERVAL_SECONDS)
                current_premium = _call_with_retry(get_option_premium, kite, tradingsymbol)
                exit_reason = check_exit_condition(entry_premium, current_premium, datetime.now())
                exit_premium = current_premium

            exit_time = datetime.now()
            pct_change = (exit_premium - entry_premium) / entry_premium
            new_capital = apply_trade_pnl(capital, lots, entry_premium, exit_premium)
            pnl_rupees = new_capital - capital
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
            )
            out_path = _log_trade(trade)
            print(f"Logged trade -> {out_path}")
            capital = new_capital  # so the next trade slot (if max_trades_per_day > 1) sizes off the updated balance
            trades_taken_today += 1

        except Exception as exc:
            # Final safety net: even after retries, something unexpected went wrong.
            # This must never take the whole script down mid-day - log it clearly,
            # pause briefly, and keep going rather than silently dying like it did
            # in an earlier live run (a network blip killed the entire day's monitoring).
            print(f"[{datetime.now().time()}] UNEXPECTED ERROR (continuing): {exc!r}")
            time_module.sleep(RECOVERY_SLEEP_SECONDS)


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
    parser.add_argument("--put-only", action="store_true",
                         help="Skip CALL signals entirely, only trade PUT (default: off, both directions "
                              "allowed). CALL's confidence hasn't tracked real accuracy as well as PUT's in "
                              "backtest or live results - available as an option, not the default.")
    args = parser.parse_args()
    run(
        max_minutes=args.max_minutes,
        signal_source=args.signal_source,
        max_trades_per_day=args.max_trades_per_day,
        max_capital_per_trade=args.max_capital_per_trade,
        put_only=args.put_only,
    )
