"""
Replays the signal engine over cached historical data, one trading day at a
time, simulating the +10%/-10% option-premium bracket via options_pricing.py
(see that module's docstring for why premiums are simulated rather than
real - Kite doesn't retain historical data for expired option contracts).

Respects the same no-lookahead contract as signal_engine.generate_signal:
on each simulated day, the trend/Fibonacci inputs only ever see days
strictly before that day, and the intraday walk only sees that day's own
candles up to whatever point it's evaluating.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from options_pricing import black_scholes_price, historical_volatility, nearest_strike, next_weekly_expiry
from signal_engine import generate_signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DAILY_CSV = PROJECT_ROOT / "data" / "historical" / "NIFTY_50_day.csv"
INTRADAY_CSV = PROJECT_ROOT / "data" / "historical" / "NIFTY_50_15minute.csv"
RESULTS_DIR = PROJECT_ROOT / "data" / "backtest_results"

MIN_DAILY_HISTORY = 55  # enough rows for a meaningful EMA(50) and a 20-day volatility window
MARKET_CLOSE_TIME = time(15, 30)
TARGET_PCT = 0.10
STOP_LOSS_PCT = 0.10
VOLATILITY_WINDOW = 20
SECONDS_PER_YEAR = 365 * 24 * 3600


@dataclass
class TradeResult:
    date: str
    direction: str
    zone: str  # which zone triggered entry: a fib level name, or "BB_lower"/"BB_upper"
    entry_time: str
    entry_premium: float
    strike: float
    expiry: str
    exit_time: str
    exit_premium: float
    exit_reason: str  # "TARGET" | "STOPLOSS" | "EOD_CLOSE"
    pct_change: float


def _time_to_expiry_years(as_of: datetime, expiry_date) -> float:
    expiry_dt = datetime.combine(expiry_date, MARKET_CLOSE_TIME)
    as_of_naive = as_of.replace(tzinfo=None)
    return max((expiry_dt - as_of_naive).total_seconds(), 0.0) / SECONDS_PER_YEAR


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_df = pd.read_csv(DAILY_CSV)
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df = daily_df.sort_values("date").reset_index(drop=True)

    intraday_df = pd.read_csv(INTRADAY_CSV)
    intraday_df["date"] = pd.to_datetime(intraday_df["date"])
    intraday_df = intraday_df.sort_values("date").reset_index(drop=True)
    return daily_df, intraday_df


def simulate_trade(
    direction: str,
    zone: str,
    entry_time: datetime,
    entry_spot: float,
    daily_history: pd.DataFrame,
    remaining_intraday: pd.DataFrame,
) -> TradeResult:
    option_type = "CE" if direction == "CALL" else "PE"
    strike = nearest_strike(entry_spot)
    expiry = next_weekly_expiry(entry_time.date())
    volatility = historical_volatility(daily_history["close"], window=VOLATILITY_WINDOW)

    entry_tte = _time_to_expiry_years(entry_time, expiry)
    entry_premium = black_scholes_price(entry_spot, strike, entry_tte, volatility, option_type)

    exit_time, exit_premium, exit_reason = entry_time, entry_premium, "EOD_CLOSE"

    for _, candle in remaining_intraday.iterrows():
        candle_time = candle["date"].to_pydatetime()
        tte = _time_to_expiry_years(candle_time, expiry)

        # A real stop/target order fills near the trigger price, not wherever the
        # underlying happens to be when we next check - so check the candle's full
        # high/low range, not just its close, and exit AT the threshold if crossed.
        premium_at_high = black_scholes_price(candle["high"], strike, tte, volatility, option_type)
        premium_at_low = black_scholes_price(candle["low"], strike, tte, volatility, option_type)
        best_premium, worst_premium = (
            (premium_at_high, premium_at_low) if option_type == "CE" else (premium_at_low, premium_at_high)
        )
        worst_pct = (worst_premium - entry_premium) / entry_premium
        best_pct = (best_premium - entry_premium) / entry_premium

        # If both the stop and target were theoretically crossable within this single
        # candle, we can't know the true intra-candle order without tick data - assume
        # the stop hit first (the conservative assumption, not the favorable one).
        if worst_pct <= -STOP_LOSS_PCT:
            exit_time, exit_reason = candle_time, "STOPLOSS"
            exit_premium = entry_premium * (1 - STOP_LOSS_PCT)
            break
        if best_pct >= TARGET_PCT:
            exit_time, exit_reason = candle_time, "TARGET"
            exit_premium = entry_premium * (1 + TARGET_PCT)
            break

        exit_time = candle_time
        exit_premium = black_scholes_price(candle["close"], strike, tte, volatility, option_type)
    else:
        exit_reason = "EOD_CLOSE"

    pct_change = (exit_premium - entry_premium) / entry_premium
    return TradeResult(
        date=entry_time.date().isoformat(),
        direction=direction,
        zone=zone,
        entry_time=entry_time.isoformat(),
        entry_premium=round(entry_premium, 2),
        strike=strike,
        expiry=expiry.isoformat(),
        exit_time=exit_time.isoformat(),
        exit_premium=round(exit_premium, 2),
        exit_reason=exit_reason,
        pct_change=round(pct_change, 4),
    )


def run_backtest(signal_fn=generate_signal, start_date=None, end_date=None) -> list[TradeResult]:
    """Replays signal_fn (any function with generate_signal's exact interface)
    day-by-day. Passing start_date/end_date restricts which days are replayed -
    used to run the same held-out period through a different signal_fn for an
    apples-to-apples comparison (see ml_signal.py)."""
    daily_df, intraday_df = _load_data()
    intraday_df["trading_day"] = intraday_df["date"].dt.date

    all_days = sorted(intraday_df["trading_day"].unique())
    full_day_candle_count = intraday_df["trading_day"].value_counts().median()
    complete_days = [
        d for d in all_days if (intraday_df["trading_day"] == d).sum() >= full_day_candle_count * 0.8
    ]
    if start_date is not None:
        complete_days = [d for d in complete_days if d >= start_date]
    if end_date is not None:
        complete_days = [d for d in complete_days if d <= end_date]

    results: list[TradeResult] = []

    for day in complete_days:
        daily_history = daily_df[daily_df["date"].dt.date < day]
        if len(daily_history) < MIN_DAILY_HISTORY:
            continue

        day_intraday = intraday_df[intraday_df["trading_day"] == day].drop(columns=["trading_day"]).reset_index(drop=True)

        signal = signal_fn(daily_history, day_intraday)
        if signal.direction == "NO_TRADE":
            results.append(
                TradeResult(
                    date=day.isoformat(), direction="NO_TRADE", zone="", entry_time="", entry_premium=0,
                    strike=0, expiry="", exit_time="", exit_premium=0, exit_reason="", pct_change=0,
                )
            )
            continue

        entry_time = signal.timestamp.to_pydatetime()
        remaining_intraday = day_intraday[day_intraday["date"] > signal.timestamp]

        trade = simulate_trade(
            signal.direction, signal.fib_level, entry_time, signal.trigger_price, daily_history, remaining_intraday
        )
        results.append(trade)

    return results


def summarize(results: list[TradeResult]) -> None:
    total_days = len(results)
    trades = [r for r in results if r.direction != "NO_TRADE"]
    no_trade_days = total_days - len(trades)

    wins = [t for t in trades if t.exit_reason == "TARGET"]
    losses = [t for t in trades if t.exit_reason == "STOPLOSS"]
    eod = [t for t in trades if t.exit_reason == "EOD_CLOSE"]

    print(f"Days evaluated:  {total_days}")
    print(f"No-trade days:   {no_trade_days}")
    print(f"Trades taken:    {len(trades)}")
    if trades:
        win_rate = len(wins) / len(trades) * 100
        avg_pct = sum(t.pct_change for t in trades) / len(trades) * 100
        print(f"  Hit +10% target:  {len(wins)}")
        print(f"  Hit -10% stop:    {len(losses)}")
        print(f"  Closed at EOD:    {len(eod)}")
        print(f"  Win rate (target vs stop, excl. EOD closes): "
              f"{len(wins) / max(len(wins) + len(losses), 1) * 100:.1f}%")
        print(f"  Overall win rate (target hits / all trades): {win_rate:.1f}%")
        print(f"  Average P&L per trade: {avg_pct:+.2f}%")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = run_backtest()

    out_path = RESULTS_DIR / "trades.csv"
    pd.DataFrame([asdict(r) for r in results]).to_csv(out_path, index=False)
    print(f"Full day-by-day results saved to {out_path}\n")

    summarize(results)


if __name__ == "__main__":
    main()
