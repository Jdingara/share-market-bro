"""
The confluence rule: decides once per trading day whether to buy a call, buy
a put, or take no trade at all.

Contract for callers (both the future backtester and paper trader must
follow this to avoid lookahead bias):
  - `daily_df` must contain only days strictly BEFORE the day being traded,
    sorted ascending by date. This is what makes the trend filter and
    Fibonacci zones legitimate - they can only see what was actually known
    before today's session started.
  - `intraday_df` must contain only the 15-minute candles for the single day
    being evaluated, sorted ascending by time, exactly as they would have
    arrived in real time.

All the thresholds below are named constants specifically so Phase 3
(backtesting) can find and tune them against measured historical win rates,
rather than guessing at good values now.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Literal, Optional

import pandas as pd

from indicators import (
    bollinger_bands,
    ema,
    fibonacci_levels,
    is_bearish_engulfing,
    is_bullish_engulfing,
    is_hammer,
    is_shooting_star,
    rsi,
)

EMA_FAST_PERIOD = 20
EMA_SLOW_PERIOD = 50
TREND_NEUTRAL_BAND_PCT = 0.001  # EMA20/EMA50 within 0.1% of each other -> no clear trend

RSI_PERIOD = 14
RSI_TURN_LOOKBACK = 2  # candles back to compare RSI against, to detect a "turn"
RSI_BULLISH_CEILING = 60  # don't buy calls if RSI already this high (little room left to run)
RSI_BEARISH_FLOOR = 40  # don't buy puts if RSI already this low (little room left to fall)

FIB_PROXIMITY_PCT = 0.003  # price must be within 0.3% of a fib level to count as "at" it
SIGNAL_CUTOFF_TIME = time(15, 0)  # stop looking for new entries after 3:00 PM

# Bollinger Bands are a second, independent way to flag "price is at an extreme" -
# dynamic/volatility-based (computed from today's own intraday action), unlike
# Fibonacci's static levels from yesterday's range. Either counts as a qualifying zone.
BB_WINDOW = 20
BB_NUM_STD = 2.0

# Candlestick confirmation doesn't have to land on the exact same candle as the fib+RSI
# alignment - a confirming candle shortly after is still a real confirmation. Checked as
# "any of the last N candles up to and including this one", not just the current one.
CANDLE_CONFIRM_LOOKBACK = 2


@dataclass
class Signal:
    direction: Literal["CALL", "PUT", "NO_TRADE"]
    timestamp: Optional[pd.Timestamp]
    trigger_price: Optional[float]
    fib_level: Optional[str]
    reasoning: str


def _as_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series) if not pd.api.types.is_datetime64_any_dtype(series) else series


def _bullish_candle_confirms(df: pd.DataFrame, end_index: int, lookback: int) -> bool:
    """True if a bullish engulfing or hammer appeared on any candle in [end_index - lookback, end_index]."""
    for j in range(max(end_index - lookback, 1), end_index + 1):
        prev_row, curr_row = df.iloc[j - 1], df.iloc[j]
        if is_bullish_engulfing(prev_row["open"], prev_row["close"], curr_row["open"], curr_row["close"]):
            return True
        if is_hammer(curr_row["open"], curr_row["high"], curr_row["low"], curr_row["close"]):
            return True
    return False


def _bearish_candle_confirms(df: pd.DataFrame, end_index: int, lookback: int) -> bool:
    """True if a bearish engulfing or shooting star appeared on any candle in [end_index - lookback, end_index]."""
    for j in range(max(end_index - lookback, 1), end_index + 1):
        prev_row, curr_row = df.iloc[j - 1], df.iloc[j]
        if is_bearish_engulfing(prev_row["open"], prev_row["close"], curr_row["open"], curr_row["close"]):
            return True
        if is_shooting_star(curr_row["open"], curr_row["high"], curr_row["low"], curr_row["close"]):
            return True
    return False


def daily_trend_bias(daily_df: pd.DataFrame) -> Literal["bullish", "bearish", "neutral"]:
    """EMA(20) vs EMA(50) on daily closes, as of the last row in daily_df."""
    closes = daily_df["close"]
    ema_fast = ema(closes, EMA_FAST_PERIOD).iloc[-1]
    ema_slow = ema(closes, EMA_SLOW_PERIOD).iloc[-1]

    if abs(ema_fast - ema_slow) / ema_slow <= TREND_NEUTRAL_BAND_PCT:
        return "neutral"
    return "bullish" if ema_fast > ema_slow else "bearish"


def fib_zones_for_today(daily_df: pd.DataFrame) -> dict[str, float]:
    """Fibonacci levels from the most recent completed day's high-to-low range."""
    previous_day = daily_df.iloc[-1]
    return fibonacci_levels(high=previous_day["high"], low=previous_day["low"])


def generate_signal(daily_df: pd.DataFrame, intraday_df: pd.DataFrame) -> Signal:
    trend = daily_trend_bias(daily_df)
    if trend == "neutral":
        return Signal("NO_TRADE", None, None, None, "No clear daily trend bias (EMA20/EMA50 too close)")

    fib_levels = fib_zones_for_today(daily_df)

    intraday_df = intraday_df.reset_index(drop=True)
    timestamps = _as_datetime(intraday_df["date"])
    closes = intraday_df["close"]
    rsi_values = rsi(closes, period=RSI_PERIOD)
    _, bb_upper, bb_lower = bollinger_bands(closes, window=BB_WINDOW, num_std=BB_NUM_STD)

    start_index = RSI_PERIOD + RSI_TURN_LOOKBACK
    for i in range(start_index, len(intraday_df)):
        ts = timestamps.iloc[i]
        if ts.time() > SIGNAL_CUTOFF_TIME:
            break

        current_price = closes.iloc[i]
        current_rsi = rsi_values.iloc[i]
        prior_rsi = rsi_values.iloc[i - RSI_TURN_LOOKBACK]

        zone_hit = next(
            (
                name
                for name, price in fib_levels.items()
                if abs(current_price - price) / price <= FIB_PROXIMITY_PCT
            ),
            None,
        )
        if zone_hit is None:
            lower, upper = bb_lower.iloc[i], bb_upper.iloc[i]
            if trend == "bullish" and pd.notna(lower) and current_price <= lower:
                zone_hit = "BB_lower"
            elif trend == "bearish" and pd.notna(upper) and current_price >= upper:
                zone_hit = "BB_upper"
        if zone_hit is None:
            continue

        if trend == "bullish":
            rsi_confirms = current_rsi > prior_rsi and current_rsi < RSI_BULLISH_CEILING
            candle_confirms = _bullish_candle_confirms(intraday_df, i, CANDLE_CONFIRM_LOOKBACK)

            if rsi_confirms and candle_confirms:
                return Signal(
                    "CALL",
                    ts,
                    current_price,
                    zone_hit,
                    f"Bullish trend + price at {zone_hit} zone + RSI turning up "
                    f"({prior_rsi:.1f}->{current_rsi:.1f}) + confirming candle",
                )

        else:  # bearish
            rsi_confirms = current_rsi < prior_rsi and current_rsi > RSI_BEARISH_FLOOR
            candle_confirms = _bearish_candle_confirms(intraday_df, i, CANDLE_CONFIRM_LOOKBACK)

            if rsi_confirms and candle_confirms:
                return Signal(
                    "PUT",
                    ts,
                    current_price,
                    zone_hit,
                    f"Bearish trend + price at {zone_hit} zone + RSI turning down "
                    f"({prior_rsi:.1f}->{current_rsi:.1f}) + confirming candle",
                )

    return Signal("NO_TRADE", None, None, None, f"No qualifying confluence found before {SIGNAL_CUTOFF_TIME} (trend was {trend})")
