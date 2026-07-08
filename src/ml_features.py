"""
Feature engineering for the ML signal engine: turns a point in time (one
intraday candle, with its preceding history) into a numeric feature vector.

Deliberately pure/stateless - no training or prediction logic here, so each
feature can be tested independently. Reuses the same indicator building
blocks as the rule-based signal_engine.py, on purpose: the ML model should
be learning nuanced *combinations* of these same well-understood signals,
not inventing an unrelated set of inputs that would make an honest
comparison between the two approaches meaningless.
"""

from __future__ import annotations

from datetime import datetime, time

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

RSI_PERIOD = 14
RSI_TURN_LOOKBACK = 2
EMA_FAST_PERIOD = 20
EMA_SLOW_PERIOD = 50
BB_WINDOW = 20
BB_NUM_STD = 2.0
MOMENTUM_LOOKBACK = 3
MARKET_OPEN_TIME = time(9, 15)

FEATURE_NAMES = [
    "rsi_value",
    "rsi_change",
    "fib_dist_38_2",
    "fib_dist_50_0",
    "fib_dist_61_8",
    "bb_dist_upper",
    "bb_dist_lower",
    "ema_trend_ratio",
    "momentum",
    "realized_vol",
    "is_bullish_engulfing",
    "is_bearish_engulfing",
    "is_hammer",
    "is_shooting_star",
    "minutes_since_open",
    "day_of_week",
]


def extract_features(daily_df: pd.DataFrame, intraday_df: pd.DataFrame, index: int) -> dict[str, float]:
    """Feature vector for intraday_df.iloc[index], using only data up to and
    including that row (plus daily_df, which must already be strictly prior
    days per signal_engine's no-lookahead contract)."""
    closes = intraday_df["close"]
    price = closes.iloc[index]

    rsi_values = rsi(closes, period=RSI_PERIOD)
    rsi_value = rsi_values.iloc[index]
    lookback_index = max(index - RSI_TURN_LOOKBACK, 0)
    rsi_change = rsi_value - rsi_values.iloc[lookback_index]

    previous_day = daily_df.iloc[-1]
    fib_levels = fibonacci_levels(high=previous_day["high"], low=previous_day["low"])
    fib_dist_38_2 = (price - fib_levels["38.2"]) / fib_levels["38.2"]
    fib_dist_50_0 = (price - fib_levels["50.0"]) / fib_levels["50.0"]
    fib_dist_61_8 = (price - fib_levels["61.8"]) / fib_levels["61.8"]

    _, bb_upper, bb_lower = bollinger_bands(closes, window=BB_WINDOW, num_std=BB_NUM_STD)
    upper, lower = bb_upper.iloc[index], bb_lower.iloc[index]
    bb_dist_upper = (price - upper) / price if pd.notna(upper) else 0.0
    bb_dist_lower = (price - lower) / price if pd.notna(lower) else 0.0

    daily_closes = daily_df["close"]
    ema_fast = ema(daily_closes, EMA_FAST_PERIOD).iloc[-1]
    ema_slow = ema(daily_closes, EMA_SLOW_PERIOD).iloc[-1]
    ema_trend_ratio = (ema_fast - ema_slow) / ema_slow

    momentum_index = max(index - MOMENTUM_LOOKBACK, 0)
    momentum = (price - closes.iloc[momentum_index]) / closes.iloc[momentum_index]

    returns = closes.iloc[max(index - BB_WINDOW, 0) : index + 1].pct_change()
    realized_vol = returns.std() if len(returns) > 1 else 0.0

    prev_row = intraday_df.iloc[max(index - 1, 0)]
    curr_row = intraday_df.iloc[index]
    bullish_engulfing = is_bullish_engulfing(prev_row["open"], prev_row["close"], curr_row["open"], curr_row["close"])
    bearish_engulfing = is_bearish_engulfing(prev_row["open"], prev_row["close"], curr_row["open"], curr_row["close"])
    hammer = is_hammer(curr_row["open"], curr_row["high"], curr_row["low"], curr_row["close"])
    shooting_star = is_shooting_star(curr_row["open"], curr_row["high"], curr_row["low"], curr_row["close"])

    timestamp = pd.Timestamp(intraday_df["date"].iloc[index]).to_pydatetime()
    timestamp_naive = timestamp.replace(tzinfo=None)
    minutes_since_open = (timestamp_naive - datetime.combine(timestamp_naive.date(), MARKET_OPEN_TIME)).total_seconds() / 60

    return {
        "rsi_value": rsi_value,
        "rsi_change": rsi_change,
        "fib_dist_38_2": fib_dist_38_2,
        "fib_dist_50_0": fib_dist_50_0,
        "fib_dist_61_8": fib_dist_61_8,
        "bb_dist_upper": bb_dist_upper,
        "bb_dist_lower": bb_dist_lower,
        "ema_trend_ratio": ema_trend_ratio,
        "momentum": momentum,
        "realized_vol": realized_vol,
        "is_bullish_engulfing": float(bullish_engulfing),
        "is_bearish_engulfing": float(bearish_engulfing),
        "is_hammer": float(hammer),
        "is_shooting_star": float(shooting_star),
        "minutes_since_open": minutes_since_open,
        "day_of_week": float(timestamp.weekday()),
    }
