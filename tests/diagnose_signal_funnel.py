"""
Not a test - a one-off diagnostic to see which confluence condition is the
actual bottleneck (trend / fib proximity / RSI turn / candlestick), rather
than guessing which threshold to loosen after a low trade count.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indicators import is_bearish_engulfing, is_bullish_engulfing, is_hammer, is_shooting_star, rsi
from signal_engine import (
    FIB_PROXIMITY_PCT,
    RSI_BEARISH_FLOOR,
    RSI_BULLISH_CEILING,
    RSI_PERIOD,
    RSI_TURN_LOOKBACK,
    SIGNAL_CUTOFF_TIME,
    daily_trend_bias,
    fib_zones_for_today,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
daily_df = pd.read_csv(PROJECT_ROOT / "data" / "historical" / "NIFTY_50_day.csv")
daily_df["date"] = pd.to_datetime(daily_df["date"])
daily_df = daily_df.sort_values("date").reset_index(drop=True)

intraday_df = pd.read_csv(PROJECT_ROOT / "data" / "historical" / "NIFTY_50_15minute.csv")
intraday_df["date"] = pd.to_datetime(intraday_df["date"])
intraday_df = intraday_df.sort_values("date").reset_index(drop=True)
intraday_df["trading_day"] = intraday_df["date"].dt.date

all_days = sorted(intraday_df["trading_day"].unique())
full_day_candle_count = intraday_df["trading_day"].value_counts().median()
complete_days = [d for d in all_days if (intraday_df["trading_day"] == d).sum() >= full_day_candle_count * 0.8]

counts = {"candles_checked": 0, "neutral_trend_days": 0, "fib_proximity_hit": 0, "rsi_confirms": 0, "candle_confirms": 0}
trending_days = 0

for day in complete_days:
    daily_history = daily_df[daily_df["date"].dt.date < day]
    if len(daily_history) < 55:
        continue

    trend = daily_trend_bias(daily_history)
    if trend == "neutral":
        counts["neutral_trend_days"] += 1
        continue
    trending_days += 1

    fib_levels = fib_zones_for_today(daily_history)
    day_intraday = intraday_df[intraday_df["trading_day"] == day].drop(columns=["trading_day"]).reset_index(drop=True)
    closes = day_intraday["close"]
    rsi_values = rsi(closes, period=RSI_PERIOD)
    timestamps = day_intraday["date"]

    start = RSI_PERIOD + RSI_TURN_LOOKBACK
    for i in range(start, len(day_intraday)):
        if timestamps.iloc[i].time() > SIGNAL_CUTOFF_TIME:
            break
        counts["candles_checked"] += 1

        price = closes.iloc[i]
        matching_level = next((n for n, p in fib_levels.items() if abs(price - p) / p <= FIB_PROXIMITY_PCT), None)
        if matching_level is None:
            continue
        counts["fib_proximity_hit"] += 1

        current_rsi, prior_rsi = rsi_values.iloc[i], rsi_values.iloc[i - RSI_TURN_LOOKBACK]
        if trend == "bullish":
            rsi_ok = current_rsi > prior_rsi and current_rsi < RSI_BULLISH_CEILING
        else:
            rsi_ok = current_rsi < prior_rsi and current_rsi > RSI_BEARISH_FLOOR
        if not rsi_ok:
            continue
        counts["rsi_confirms"] += 1

        prev_row, curr_row = day_intraday.iloc[i - 1], day_intraday.iloc[i]
        if trend == "bullish":
            candle_ok = is_bullish_engulfing(prev_row["open"], prev_row["close"], curr_row["open"], curr_row["close"]) or is_hammer(
                curr_row["open"], curr_row["high"], curr_row["low"], curr_row["close"]
            )
        else:
            candle_ok = is_bearish_engulfing(prev_row["open"], prev_row["close"], curr_row["open"], curr_row["close"]) or is_shooting_star(
                curr_row["open"], curr_row["high"], curr_row["low"], curr_row["close"]
            )
        if candle_ok:
            counts["candle_confirms"] += 1

print(f"Trending days (bullish/bearish): {trending_days}  |  Neutral days: {counts['neutral_trend_days']}")
print(f"Total candles checked (across trending days): {counts['candles_checked']}")
print(f"  -> candles where price was AT a fib level:        {counts['fib_proximity_hit']}")
print(f"  -> ...of those, where RSI also confirmed:          {counts['rsi_confirms']}")
print(f"  -> ...of those, where candlestick ALSO confirmed:  {counts['candle_confirms']}  <- this is what fires a trade")
