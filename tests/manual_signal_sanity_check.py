"""
Not an automated test - a manual sanity check for signal_engine.py against
real cached data. Prints the trend bias, fib levels, and resulting signal
for each of the last several complete trading days, so a human can eyeball
whether the output looks like a reasonable read of the actual market.

Run after fetching real daily + 15minute data via data_fetch.py.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from signal_engine import daily_trend_bias, fib_zones_for_today, generate_signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DAILY_CSV = PROJECT_ROOT / "data" / "historical" / "NIFTY_50_day.csv"
INTRADAY_CSV = PROJECT_ROOT / "data" / "historical" / "NIFTY_50_15minute.csv"

DAYS_TO_CHECK = 10

daily_df = pd.read_csv(DAILY_CSV)
daily_df["date"] = pd.to_datetime(daily_df["date"])
daily_df = daily_df.sort_values("date").reset_index(drop=True)

intraday_df = pd.read_csv(INTRADAY_CSV)
intraday_df["date"] = pd.to_datetime(intraday_df["date"])
intraday_df = intraday_df.sort_values("date").reset_index(drop=True)
intraday_df["trading_day"] = intraday_df["date"].dt.date

all_days = sorted(intraday_df["trading_day"].unique())
# Drop the last day if it's not a full session (fewer candles than a typical day).
full_day_candle_count = intraday_df["trading_day"].value_counts().median()
complete_days = [d for d in all_days if (intraday_df["trading_day"] == d).sum() >= full_day_candle_count * 0.8]

days_to_test = complete_days[-DAYS_TO_CHECK:]

for day in days_to_test:
    day_intraday = intraday_df[intraday_df["trading_day"] == day].drop(columns=["trading_day"])
    day_daily_history = daily_df[daily_df["date"].dt.date < day]

    if len(day_daily_history) < 5:
        print(f"{day}: skipped, not enough prior daily history")
        continue

    trend = daily_trend_bias(day_daily_history)
    fibs = fib_zones_for_today(day_daily_history)
    signal = generate_signal(day_daily_history, day_intraday)

    print(f"\n=== {day} ===")
    print(f"  trend bias: {trend}")
    print(f"  fib levels: {', '.join(f'{k}%={v:.1f}' for k, v in fibs.items())}")
    print(f"  day range:  low={day_intraday['low'].min():.1f} high={day_intraday['high'].max():.1f}")
    print(f"  signal:     {signal.direction} | {signal.reasoning}")
    if signal.direction != "NO_TRADE":
        print(f"              trigger @ {signal.timestamp} price={signal.trigger_price:.1f} level={signal.fib_level}%")
