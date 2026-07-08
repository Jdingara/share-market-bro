"""Unit tests for ml_features.py against small, hand-checkable synthetic data."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ml_features import FEATURE_NAMES, extract_features


def _make_daily_df(n=55, close=100.0, last_high=110.0, last_low=90.0):
    rows = [{"date": pd.Timestamp("2026-01-01") + timedelta(days=i), "open": close, "high": close, "low": close, "close": close, "volume": 0} for i in range(n)]
    rows[-1]["high"] = last_high
    rows[-1]["low"] = last_low
    return pd.DataFrame(rows)


def _make_intraday_df(n=30, start_price=100.0):
    base = datetime(2026, 7, 7, 9, 15)
    rows = []
    price = start_price
    for i in range(n):
        rows.append({
            "date": pd.Timestamp(base + timedelta(minutes=15 * i)),
            "open": price, "high": price + 0.5, "low": price - 0.5, "close": price, "volume": 0,
        })
    return pd.DataFrame(rows)


def test_returns_all_expected_feature_names():
    daily_df = _make_daily_df()
    intraday_df = _make_intraday_df()
    features = extract_features(daily_df, intraday_df, index=20)
    assert set(features.keys()) == set(FEATURE_NAMES)


def test_fibonacci_distances_hand_computed():
    # last day's high=110, low=90 -> 38.2%=102.36, 50%=100, 61.8%=97.64
    daily_df = _make_daily_df(last_high=110.0, last_low=90.0)
    intraday_df = _make_intraday_df(start_price=100.0)  # price is flat at 100 throughout
    features = extract_features(daily_df, intraday_df, index=20)

    assert abs(features["fib_dist_50_0"] - 0.0) < 1e-9  # price 100 exactly at the 50% level
    assert features["fib_dist_38_2"] < 0  # price 100 is below the 38.2% level (102.36)
    assert features["fib_dist_61_8"] > 0  # price 100 is above the 61.8% level (97.64)


def test_ema_trend_ratio_near_zero_for_flat_closes():
    daily_df = _make_daily_df(close=100.0)
    intraday_df = _make_intraday_df()
    features = extract_features(daily_df, intraday_df, index=20)
    assert abs(features["ema_trend_ratio"]) < 1e-9


def test_day_of_week_and_minutes_since_open():
    daily_df = _make_daily_df()
    intraday_df = _make_intraday_df()  # starts 2026-07-07 09:15, a Tuesday
    features = extract_features(daily_df, intraday_df, index=4)  # 4 candles * 15min = 60 min later
    assert features["minutes_since_open"] == 60.0
    assert features["day_of_week"] == 1  # Tuesday = 1 (Monday=0)


def test_candlestick_flags_detect_bullish_engulfing():
    daily_df = _make_daily_df()
    intraday_df = _make_intraday_df(n=25)
    # Overwrite two rows to form a clean bullish engulfing at index 20.
    intraday_df.loc[19, ["open", "close"]] = [101.0, 99.0]  # bearish candle
    intraday_df.loc[20, ["open", "close"]] = [98.0, 102.0]  # engulfing bullish candle
    features = extract_features(daily_df, intraday_df, index=20)
    assert features["is_bullish_engulfing"] == 1.0
