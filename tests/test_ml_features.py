"""Unit tests for ml_features.py against small, hand-checkable synthetic data."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ml_features import FEATURE_NAMES, attach_vix, extract_features


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


def test_vix_features_default_to_neutral_when_not_merged():
    # No vix_close column at all (e.g. a caller that never merged VIX in) -
    # must not crash, and must return clearly neutral, non-informative values.
    daily_df = _make_daily_df()
    intraday_df = _make_intraday_df()
    features = extract_features(daily_df, intraday_df, index=20)
    assert features["vix_level"] == 0.0
    assert features["vix_change"] == 0.0
    assert features["vix_pctile"] == 0.5


def test_vix_level_and_change_hand_computed():
    daily_df = _make_daily_df()
    intraday_df = _make_intraday_df(n=25)
    intraday_df["vix_close"] = 12.0  # flat baseline
    intraday_df.loc[20, "vix_close"] = 14.5  # spiked at the candle under test
    # index 20, VIX_LOOKBACK=3 -> compares against index 17, still 12.0
    features = extract_features(daily_df, intraday_df, index=20)
    assert features["vix_level"] == 14.5
    assert abs(features["vix_change"] - 2.5) < 1e-9


def test_vix_pctile_hand_computed():
    daily_df = _make_daily_df(n=25)
    # Trailing 20 daily VIX closes, all below the current live level of 20.0
    # except one right at it - current level should rank near (but not at) the top.
    daily_df["vix_close"] = [10.0 + i * 0.2 for i in range(len(daily_df))]  # last 20 values: 10.8..14.6-ish range
    intraday_df = _make_intraday_df()
    intraday_df["vix_close"] = 20.0  # comfortably above every historical daily close
    features = extract_features(daily_df, intraday_df, index=20)
    assert features["vix_pctile"] == 1.0  # every one of the trailing 20 daily closes was below it


def test_vix_pctile_neutral_with_too_little_daily_history():
    daily_df = _make_daily_df(n=3)  # fewer than VIX_PCTILE_MIN_HISTORY (5)
    daily_df["vix_close"] = 12.0
    intraday_df = _make_intraday_df()
    intraday_df["vix_close"] = 15.0
    features = extract_features(daily_df, intraday_df, index=20)
    assert features["vix_pctile"] == 0.5


def test_attach_vix_left_joins_by_date_without_dropping_rows():
    price_df = _make_daily_df(n=5)
    vix_df = price_df.copy()
    vix_df["close"] = [11.0, 12.0, 13.0, 14.0, 15.0]
    merged = attach_vix(price_df, vix_df)
    assert len(merged) == len(price_df)  # no rows lost
    assert list(merged["vix_close"]) == [11.0, 12.0, 13.0, 14.0, 15.0]


def test_attach_vix_leaves_nan_for_a_missing_vix_date_rather_than_dropping_the_row():
    price_df = _make_daily_df(n=3)
    vix_df = price_df.iloc[:2].copy()  # missing the 3rd day's VIX candle entirely
    vix_df["close"] = [11.0, 12.0]
    merged = attach_vix(price_df, vix_df)
    assert len(merged) == 3  # the NIFTY row is still there
    assert pd.isna(merged["vix_close"].iloc[2])


def test_greeks_default_to_neutral_when_vix_not_merged():
    daily_df = _make_daily_df()
    intraday_df = _make_intraday_df()  # no vix_close column at all
    features = extract_features(daily_df, intraday_df, index=20)
    assert features["atm_gamma"] == 0.0
    assert features["atm_theta"] == 0.0
    assert features["atm_vega"] == 0.0


def test_greeks_are_computed_when_vix_is_present():
    daily_df = _make_daily_df()
    intraday_df = _make_intraday_df(n=25)
    intraday_df["vix_close"] = 12.0
    # index 20 -> 2026-07-07 09:15 + 20*15min = 12:15, comfortably mid-session,
    # plenty of real (non-zero) time to expiry for the nearest weekly expiry.
    features = extract_features(daily_df, intraday_df, index=20)
    assert features["atm_gamma"] > 0
    assert features["atm_vega"] > 0
    assert features["atm_theta"] < 0  # time decay: should be negative for a long option
