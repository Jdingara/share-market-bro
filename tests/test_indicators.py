"""
Unit tests for indicators.py against hand-computed / analytically-known values.

These check the math is right in isolation, before it becomes load-bearing
for actual trading decisions in signal_engine.py.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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


def test_ema_matches_hand_computed_value():
    # span=3 -> alpha = 2/(3+1) = 0.5. EMA[0]=10, EMA[1]=0.5*20+0.5*10=15.
    series = pd.Series([10, 20])
    result = ema(series, period=3)
    assert result.iloc[0] == 10
    assert result.iloc[1] == 15


def test_ema_of_constant_series_equals_the_constant():
    series = pd.Series([42.0] * 10)
    result = ema(series, period=5)
    assert (result == 42.0).all()


def test_rsi_is_100_when_all_moves_are_gains():
    series = pd.Series(range(1, 20))  # strictly increasing -> no losses at all
    result = rsi(series, period=14)
    assert result.iloc[-1] == 100.0


def test_rsi_is_0_when_all_moves_are_losses():
    series = pd.Series(range(20, 1, -1))  # strictly decreasing -> no gains at all
    result = rsi(series, period=14)
    assert result.iloc[-1] == 0.0


def test_bollinger_bands_of_constant_series_has_zero_width():
    series = pd.Series([100.0] * 25)
    middle, upper, lower = bollinger_bands(series, window=20, num_std=2)
    assert middle.iloc[-1] == 100.0
    assert upper.iloc[-1] == 100.0
    assert lower.iloc[-1] == 100.0


def test_bollinger_bands_hand_computed():
    # window=3, last 3 values = [10, 20, 30] -> mean=20, population std via pandas (ddof=1) = 10
    series = pd.Series([10, 20, 30])
    middle, upper, lower = bollinger_bands(series, window=3, num_std=2)
    assert middle.iloc[-1] == 20.0
    assert upper.iloc[-1] == 40.0
    assert lower.iloc[-1] == 0.0


def test_fibonacci_levels_hand_computed():
    levels = fibonacci_levels(high=200, low=100)
    assert levels["38.2"] == 161.8
    assert levels["50.0"] == 150.0
    assert levels["61.8"] == 138.2


def test_bullish_engulfing_detected():
    # prev bearish (10 -> 8), curr bullish engulfing it (7 -> 11)
    assert is_bullish_engulfing(prev_open=10, prev_close=8, curr_open=7, curr_close=11) is True


def test_bullish_engulfing_not_detected_when_body_too_small():
    assert is_bullish_engulfing(prev_open=10, prev_close=8, curr_open=8.5, curr_close=9.5) is False


def test_bearish_engulfing_detected():
    # prev bullish (8 -> 10), curr bearish engulfing it (11 -> 7)
    assert is_bearish_engulfing(prev_open=8, prev_close=10, curr_open=11, curr_close=7) is True


def test_hammer_detected():
    assert is_hammer(open_=10, high=10.6, low=8, close=10.5) is True


def test_hammer_not_detected_for_long_upper_wick():
    assert is_hammer(open_=10, high=12, low=9.9, close=10.5) is False


def test_shooting_star_detected():
    assert is_shooting_star(open_=10, high=12, low=9.4, close=9.5) is True


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
