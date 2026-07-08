"""
Generic technical-analysis math: EMA, RSI, Bollinger Bands, Fibonacci
retracement levels, and a handful of candlestick pattern detectors.

Deliberately has no trading logic or knowledge of "today"/"signals" - that
lives in signal_engine.py, which combines these building blocks into an
actual buy/sell decision. Keeping this module purely mathematical makes each
piece independently testable against hand-computed values.
"""

from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI (the standard RSI formula, using Wilder's smoothing)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Rolling mean +/- num_std standard deviations. Returns (middle, upper, lower)."""
    middle = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return middle, upper, lower


def fibonacci_levels(high: float, low: float) -> dict[str, float]:
    """Standard retracement levels (38.2%, 50%, 61.8%) between a high and low."""
    diff = high - low
    return {
        "38.2": high - 0.382 * diff,
        "50.0": high - 0.5 * diff,
        "61.8": high - 0.618 * diff,
    }


def is_bullish_engulfing(prev_open: float, prev_close: float, curr_open: float, curr_close: float) -> bool:
    """Current bullish candle's body fully engulfs the prior bearish candle's body."""
    prev_bearish = prev_close < prev_open
    curr_bullish = curr_close > curr_open
    engulfs = curr_open <= prev_close and curr_close >= prev_open
    return prev_bearish and curr_bullish and engulfs


def is_bearish_engulfing(prev_open: float, prev_close: float, curr_open: float, curr_close: float) -> bool:
    """Current bearish candle's body fully engulfs the prior bullish candle's body."""
    prev_bullish = prev_close > prev_open
    curr_bearish = curr_close < curr_open
    engulfs = curr_open >= prev_close and curr_close <= prev_open
    return prev_bullish and curr_bearish and engulfs


def is_hammer(open_: float, high: float, low: float, close: float) -> bool:
    """Small body near the top of the range, long lower wick, little/no upper wick."""
    body = abs(close - open_)
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low
    total_range = high - low
    if total_range == 0 or body == 0:
        return False
    return lower_wick >= 2 * body and upper_wick <= 0.1 * total_range


def is_shooting_star(open_: float, high: float, low: float, close: float) -> bool:
    """Small body near the bottom of the range, long upper wick, little/no lower wick."""
    body = abs(close - open_)
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low
    total_range = high - low
    if total_range == 0 or body == 0:
        return False
    return upper_wick >= 2 * body and lower_wick <= 0.1 * total_range
