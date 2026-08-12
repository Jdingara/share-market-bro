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
from options_pricing import black_scholes_greeks, nearest_strike, next_weekly_expiry, time_to_expiry_years

RSI_PERIOD = 14
RSI_TURN_LOOKBACK = 2
EMA_FAST_PERIOD = 20
EMA_SLOW_PERIOD = 50
BB_WINDOW = 20
BB_NUM_STD = 2.0
MOMENTUM_LOOKBACK = 3
MARKET_OPEN_TIME = time(9, 15)

# India VIX features (added 2026-08-12, first piece of the CALL-improvement plan
# agreed 2026-07-28/08-06). VIX_LOOKBACK reuses MOMENTUM_LOOKBACK so "recent VIX
# trend" is measured over the same window as price momentum, for consistency.
# VIX_PCTILE_WINDOW reuses BB_WINDOW (20 trailing days) for the same reason -
# a regime-detection window that's already an established convention here,
# not a newly-guessed number.
VIX_LOOKBACK = MOMENTUM_LOOKBACK
VIX_PCTILE_WINDOW = BB_WINDOW
VIX_PCTILE_MIN_HISTORY = 5  # below this, a percentile isn't meaningful - fall back to neutral (0.5)

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
    "vix_level",
    "vix_change",
    "vix_pctile",
    "atm_gamma",
    "atm_theta",
    "atm_vega",
]


def attach_vix(price_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """Left-joins India VIX closes onto price_df (daily or intraday, whichever
    is being prepared) by 'date', as a new 'vix_close' column - the shape
    extract_features expects. Left join, not inner: a NIFTY row must never be
    dropped just because a VIX candle happened to be missing that timestamp -
    extract_features falls back to neutral defaults wherever vix_close ends up
    NaN (or is absent altogether, e.g. in tests that don't care about VIX)."""
    vix_slim = vix_df[["date", "close"]].rename(columns={"close": "vix_close"})
    return price_df.merge(vix_slim, on="date", how="left")


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

    # VIX features: current live level, its recent trend, and where that level
    # sits relative to the trailing window - all computed only from data up to
    # and including this candle (intraday) or strictly prior days (daily), same
    # no-lookahead contract as everything else here. Falls back to neutral
    # defaults (0.0 / 0.0 / 0.5) rather than raising when vix_close isn't
    # present at all - keeps this function usable on data that hasn't had VIX
    # merged in (e.g. existing tests, or a caller that skips it deliberately).
    has_vix = "vix_close" in intraday_df.columns and pd.notna(intraday_df["vix_close"].iloc[index])
    if has_vix:
        vix_series = intraday_df["vix_close"]
        vix_level = vix_series.iloc[index]
        vix_lookback_index = max(index - VIX_LOOKBACK, 0)
        prior_vix = vix_series.iloc[vix_lookback_index]
        vix_change = (vix_level - prior_vix) if pd.notna(prior_vix) else 0.0

        vix_daily_hist = daily_df["vix_close"].dropna() if "vix_close" in daily_df.columns else pd.Series(dtype=float)
        vix_daily_hist = vix_daily_hist.iloc[-VIX_PCTILE_WINDOW:]
        vix_pctile = (vix_daily_hist < vix_level).mean() if len(vix_daily_hist) >= VIX_PCTILE_MIN_HISTORY else 0.5
    else:
        vix_level, vix_change, vix_pctile = 0.0, 0.0, 0.5

    # Options Greeks (added 2026-08-12, second piece of the CALL-improvement plan
    # alongside VIX): computed via Black-Scholes at the strike the bot would
    # actually trade if it signaled here (nearest_strike to the current price),
    # using the live India VIX reading as the volatility input - VIX literally
    # IS the market's own implied-vol estimate, a more honest input here than a
    # lagging realized-vol calculation would be. Reuses has_vix/vix_level from
    # the block above: without a real live VIX reading there's no sound
    # volatility input to price Greeks with, so these default to neutral (0.0)
    # too rather than guessing one. Only gamma/theta/vega are included, not
    # delta - an ATM option's delta sits close to 0.5 by construction (that's
    # what "at the money" means) and is already largely redundant with the
    # existing fib/momentum features; gamma/theta/vega each carry genuinely new
    # information (convexity, time-decay pressure, vol-sensitivity) that no
    # other feature here captures. Priced as a call throughout (gamma and vega
    # are identical for a call and a put at the same strike/expiry, by put-call
    # parity - see black_scholes_greeks), so this one calculation covers both
    # the call_model and put_model consumers of this same feature vector.
    if has_vix:
        atm_strike = nearest_strike(price)
        atm_expiry = next_weekly_expiry(timestamp_naive.date())
        atm_tte = time_to_expiry_years(timestamp_naive, atm_expiry)
        greeks = black_scholes_greeks(price, atm_strike, atm_tte, vix_level / 100, "CE")
        atm_gamma = greeks["gamma"]
        atm_theta = greeks["theta"]
        atm_vega = greeks["vega"]
    else:
        atm_gamma, atm_theta, atm_vega = 0.0, 0.0, 0.0

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
        "vix_level": vix_level,
        "vix_change": vix_change,
        "vix_pctile": vix_pctile,
        "atm_gamma": atm_gamma,
        "atm_theta": atm_theta,
        "atm_vega": atm_vega,
    }
