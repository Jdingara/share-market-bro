"""
Approximates what an option's premium would have been, since Kite Connect
does not retain historical data for expired option contracts (confirmed
directly: the earliest listed expiry in the live instrument dump is today -
there is no way to fetch real historical premiums for past months).

Uses a standard Black-Scholes model driven by the real underlying NIFTY
price history, with *historical/realized* volatility standing in for the
implied volatility options actually trade on. These are related but not
identical - implied vol usually runs a bit higher - so treat results built
on this module as directionally useful for filtering strategies, not a
precise prediction of real P&L. Real premiums only enter the picture in
Phase 4 (paper trading), via Kite's live option quotes.
"""

from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta
from typing import Literal

import pandas as pd

RISK_FREE_RATE = 0.065  # approximate India risk-free rate; short-dated weekly options are not very sensitive to this
NIFTY_STRIKE_INTERVAL = 50
NIFTY_LOT_SIZE = 65  # confirmed live against Kite's instrument dump (not 50, not 40 - both common guesses are wrong)
MARKET_CLOSE_TIME = time(15, 30)
SECONDS_PER_YEAR = 365 * 24 * 3600
WEEKLY_EXPIRY_WEEKDAY = 1  # Tuesday (Monday=0) - confirmed live against Kite's instrument list on 2026-07-08;
# NSE has changed NIFTY's weekly expiry day before (was Thursday) and may again - this is only an
# approximation for backtesting/simulation (no real historical expiry calendar is fetchable, same
# reason expired contracts aren't - see PROJECT_STATUS.md). Live trading does NOT rely on this guess -
# option_lookup.find_nearest_valid_expiry() queries the real listed expiries instead (see paper_trader.py).


def _standard_normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry_years: float,
    volatility: float,
    option_type: Literal["CE", "PE"],
    risk_free_rate: float = RISK_FREE_RATE,
) -> float:
    """Standard closed-form Black-Scholes price. Falls back to intrinsic value at/after expiry."""
    if time_to_expiry_years <= 0:
        return max(spot - strike, 0.0) if option_type == "CE" else max(strike - spot, 0.0)

    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry_years) / (
        volatility * math.sqrt(time_to_expiry_years)
    )
    d2 = d1 - volatility * math.sqrt(time_to_expiry_years)
    discount = math.exp(-risk_free_rate * time_to_expiry_years)

    if option_type == "CE":
        return spot * _standard_normal_cdf(d1) - strike * discount * _standard_normal_cdf(d2)
    return strike * discount * _standard_normal_cdf(-d2) - spot * _standard_normal_cdf(-d1)


def _standard_normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)


def time_to_expiry_years(as_of: datetime, expiry_date: date, market_close_time: time = MARKET_CLOSE_TIME) -> float:
    """Time remaining until an option's expiry, in years - shared by backtester.py
    (bracket simulation) and ml_features.py (Greeks features) so both compute
    time-to-expiry exactly the same way. Lives here, not in either caller, so
    neither module has to import the other (backtester.py already imports from
    ml_features.py for VIX merging - a reverse import would be circular)."""
    expiry_dt = datetime.combine(expiry_date, market_close_time)
    as_of_naive = as_of.replace(tzinfo=None)
    return max((expiry_dt - as_of_naive).total_seconds(), 0.0) / SECONDS_PER_YEAR


def black_scholes_greeks(
    spot: float,
    strike: float,
    time_to_expiry_years: float,
    volatility: float,
    option_type: Literal["CE", "PE"],
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict[str, float]:
    """Standard closed-form Greeks, same d1/d2 as black_scholes_price (same
    inputs, same assumptions - see module docstring). Theta is returned per
    CALENDAR DAY (annual/365), the more directly useful unit for "how much
    value does one more day cost" as an ML feature, not the raw per-year
    convention some references use. Vega is per 1-percentage-point change in
    volatility, the standard convention. Falls back to all-zeros at/after
    expiry or for non-positive volatility - Greeks are undefined right at
    expiry, and zero is the reasonable "no more time value left" answer."""
    if time_to_expiry_years <= 0 or volatility <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    sqrt_t = math.sqrt(time_to_expiry_years)
    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry_years) / (
        volatility * sqrt_t
    )
    d2 = d1 - volatility * sqrt_t
    pdf_d1 = _standard_normal_pdf(d1)
    discount = math.exp(-risk_free_rate * time_to_expiry_years)

    # Gamma and vega are identical for a call and a put at the same strike/expiry
    # (put-call parity) - only delta and theta actually differ by option_type.
    gamma = pdf_d1 / (spot * volatility * sqrt_t)
    vega = spot * pdf_d1 * sqrt_t / 100

    if option_type == "CE":
        delta = _standard_normal_cdf(d1)
        theta_annual = -(spot * pdf_d1 * volatility) / (2 * sqrt_t) - risk_free_rate * strike * discount * _standard_normal_cdf(d2)
    else:
        delta = _standard_normal_cdf(d1) - 1
        theta_annual = -(spot * pdf_d1 * volatility) / (2 * sqrt_t) + risk_free_rate * strike * discount * _standard_normal_cdf(-d2)

    return {"delta": delta, "gamma": gamma, "theta": theta_annual / 365, "vega": vega}


def historical_volatility(daily_closes: pd.Series, window: int = 20) -> float:
    """Annualized volatility from a rolling window of daily log returns, as of the last close."""
    log_returns = (daily_closes / daily_closes.shift(1)).apply(math.log)
    rolling_std = log_returns.rolling(window=window).std()
    return rolling_std.iloc[-1] * math.sqrt(252)


def nearest_strike(spot: float, interval: float = NIFTY_STRIKE_INTERVAL) -> float:
    return round(spot / interval) * interval


MIN_DAYS_TO_EXPIRY = 3  # avoid 0-2 DTE options - confirmed empirically (Phase 3 backtest) that their
# extreme gamma/theta noise can swing premium +/-10% on a near-flat underlying move, swamping any
# real directional signal


def next_weekly_expiry(from_date: date, min_days: int = MIN_DAYS_TO_EXPIRY) -> date:
    """Nearest Thursday on/after from_date that is at least min_days away, skipping to the
    following week if the immediate one is too close (approximation - the real historical
    expiry calendar isn't fetchable for the same reason expired contracts aren't)."""
    days_until = (WEEKLY_EXPIRY_WEEKDAY - from_date.weekday()) % 7
    expiry = from_date + timedelta(days=days_until)
    if (expiry - from_date).days < min_days:
        expiry += timedelta(days=7)
    return expiry
