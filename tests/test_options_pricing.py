"""
Unit tests for options_pricing.py. Where possible these check exact
model-independent identities (like put-call parity) rather than relying on
memorized reference decimal values, since those are easy to misremember and
would create false failures.
"""

import math
import sys
from datetime import date, datetime, time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from options_pricing import (
    black_scholes_greeks,
    black_scholes_price,
    historical_volatility,
    nearest_strike,
    next_weekly_expiry,
    time_to_expiry_years,
)

SPOT = 24000.0
STRIKE = 24000.0
TIME_TO_EXPIRY = 7 / 365
VOLATILITY = 0.12
RISK_FREE_RATE = 0.065


def test_put_call_parity_holds():
    # Exact identity for Black-Scholes, independent of any specific reference values: C - P = S - K*e^(-rT)
    call = black_scholes_price(SPOT, STRIKE, TIME_TO_EXPIRY, VOLATILITY, "CE", RISK_FREE_RATE)
    put = black_scholes_price(SPOT, STRIKE, TIME_TO_EXPIRY, VOLATILITY, "PE", RISK_FREE_RATE)
    expected_diff = SPOT - STRIKE * math.exp(-RISK_FREE_RATE * TIME_TO_EXPIRY)
    assert abs((call - put) - expected_diff) < 1e-6


def test_price_at_expiry_equals_intrinsic_value():
    itm_call = black_scholes_price(spot=24100, strike=24000, time_to_expiry_years=0, volatility=0.12, option_type="CE")
    assert itm_call == 100.0

    otm_call = black_scholes_price(spot=23900, strike=24000, time_to_expiry_years=0, volatility=0.12, option_type="CE")
    assert otm_call == 0.0

    itm_put = black_scholes_price(spot=23900, strike=24000, time_to_expiry_years=0, volatility=0.12, option_type="PE")
    assert itm_put == 100.0


def test_call_price_is_at_least_intrinsic_value():
    call = black_scholes_price(spot=24500, strike=24000, time_to_expiry_years=TIME_TO_EXPIRY, volatility=VOLATILITY, option_type="CE")
    assert call >= 500.0


def test_higher_volatility_means_higher_premium():
    low_vol_price = black_scholes_price(SPOT, STRIKE, TIME_TO_EXPIRY, volatility=0.10, option_type="CE")
    high_vol_price = black_scholes_price(SPOT, STRIKE, TIME_TO_EXPIRY, volatility=0.25, option_type="CE")
    assert high_vol_price > low_vol_price


def test_historical_volatility_of_flat_series_is_zero():
    flat_prices = pd.Series([100.0] * 30)
    assert historical_volatility(flat_prices, window=20) == 0.0


def test_historical_volatility_is_positive_for_varying_series():
    varying_prices = pd.Series([100, 102, 99, 103, 101, 104, 98, 105, 100, 103] * 3)
    vol = historical_volatility(varying_prices, window=20)
    assert vol > 0


def test_nearest_strike_rounds_to_interval():
    assert nearest_strike(24012) == 24000
    assert nearest_strike(24039) == 24050


def test_next_weekly_expiry_from_monday_rolls_when_too_close():
    # Monday -> nearest Tuesday is only 1 day away, below the default 3-day minimum, so it rolls over.
    monday = date(2026, 7, 6)
    assert next_weekly_expiry(monday) == date(2026, 7, 14)


def test_next_weekly_expiry_skips_to_next_week_when_too_close():
    # Tuesday itself is 0 days to expiry - below the default 3-day minimum, so it must roll over.
    tuesday = date(2026, 7, 7)
    assert next_weekly_expiry(tuesday) == date(2026, 7, 14)


def test_next_weekly_expiry_respects_custom_min_days():
    saturday = date(2026, 7, 4)  # exactly 3 days to the following Tuesday
    assert next_weekly_expiry(saturday, min_days=3) == date(2026, 7, 7)  # exactly 3 days - not too close
    assert next_weekly_expiry(saturday, min_days=4) == date(2026, 7, 14)  # now too close, rolls over


def test_next_weekly_expiry_from_wednesday_is_far_enough_already():
    wednesday = date(2026, 7, 8)  # nearest Tuesday is 6 days away - comfortably past the minimum
    assert next_weekly_expiry(wednesday) == date(2026, 7, 14)


def test_time_to_expiry_years_at_market_close_on_expiry_day_is_zero():
    expiry = date(2026, 7, 14)
    as_of = datetime.combine(expiry, time(15, 30))
    assert time_to_expiry_years(as_of, expiry) == 0.0


def test_time_to_expiry_years_never_negative_past_expiry():
    expiry = date(2026, 7, 14)
    as_of = datetime.combine(expiry, time(16, 0))  # after market close, past expiry
    assert time_to_expiry_years(as_of, expiry) == 0.0


def test_time_to_expiry_years_one_week_out():
    as_of = datetime(2026, 7, 7, 15, 30)
    expiry = date(2026, 7, 14)
    assert abs(time_to_expiry_years(as_of, expiry) - 7 / 365) < 1e-9


def test_greeks_are_all_zero_at_expiry():
    greeks = black_scholes_greeks(SPOT, STRIKE, 0.0, VOLATILITY, "CE")
    assert greeks == {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}


def test_greeks_are_all_zero_for_non_positive_volatility():
    greeks = black_scholes_greeks(SPOT, STRIKE, TIME_TO_EXPIRY, 0.0, "CE")
    assert greeks == {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}


def test_atm_call_delta_is_near_half():
    # A textbook property: an at-the-money call's delta sits close to 0.5.
    greeks = black_scholes_greeks(SPOT, STRIKE, TIME_TO_EXPIRY, VOLATILITY, "CE")
    assert 0.4 < greeks["delta"] < 0.6


def test_put_delta_equals_call_delta_minus_one():
    # Exact identity, independent of any reference values.
    call_greeks = black_scholes_greeks(SPOT, STRIKE, TIME_TO_EXPIRY, VOLATILITY, "CE")
    put_greeks = black_scholes_greeks(SPOT, STRIKE, TIME_TO_EXPIRY, VOLATILITY, "PE")
    assert abs(put_greeks["delta"] - (call_greeks["delta"] - 1)) < 1e-9


def test_gamma_and_vega_identical_for_call_and_put():
    # Put-call parity: gamma/vega don't depend on option_type at the same strike/expiry.
    call_greeks = black_scholes_greeks(SPOT, STRIKE, TIME_TO_EXPIRY, VOLATILITY, "CE")
    put_greeks = black_scholes_greeks(SPOT, STRIKE, TIME_TO_EXPIRY, VOLATILITY, "PE")
    assert abs(call_greeks["gamma"] - put_greeks["gamma"]) < 1e-9
    assert abs(call_greeks["vega"] - put_greeks["vega"]) < 1e-9


def test_gamma_and_vega_are_positive_for_a_real_option():
    greeks = black_scholes_greeks(SPOT, STRIKE, TIME_TO_EXPIRY, VOLATILITY, "CE")
    assert greeks["gamma"] > 0
    assert greeks["vega"] > 0


def test_theta_is_negative_for_a_long_option_near_the_money():
    # Time decay: holding an ATM option one more day should lose value, not gain it.
    greeks = black_scholes_greeks(SPOT, STRIKE, TIME_TO_EXPIRY, VOLATILITY, "CE")
    assert greeks["theta"] < 0


def test_gamma_rises_as_expiry_approaches_for_an_atm_option():
    # The 0DTE gamma-spike effect already documented in PROJECT_STATUS.md (Phase 3) -
    # this is the Greek meant to capture it as an ML feature.
    far_greeks = black_scholes_greeks(SPOT, STRIKE, 7 / 365, VOLATILITY, "CE")
    near_greeks = black_scholes_greeks(SPOT, STRIKE, 1 / 365, VOLATILITY, "CE")
    assert near_greeks["gamma"] > far_greeks["gamma"]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
