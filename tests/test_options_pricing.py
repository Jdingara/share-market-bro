"""
Unit tests for options_pricing.py. Where possible these check exact
model-independent identities (like put-call parity) rather than relying on
memorized reference decimal values, since those are easy to misremember and
would create false failures.
"""

import math
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from options_pricing import (
    black_scholes_price,
    historical_volatility,
    nearest_strike,
    next_weekly_expiry,
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


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
