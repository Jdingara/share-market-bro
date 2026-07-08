"""Unit tests for option_lookup.py, using a fake instruments list (no live API needed)."""

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from option_lookup import OptionLookupError, find_nearest_valid_expiry, find_option_instrument

FAKE_INSTRUMENTS = [
    {"name": "NIFTY", "segment": "NFO-OPT", "strike": 24000.0, "expiry": date(2026, 7, 16), "instrument_type": "CE",
     "tradingsymbol": "NIFTY26071624000CE", "instrument_token": 111},
    {"name": "NIFTY", "segment": "NFO-OPT", "strike": 24000.0, "expiry": date(2026, 7, 16), "instrument_type": "PE",
     "tradingsymbol": "NIFTY26071624000PE", "instrument_token": 222},
    {"name": "NIFTY", "segment": "NFO-OPT", "strike": 24050.0, "expiry": date(2026, 7, 16), "instrument_type": "CE",
     "tradingsymbol": "NIFTY26071624050CE", "instrument_token": 333},
    {"name": "BANKNIFTY", "segment": "NFO-OPT", "strike": 24000.0, "expiry": date(2026, 7, 16), "instrument_type": "CE",
     "tradingsymbol": "BANKNIFTY26071624000CE", "instrument_token": 444},
]

FAKE_MULTI_EXPIRY_INSTRUMENTS = [
    {"name": "NIFTY", "segment": "NFO-OPT", "expiry": date(2026, 7, 7)},   # too close (0 days out)
    {"name": "NIFTY", "segment": "NFO-OPT", "expiry": date(2026, 7, 14)},  # 7 days out - valid
    {"name": "NIFTY", "segment": "NFO-OPT", "expiry": date(2026, 7, 21)},  # further out - also valid
    {"name": "BANKNIFTY", "segment": "NFO-OPT", "expiry": date(2026, 7, 8)},  # different underlying, must be ignored
]


def test_finds_matching_call():
    result = find_option_instrument(FAKE_INSTRUMENTS, strike=24000.0, expiry=date(2026, 7, 16), option_type="CE")
    assert result["tradingsymbol"] == "NIFTY26071624000CE"


def test_finds_matching_put():
    result = find_option_instrument(FAKE_INSTRUMENTS, strike=24000.0, expiry=date(2026, 7, 16), option_type="PE")
    assert result["tradingsymbol"] == "NIFTY26071624000PE"


def test_does_not_match_wrong_underlying_with_same_strike_expiry_type():
    # BANKNIFTY has an entry with the identical strike/expiry/type - must not be picked for NIFTY.
    result = find_option_instrument(FAKE_INSTRUMENTS, strike=24000.0, expiry=date(2026, 7, 16), option_type="CE", name="NIFTY")
    assert result["tradingsymbol"] == "NIFTY26071624000CE"


def test_raises_when_no_match():
    with pytest.raises(OptionLookupError):
        find_option_instrument(FAKE_INSTRUMENTS, strike=99999.0, expiry=date(2026, 7, 16), option_type="CE")


def test_find_nearest_valid_expiry_skips_ones_too_close():
    result = find_nearest_valid_expiry(FAKE_MULTI_EXPIRY_INSTRUMENTS, as_of_date=date(2026, 7, 7), min_days=3)
    assert result == date(2026, 7, 14)


def test_find_nearest_valid_expiry_ignores_other_underlyings():
    # BANKNIFTY has a closer expiry (2026-07-08) that must not leak into a NIFTY lookup.
    result = find_nearest_valid_expiry(FAKE_MULTI_EXPIRY_INSTRUMENTS, as_of_date=date(2026, 7, 7), min_days=3, name="NIFTY")
    assert result == date(2026, 7, 14)


def test_find_nearest_valid_expiry_raises_when_none_far_enough():
    with pytest.raises(OptionLookupError):
        find_nearest_valid_expiry(FAKE_MULTI_EXPIRY_INSTRUMENTS, as_of_date=date(2026, 7, 7), min_days=30)
