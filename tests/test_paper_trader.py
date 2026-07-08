"""Unit tests for paper_trader.py's pure decision logic (no live API needed)."""

import sys
from datetime import datetime, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paper_trader import check_exit_condition

BEFORE_CUTOFF = datetime(2026, 7, 7, 13, 0)
AFTER_CUTOFF = datetime(2026, 7, 7, 15, 26)


def test_target_hit():
    assert check_exit_condition(100.0, 110.0, BEFORE_CUTOFF) == "TARGET"


def test_stoploss_hit():
    assert check_exit_condition(100.0, 90.0, BEFORE_CUTOFF) == "STOPLOSS"


def test_no_exit_when_within_bracket_and_before_cutoff():
    assert check_exit_condition(100.0, 103.0, BEFORE_CUTOFF) is None


def test_forced_close_after_cutoff_time():
    assert check_exit_condition(100.0, 103.0, AFTER_CUTOFF) == "EOD_CLOSE"


def test_stoploss_takes_priority_over_forced_close_if_both_true():
    assert check_exit_condition(100.0, 89.0, AFTER_CUTOFF) == "STOPLOSS"


def test_custom_force_close_time():
    # 14:30 is before the default 15:25 cutoff (would be None normally) but after a custom 14:00 cutoff.
    at_1430 = datetime(2026, 7, 7, 14, 30)
    early_cutoff = time(14, 0)
    assert check_exit_condition(100.0, 103.0, at_1430, force_close_time=early_cutoff) == "EOD_CLOSE"
