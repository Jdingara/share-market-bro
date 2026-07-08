"""Unit tests for capital_manager.py, using the exact numbers already discussed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import capital_manager
from capital_manager import apply_trade_pnl, calculate_affordable_lots, deployable_capital, load_capital, save_capital

PREMIUM = 265.25  # today's real first trade


def test_20000_affords_exactly_1_lot():
    assert calculate_affordable_lots(20000.0, PREMIUM) == 1


def test_10000_affords_zero_lots():
    assert calculate_affordable_lots(10000.0, PREMIUM) == 0


def test_15000_affords_zero_lots():
    # 1 lot costs 265.25 * 65 = 17241.25 - still more than 15000.
    assert calculate_affordable_lots(15000.0, PREMIUM) == 0


def test_35000_affords_2_lots():
    assert calculate_affordable_lots(35000.0, PREMIUM) == 2


def test_apply_trade_pnl_matches_todays_real_trade():
    # Real trade: entry 265.25, exit 298.2, 1 lot, starting from 20000.
    new_capital = apply_trade_pnl(20000.0, lots=1, entry_premium=265.25, exit_premium=298.2)
    assert round(new_capital, 2) == 22141.75


def test_apply_trade_pnl_leaves_idle_capital_untouched():
    # 20000 capital, only 1 lot affordable (uses 17241.25), leftover 2758.75 must survive unchanged
    # even after a loss on the invested portion.
    capital_before = 20000.0
    lots = calculate_affordable_lots(capital_before, PREMIUM)
    losing_exit = PREMIUM * 0.90  # a -10% stop-loss
    new_capital = apply_trade_pnl(capital_before, lots, PREMIUM, losing_exit)
    idle = capital_before - (lots * 65 * PREMIUM)
    assert new_capital == idle + (lots * 65 * losing_exit)


def test_deployable_capital_below_cap_is_unaffected():
    assert deployable_capital(22141.75, max_per_trade=200000.0) == 22141.75


def test_deployable_capital_above_cap_is_limited():
    assert deployable_capital(700000.0, max_per_trade=200000.0) == 200000.0


def test_deployable_capital_exactly_at_cap():
    assert deployable_capital(200000.0, max_per_trade=200000.0) == 200000.0


def test_load_capital_defaults_when_no_state_file(tmp_path, monkeypatch):
    monkeypatch.setattr(capital_manager, "CAPITAL_STATE_PATH", tmp_path / "capital_state.json")
    assert load_capital() == capital_manager.STARTING_CAPITAL


def test_save_then_load_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(capital_manager, "CAPITAL_STATE_PATH", tmp_path / "capital_state.json")
    save_capital(22141.75)
    assert load_capital() == 22141.75
