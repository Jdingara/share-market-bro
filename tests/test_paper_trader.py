"""Unit tests for paper_trader.py's pure decision logic (no live API needed)."""

import sys
from datetime import date, datetime, time
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from kiteconnect.exceptions import TokenException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import os

from paper_trader import (
    LOCK_FILE_PATH,
    MIN_CANDLES_FOR_SIGNAL,
    SESSION_SPLIT_TIME,
    DuplicateProcessError,
    _acquire_lock,
    _build_early_session_signal_fn,
    _circuit_breaker_tripped,
    _current_session,
    _is_process_alive,
    _is_stale_signal,
    _kill_switch_engaged,
    _monitor_position_until_exit,
    _reauthenticate,
    _record_trade_slot,
    _release_lock,
    _todays_realized_pnl,
    _trade_slot_available,
    check_exit_condition,
)

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


def test_stoploss_at_tightened_5pct_boundary():
    # -5% is the current stop (tightened 2026-07-10 from -10%, target stays +10%) - confirms
    # a drop that the old -10% rule would have ridden out now correctly stops early.
    assert check_exit_condition(100.0, 95.0, BEFORE_CUTOFF) == "STOPLOSS"


def test_no_stoploss_just_above_5pct_boundary():
    assert check_exit_condition(100.0, 96.0, BEFORE_CUTOFF) is None


def test_min_candles_for_signal_matches_rsi_warmup():
    # RSI_PERIOD (14) + RSI_TURN_LOOKBACK (2) - the hard floor before any
    # signal (rule-based or ML) can be computed at all, regardless of confidence.
    assert MIN_CANDLES_FOR_SIGNAL == 16


def test_no_early_session_model_for_non_gradient_boosting_sources():
    # Only gradient_boosting has a trained 5-min counterpart (train_5min_model.py) -
    # everything else should behave exactly as before (no early-session fallback).
    assert _build_early_session_signal_fn("rule_based") is None
    assert _build_early_session_signal_fn("random_forest") is None
    assert _build_early_session_signal_fn("logistic_regression") is None


def test_early_session_model_available_for_gradient_boosting():
    assert _build_early_session_signal_fn("gradient_boosting") is not None


MORNING_TIME = datetime(2026, 7, 16, 11, 0)
AFTERNOON_TIME = datetime(2026, 7, 16, 14, 0)
RIGHT_AT_SPLIT = datetime.combine(datetime(2026, 7, 16).date(), SESSION_SPLIT_TIME)


def test_current_session_before_and_after_split():
    assert _current_session(MORNING_TIME) == "morning"
    assert _current_session(AFTERNOON_TIME) == "afternoon"
    assert _current_session(RIGHT_AT_SPLIT) == "afternoon"  # >= split time counts as afternoon


def test_trade_slot_available_flat_mode_ignores_sessions():
    # split_session=False - only trades_taken_today vs max_trades_per_day matters,
    # regardless of time of day or session_slots contents.
    assert _trade_slot_available(MORNING_TIME, 5, {"morning": 99, "afternoon": 99}, max_trades_per_day=6,
                                  max_trades_per_session=6, split_session=False)
    assert not _trade_slot_available(AFTERNOON_TIME, 6, {"morning": 0, "afternoon": 0}, max_trades_per_day=6,
                                      max_trades_per_session=6, split_session=False)


def test_trade_slot_available_split_mode_checks_current_session_only():
    # Morning quota full, afternoon quota empty - no slot in the morning...
    assert not _trade_slot_available(MORNING_TIME, 0, {"morning": 6, "afternoon": 0}, max_trades_per_day=1,
                                      max_trades_per_session=6, split_session=True)
    # ...but once the clock is in the afternoon window, the (separate, still-empty)
    # afternoon quota is what's checked, not the exhausted morning one.
    assert _trade_slot_available(AFTERNOON_TIME, 0, {"morning": 6, "afternoon": 0}, max_trades_per_day=1,
                                  max_trades_per_session=6, split_session=True)
    assert not _trade_slot_available(AFTERNOON_TIME, 0, {"morning": 6, "afternoon": 6}, max_trades_per_day=1,
                                      max_trades_per_session=6, split_session=True)


def test_record_trade_slot_flat_mode_increments_shared_counter():
    session_slots = {"morning": 0, "afternoon": 0}
    result = _record_trade_slot(MORNING_TIME, 3, session_slots, split_session=False)
    assert result == 4
    assert session_slots == {"morning": 0, "afternoon": 0}  # untouched in flat mode


def test_record_trade_slot_split_mode_increments_correct_session_only():
    session_slots = {"morning": 0, "afternoon": 0}
    unchanged = _record_trade_slot(MORNING_TIME, 3, session_slots, split_session=True)
    assert unchanged == 3  # trades_taken_today isn't used in split mode
    assert session_slots == {"morning": 1, "afternoon": 0}

    _record_trade_slot(AFTERNOON_TIME, 3, session_slots, split_session=True)
    assert session_slots == {"morning": 1, "afternoon": 1}


def test_is_stale_signal_before_any_entry_is_never_stale():
    # No prior entry yet this run - nothing to compare against, so never stale.
    assert not _is_stale_signal(datetime(2026, 7, 16, 10, 35), None)


def test_is_stale_signal_same_candle_as_last_entry():
    last_candle = datetime(2026, 7, 16, 10, 45)
    assert _is_stale_signal(last_candle, last_candle)


def test_is_stale_signal_older_candle_than_last_entry():
    # Shouldn't be possible in practice (candles only move forward), but the
    # guard should still treat it as stale rather than allow a re-entry.
    assert _is_stale_signal(datetime(2026, 7, 16, 10, 40), datetime(2026, 7, 16, 10, 45))


def test_is_stale_signal_fresh_candle_after_last_entry():
    assert not _is_stale_signal(datetime(2026, 7, 16, 10, 50), datetime(2026, 7, 16, 10, 45))


def test_is_process_alive_true_for_current_process():
    assert _is_process_alive(os.getpid())


def test_is_process_alive_false_for_unlikely_pid():
    # A PID this high is very unlikely to be in use - not a live-process guarantee,
    # but a reasonable smoke test that _is_process_alive doesn't just always return True.
    assert not _is_process_alive(999_999_999)


def test_acquire_and_release_lock_round_trip(tmp_path):
    lock_path = tmp_path / "paper_trader.lock"
    _acquire_lock(lock_path)
    assert lock_path.exists()
    assert int(lock_path.read_text().strip()) == os.getpid()
    _release_lock(lock_path)
    assert not lock_path.exists()


def test_acquire_lock_allows_restart_after_own_pid(tmp_path):
    # Re-acquiring with the SAME pid already in the lock file (e.g. re-entering run()
    # in some edge case) must not raise - a process can't be "duplicate" of itself.
    lock_path = tmp_path / "paper_trader.lock"
    lock_path.write_text(str(os.getpid()))
    _acquire_lock(lock_path)  # should not raise
    _release_lock(lock_path)


def test_acquire_lock_refuses_when_pid_in_lock_is_alive(tmp_path):
    lock_path = tmp_path / "paper_trader.lock"
    other_alive_pid = os.getppid()  # the parent process - guaranteed alive during this test
    lock_path.write_text(str(other_alive_pid))
    try:
        with pytest.raises(DuplicateProcessError):
            _acquire_lock(lock_path)
    finally:
        lock_path.unlink(missing_ok=True)  # test-only cleanup, not the real _release_lock path


def test_acquire_lock_succeeds_when_lock_is_stale(tmp_path):
    # Lock file exists but names a PID that's no longer running - a crashed/force-killed
    # process left this behind, and a new instance should be allowed to start.
    lock_path = tmp_path / "paper_trader.lock"
    lock_path.write_text("999999999")
    _acquire_lock(lock_path)  # should not raise - stale lock is overwritten
    assert int(lock_path.read_text().strip()) == os.getpid()
    _release_lock(lock_path)


def test_reauthenticate_reads_the_cache_not_a_forced_fresh_login():
    # Flipped 2026-07-30: force=True (the 2026-07-27 fix) tried an automated
    # fresh login every time, but Zerodha started requiring a CAPTCHA on this
    # account's login page, which that flow can't solve - so force=True just
    # fails every retry, repeatedly hitting the CAPTCHA-protected endpoint
    # without ever recovering. force=False reads whatever's cached, which
    # manual_login.py (a human solving the CAPTCHA once) can update - this
    # confirms _reauthenticate never falls back to the automated flow.
    with patch("paper_trader.login") as mock_login:
        _reauthenticate()
        mock_login.assert_called_once_with(force=False)


def test_monitor_position_survives_token_exception_without_losing_the_position():
    # Found live 2026-07-28: a TokenException mid-monitoring used to escape this
    # loop entirely and propagate to the caller's outer handler, which
    # reauthenticates but then returns to the top of the main loop - forgetting
    # this open position ever existed and letting the bot open a SECOND one on
    # top of it. This confirms recovery now happens IN PLACE, still watching the
    # same position, instead.
    call_count = {"n": 0}

    def fake_get_option_premium(kite, tradingsymbol):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise TokenException("Incorrect `api_key` or `access_token`.")
        return 110.0  # +10% from entry_premium=100.0 below -> TARGET

    fake_new_kite = object()
    with patch("paper_trader.time_module.sleep"), \
         patch("paper_trader.get_option_premium", side_effect=fake_get_option_premium), \
         patch("paper_trader._reauthenticate", return_value=fake_new_kite) as mock_reauth:
        kite, exit_reason, exit_premium = _monitor_position_until_exit("old_kite", "NIFTY123PE", 100.0)

    assert exit_reason == "TARGET"
    assert exit_premium == 110.0
    assert kite is fake_new_kite  # the loop picked up the reauthenticated client
    mock_reauth.assert_called_once()


def test_monitor_position_survives_a_failed_reauth_attempt_too():
    # Even if the reauth attempt itself fails, the loop must not escape - it
    # should keep watching the same position and retry on the next poll.
    call_count = {"n": 0}

    def fake_get_option_premium(kite, tradingsymbol):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            raise TokenException("Incorrect `api_key` or `access_token`.")
        return 95.0  # -5% from entry_premium=100.0 below -> STOPLOSS

    reauth_calls = {"n": 0}

    def flaky_reauth():
        reauth_calls["n"] += 1
        if reauth_calls["n"] == 1:
            raise TokenException("still broken")
        return "recovered_kite"

    with patch("paper_trader.time_module.sleep"), \
         patch("paper_trader.get_option_premium", side_effect=fake_get_option_premium), \
         patch("paper_trader._reauthenticate", side_effect=flaky_reauth):
        kite, exit_reason, exit_premium = _monitor_position_until_exit("old_kite", "NIFTY123PE", 100.0)

    assert exit_reason == "STOPLOSS"
    assert exit_premium == 95.0
    assert kite == "recovered_kite"
    assert reauth_calls["n"] == 2


def test_monitor_position_force_closes_immediately_when_kill_switch_engaged():
    # Must not wait for target/stop-loss/EOD once the kill switch is on - the whole
    # point is an emergency stop that can't get stuck behind a position's normal exit.
    with patch("paper_trader.time_module.sleep"), \
         patch("paper_trader._kill_switch_engaged", return_value=True), \
         patch("paper_trader.get_option_premium", return_value=101.0) as mock_get_premium:
        kite, exit_reason, exit_premium = _monitor_position_until_exit("kite", "NIFTY123PE", 100.0)

    assert exit_reason == "KILL_SWITCH"
    assert exit_premium == 101.0
    assert kite == "kite"
    mock_get_premium.assert_called_once()


def test_monitor_position_kill_switch_falls_back_to_entry_price_if_quote_fails():
    # Even if fetching a live exit price fails, the kill switch must still terminate
    # monitoring immediately rather than looping and retrying like a normal exit would.
    with patch("paper_trader.time_module.sleep"), \
         patch("paper_trader._kill_switch_engaged", return_value=True), \
         patch("paper_trader.get_option_premium", side_effect=TokenException("still broken")):
        kite, exit_reason, exit_premium = _monitor_position_until_exit("kite", "NIFTY123PE", 100.0)

    assert exit_reason == "KILL_SWITCH"
    assert exit_premium == 100.0  # entry_premium fallback


def test_kill_switch_engaged_reflects_file_presence(tmp_path):
    switch_path = tmp_path / "KILL_SWITCH"
    assert not _kill_switch_engaged(switch_path)
    switch_path.write_text("on")
    assert _kill_switch_engaged(switch_path)


def test_circuit_breaker_not_tripped_within_limit():
    # -5% loss against a 10% limit shouldn't trip it.
    assert not _circuit_breaker_tripped(daily_pnl=-500.0, day_start_capital=10000.0, max_daily_loss_pct=0.10)


def test_circuit_breaker_tripped_at_exact_limit():
    assert _circuit_breaker_tripped(daily_pnl=-1000.0, day_start_capital=10000.0, max_daily_loss_pct=0.10)


def test_circuit_breaker_tripped_beyond_limit():
    assert _circuit_breaker_tripped(daily_pnl=-1500.0, day_start_capital=10000.0, max_daily_loss_pct=0.10)


def test_circuit_breaker_ignores_profit():
    assert not _circuit_breaker_tripped(daily_pnl=2000.0, day_start_capital=10000.0, max_daily_loss_pct=0.10)


def test_circuit_breaker_never_trips_with_zero_start_capital():
    # Guards div-by-zero-shaped edge case (shouldn't happen in practice, but must not crash).
    assert not _circuit_breaker_tripped(daily_pnl=-100.0, day_start_capital=0.0, max_daily_loss_pct=0.10)


def test_todays_realized_pnl_zero_when_no_csv_yet(tmp_path):
    assert _todays_realized_pnl(tmp_path / "paper_trades.csv", date(2026, 8, 12)) == 0.0


def test_todays_realized_pnl_sums_only_todays_rows(tmp_path):
    csv_path = tmp_path / "paper_trades.csv"
    pd.DataFrame([
        {"date": "2026-08-11", "pnl_rupees": -1000.0},
        {"date": "2026-08-12", "pnl_rupees": 300.0},
        {"date": "2026-08-12", "pnl_rupees": -150.0},
    ]).to_csv(csv_path, index=False)
    assert _todays_realized_pnl(csv_path, date(2026, 8, 12)) == pytest.approx(150.0)


def test_todays_realized_pnl_zero_when_nothing_logged_today(tmp_path):
    csv_path = tmp_path / "paper_trades.csv"
    pd.DataFrame([{"date": "2026-08-11", "pnl_rupees": -1000.0}]).to_csv(csv_path, index=False)
    assert _todays_realized_pnl(csv_path, date(2026, 8, 12)) == 0.0
