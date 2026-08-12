"""Unit tests for kill_switch.py's on/off/status file toggling."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import kill_switch


def test_turn_on_creates_the_flag_file(tmp_path):
    switch_path = tmp_path / "KILL_SWITCH"
    with patch.object(kill_switch, "KILL_SWITCH_PATH", switch_path):
        kill_switch.turn_on()
    assert switch_path.exists()


def test_turn_off_removes_the_flag_file(tmp_path):
    switch_path = tmp_path / "KILL_SWITCH"
    switch_path.write_text("on")
    with patch.object(kill_switch, "KILL_SWITCH_PATH", switch_path):
        kill_switch.turn_off()
    assert not switch_path.exists()


def test_turn_off_is_a_no_op_when_already_off(tmp_path):
    # Must not raise just because there was nothing to remove.
    switch_path = tmp_path / "KILL_SWITCH"
    with patch.object(kill_switch, "KILL_SWITCH_PATH", switch_path):
        kill_switch.turn_off()
    assert not switch_path.exists()


def test_turn_on_creates_parent_directory_if_missing(tmp_path):
    switch_path = tmp_path / "nested" / "KILL_SWITCH"
    with patch.object(kill_switch, "KILL_SWITCH_PATH", switch_path):
        kill_switch.turn_on()
    assert switch_path.exists()


def test_status_does_not_raise_either_way(tmp_path, capsys):
    switch_path = tmp_path / "KILL_SWITCH"
    with patch.object(kill_switch, "KILL_SWITCH_PATH", switch_path):
        kill_switch.status()
        assert "OFF" in capsys.readouterr().out
        switch_path.write_text("on")
        kill_switch.status()
        assert "ON" in capsys.readouterr().out
