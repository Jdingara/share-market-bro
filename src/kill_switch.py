"""
Emergency stop for paper_trader.py (and, later, real trading): drops or clears a
flag file that the trading loop checks before every new entry and while any
position is open (see paper_trader.py's KILL_SWITCH_PATH). Independent of the
dashboard and of whether paper_trader.py's process is even reachable from here -
this only touches a file, so it always works even if the bot was launched from a
different terminal or is stuck.

    py src/kill_switch.py on       # stop trading now - also force-closes an open position
    py src/kill_switch.py off      # clear it, allow trading to resume
    py src/kill_switch.py status   # check whether it's currently engaged

paper_trader.py checks this on every loop iteration (up to ~60s while waiting for
a signal, up to ~15s while a position is open) - "on" takes effect within
seconds, no restart needed. Also available as a button on the dashboard.
"""

from __future__ import annotations

import argparse

from paper_trader import KILL_SWITCH_PATH


def turn_on() -> None:
    KILL_SWITCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    KILL_SWITCH_PATH.write_text("Kill switch engaged manually via kill_switch.py.\n")
    print(f"Kill switch ON ({KILL_SWITCH_PATH}). Any running bot will stop new entries and "
          "force-close an open position within seconds.")


def turn_off() -> None:
    KILL_SWITCH_PATH.unlink(missing_ok=True)
    print(f"Kill switch OFF ({KILL_SWITCH_PATH} removed). Trading can resume normally.")


def status() -> None:
    if KILL_SWITCH_PATH.exists():
        print(f"Kill switch is ON ({KILL_SWITCH_PATH} exists).")
    else:
        print("Kill switch is OFF.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["on", "off", "status"])
    args = parser.parse_args()
    {"on": turn_on, "off": turn_off, "status": status}[args.action]()


if __name__ == "__main__":
    main()
