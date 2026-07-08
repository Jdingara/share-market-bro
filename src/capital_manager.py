"""
Capital-based position sizing for paper trading: tracks a running rupee
balance (persisted across days, not reset each morning), and works out how
many whole option lots that balance can actually afford at a given premium.

This resolves the "position sizing" question left open since the start of
the project. Kept deliberately simple for now (fixed starting capital,
whole-lot rounding, no risk-based sizing) - a reasonable first version to
validate the mechanics before anything more sophisticated.

Also caps how much of a growing balance is ever risked on a single trade
(MAX_CAPITAL_PER_TRADE) - the user's own idea, so that even if the account
compounds up to a very large balance one day, the worst-case single-trade
loss stays bounded forever rather than growing right alongside the account.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from options_pricing import NIFTY_LOT_SIZE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAPITAL_STATE_PATH = PROJECT_ROOT / "data" / "paper_trades" / "capital_state.json"

STARTING_CAPITAL = 20000.0
MAX_CAPITAL_PER_TRADE = 200000.0  # never deploy more than this on one trade, no matter how large the balance grows


def load_capital() -> float:
    """Returns the running paper-trading balance, or STARTING_CAPITAL if this
    is the very first run (no state file yet)."""
    if not CAPITAL_STATE_PATH.exists():
        return STARTING_CAPITAL
    return json.loads(CAPITAL_STATE_PATH.read_text())["capital"]


def save_capital(capital: float) -> None:
    CAPITAL_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CAPITAL_STATE_PATH.write_text(json.dumps({"capital": capital}))


def deployable_capital(capital: float, max_per_trade: float = MAX_CAPITAL_PER_TRADE) -> float:
    """How much of the current balance is actually allowed to be risked on a single
    trade - the lesser of the full balance or the cap. Anything above the cap stays
    untouched/idle, regardless of how large the account has grown."""
    return min(capital, max_per_trade)


def calculate_affordable_lots(capital: float, premium: float, lot_size: int = NIFTY_LOT_SIZE) -> int:
    """How many whole lots the given capital can afford at this premium - always
    rounds down, since options can only be bought in whole-lot quantities."""
    return math.floor(capital / (premium * lot_size))


def apply_trade_pnl(
    capital: float,
    lots: int,
    entry_premium: float,
    exit_premium: float,
    lot_size: int = NIFTY_LOT_SIZE,
) -> float:
    """Updates capital after a trade closes. Only the invested portion moves -
    any leftover capital that wasn't enough for one more lot stays untouched."""
    invested = lots * lot_size * entry_premium
    exit_value = lots * lot_size * exit_premium
    return capital - invested + exit_value
