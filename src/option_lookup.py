"""
Finds the real, currently-tradable NIFTY option contract matching a given
strike/expiry/type, and fetches its real quoted premium - the key
difference from Phase 3's backtester, which had to simulate premiums via
Black-Scholes because no historical option data exists. Here we're live, so
we use Kite's real quotes directly.
"""

from __future__ import annotations

from datetime import date
from typing import Literal


class OptionLookupError(RuntimeError):
    pass


def find_nearest_valid_expiry(
    instruments: list[dict],
    as_of_date: date,
    min_days: int = 3,
    name: str = "NIFTY",
) -> date:
    """Finds the nearest REAL listed expiry date at least min_days out, by reading
    the actual live instrument list - robust to NSE changing which weekday it uses
    for weekly expiries (confirmed live on 2026-07-08: it changed from Thursday to
    Tuesday at some point, which silently broke a live trade that day when the code
    was still guessing Thursday). Use this for live trading. options_pricing.next_weekly_expiry()
    (a weekday guess) remains necessary for backtesting/simulation, where no real
    instrument list exists for past dates - but it is NOT robust to this kind of
    change and must not be used for live trading."""
    expiries = sorted({i["expiry"] for i in instruments if i.get("name") == name and i.get("segment") == "NFO-OPT"})
    valid = [e for e in expiries if (e - as_of_date).days >= min_days]
    if not valid:
        raise OptionLookupError(f"No valid {name} expiry at least {min_days} days out found in the instrument list")
    return valid[0]


def find_option_instrument(
    instruments: list[dict],
    strike: float,
    expiry: date,
    option_type: Literal["CE", "PE"],
    name: str = "NIFTY",
) -> dict:
    """Looks up a specific option's instrument record from an NFO instruments dump
    (as returned by kite.instruments("NFO")). Takes the list rather than a kite
    client so this is testable with a fake list, without hitting the live API."""
    for instrument in instruments:
        if (
            instrument.get("name") == name
            and instrument.get("segment") == "NFO-OPT"
            and instrument.get("strike") == strike
            and instrument.get("expiry") == expiry
            and instrument.get("instrument_type") == option_type
        ):
            return instrument
    raise OptionLookupError(f"No matching {name} option found for strike={strike} expiry={expiry} type={option_type}")


def get_option_premium(kite, tradingsymbol: str, exchange: str = "NFO") -> float:
    """Real current premium via Kite's quote API (last traded price)."""
    quote = kite.quote(f"{exchange}:{tradingsymbol}")
    return quote[f"{exchange}:{tradingsymbol}"]["last_price"]


def compute_max_pain_from_oi(oi_by_strike_type: dict[tuple[float, str], int]) -> float:
    """Pure function: the strike that minimizes total payout to option holders
    (calls + puts combined) at expiry, given each strike/type's open interest.
    Theory: option WRITERS (who net short far more contracts than they hold) are
    incentivized to hedge into price gravitating here as expiry nears - the "seller
    mob" behavior the user described. Split out from compute_max_pain() so this can
    be unit-tested without a live Kite connection - OI data isn't retained after
    expiry, so the live-fetching half can never be backtested, only this pure half."""
    strikes = sorted({strike for strike, _ in oi_by_strike_type})
    if not strikes:
        raise OptionLookupError("No strikes with OI data given - cannot compute Max Pain")

    best_strike, best_payout = None, None
    for s in strikes:
        total_payout = 0.0
        for (k, option_type), oi in oi_by_strike_type.items():
            if option_type == "CE" and s > k:
                total_payout += (s - k) * oi
            elif option_type == "PE" and s < k:
                total_payout += (k - s) * oi
        if best_payout is None or total_payout < best_payout:
            best_payout = total_payout
            best_strike = s
    return best_strike


def compute_max_pain(kite, instruments: list[dict], expiry: date, name: str = "NIFTY") -> float:
    """Live Max Pain strike for one expiry, using real Open Interest across every
    strike (both CE and PE) via a single batched kite.quote() call (confirmed live
    on 2026-07-15: all 186 NIFTY contracts for one expiry fit in one call)."""
    relevant = [
        i for i in instruments
        if i.get("name") == name and i.get("segment") == "NFO-OPT" and i.get("expiry") == expiry
    ]
    if not relevant:
        raise OptionLookupError(f"No {name} option instruments found for expiry={expiry}")

    symbols = [f"NFO:{i['tradingsymbol']}" for i in relevant]
    quotes = kite.quote(symbols)

    oi_by_strike_type: dict[tuple[float, str], int] = {}
    for i in relevant:
        symbol = f"NFO:{i['tradingsymbol']}"
        oi = quotes.get(symbol, {}).get("oi", 0)
        oi_by_strike_type[(i["strike"], i["instrument_type"])] = oi

    return compute_max_pain_from_oi(oi_by_strike_type)
