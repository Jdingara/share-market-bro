"""
Historical candle data fetcher.

Kite's historical data endpoint caps how much history you can pull in a
single request, and the cap depends on the interval. This module chunks a
requested date range into calls that respect those caps, stitches the
results together, and caches them to CSV under data/historical/ so repeated
backtesting runs don't have to re-hit the API every time.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from auth import login

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HISTORICAL_DIR = PROJECT_ROOT / "data" / "historical"

# Max number of days worth of data Kite allows per single historical_data() call,
# per interval. Source: Kite Connect historical data docs.
MAX_DAYS_PER_REQUEST = {
    "minute": 60,
    "3minute": 100,
    "5minute": 100,
    "10minute": 100,
    "15minute": 200,
    "30minute": 200,
    "60minute": 400,
    "day": 2000,
}


class DataFetchError(RuntimeError):
    pass


def get_instrument_token(kite, tradingsymbol: str, exchange: str = "NSE") -> int:
    """Look up an instrument's token from Kite's instrument dump rather than hardcoding it."""
    for instrument in kite.instruments(exchange):
        if instrument["tradingsymbol"] == tradingsymbol:
            return instrument["instrument_token"]
    raise DataFetchError(f"Could not find instrument '{tradingsymbol}' on exchange '{exchange}'")


def fetch_historical_data(
    kite,
    instrument_token: int,
    from_date: date,
    to_date: date,
    interval: str,
) -> pd.DataFrame:
    """Fetch historical candles for a range, chunking requests to respect Kite's per-interval limits."""
    if interval not in MAX_DAYS_PER_REQUEST:
        raise DataFetchError(f"Unknown interval '{interval}'. Valid: {list(MAX_DAYS_PER_REQUEST)}")

    max_days = MAX_DAYS_PER_REQUEST[interval]
    chunks: list[pd.DataFrame] = []

    chunk_start = from_date
    while chunk_start <= to_date:
        chunk_end = min(chunk_start + timedelta(days=max_days - 1), to_date)

        candles = kite.historical_data(
            instrument_token,
            chunk_start.isoformat(),
            chunk_end.isoformat(),
            interval,
        )
        if candles:
            chunks.append(pd.DataFrame(candles))

        chunk_start = chunk_end + timedelta(days=1)

    if not chunks:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    return df


def save_to_csv(df: pd.DataFrame, tradingsymbol: str, interval: str) -> Path:
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
    safe_symbol = tradingsymbol.replace(" ", "_")
    out_path = HISTORICAL_DIR / f"{safe_symbol}_{interval}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and cache historical candle data from Kite Connect.")
    parser.add_argument("--symbol", default="NIFTY 50", help="Trading symbol, e.g. 'NIFTY 50'")
    parser.add_argument("--exchange", default="NSE", help="Exchange segment, e.g. NSE")
    parser.add_argument("--interval", default="day", choices=list(MAX_DAYS_PER_REQUEST))
    parser.add_argument("--days", type=int, default=30, help="How many days of history to fetch, ending today")
    args = parser.parse_args()

    kite = login()
    instrument_token = get_instrument_token(kite, args.symbol, args.exchange)

    to_date = date.today()
    from_date = to_date - timedelta(days=args.days)

    df = fetch_historical_data(kite, instrument_token, from_date, to_date, args.interval)
    out_path = save_to_csv(df, args.symbol, args.interval)

    print(f"Fetched {len(df)} candles for {args.symbol} ({args.interval}) -> {out_path}")
    if not df.empty:
        print(df.head())
        print("...")
        print(df.tail())


if __name__ == "__main__":
    main()
