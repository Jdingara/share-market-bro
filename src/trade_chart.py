"""
Per-trade candlestick snapshot: a 2-hour, 5-minute-candle chart of the traded
option's own premium (not the underlying NIFTY index), centered on entry time
(1 hour before to 1 hour after), with the entry and exit points marked. Lets
us eyeball, after the fact, whether a stop-loss exit would have recovered if
held longer, or whether a target exit left further upside on the table.

Must be generated soon after each trade closes, not on-demand later - Kite
Connect does not retain historical data for expired option contracts at all
(see PROJECT_STATUS.md finding #6), so waiting until the dashboard is viewed
days later risks the underlying data already being gone.

Chart generation failures must never take down the trading loop - this is a
diagnostic nice-to-have, not core trading logic, so callers should treat
this as best-effort (wrap in try/except).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless - never try to open a GUI window from a background trading loop

import mplfinance as mpf
import pandas as pd

from data_fetch import fetch_historical_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHARTS_DIR = PROJECT_ROOT / "data" / "paper_trades" / "charts"

WINDOW_BEFORE = timedelta(hours=1)
WINDOW_AFTER = timedelta(hours=1)
CHART_INTERVAL = "5minute"


def generate_trade_chart(
    kite,
    instrument_token: int,
    tradingsymbol: str,
    direction: str,
    entry_time: datetime,
    exit_time: datetime,
    entry_premium: float,
    exit_premium: float,
    exit_reason: str,
) -> Path:
    """Fetches the option's own 5-minute candles for the entry day, slices to a
    1-hour window around entry time, marks entry/exit, and saves a PNG. Returns
    the saved file's path."""
    trading_day = entry_time.date()

    raw = fetch_historical_data(kite, instrument_token, trading_day, trading_day, CHART_INTERVAL)
    raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)
    raw = raw.set_index("date").sort_index()

    window_start = entry_time - WINDOW_BEFORE
    window_end = entry_time + WINDOW_AFTER
    windowed = raw.loc[window_start:window_end]

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    file_name = f"{trading_day.isoformat()}_{entry_time.strftime('%H%M%S')}_{tradingsymbol}.png"
    out_path = CHARTS_DIR / file_name

    if windowed.empty:
        # No candles in the window (e.g. entry right at market open/close) - still
        # save a placeholder-free skip rather than crash the caller.
        raise ValueError(f"No {CHART_INTERVAL} candles found for {tradingsymbol} in the requested window")

    entry_marker = pd.Series(float("nan"), index=windowed.index)
    exit_marker = pd.Series(float("nan"), index=windowed.index)
    entry_idx = windowed.index[windowed.index.get_indexer([entry_time], method="nearest")[0]]
    exit_idx = windowed.index[windowed.index.get_indexer([exit_time], method="nearest")[0]]
    entry_marker.loc[entry_idx] = entry_premium
    exit_marker.loc[exit_idx] = exit_premium

    win = exit_reason == "TARGET"
    exit_color = "lime" if win else ("red" if exit_reason == "STOPLOSS" else "orange")

    addplots = [
        mpf.make_addplot(entry_marker, type="scatter", markersize=120, marker="^", color="blue"),
        mpf.make_addplot(exit_marker, type="scatter", markersize=120, marker="v", color=exit_color),
    ]

    pct_change = (exit_premium - entry_premium) / entry_premium
    title = (
        f"{tradingsymbol} ({direction})  entry {entry_premium} -> exit {exit_premium}  "
        f"({pct_change:+.2%}, {exit_reason})"
    )

    mpf.plot(
        windowed[["open", "high", "low", "close", "volume"]],
        type="candle",
        style="yahoo",
        addplot=addplots,
        title=title,
        ylabel="Premium (Rs)",
        volume=False,
        savefig=dict(fname=str(out_path), dpi=100, bbox_inches="tight"),
    )

    return out_path
