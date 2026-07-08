"""
Local web dashboard: a control panel (start/stop the live paper trader) plus
report views (backtest results, paper trading results) on top of the CSVs
already produced by backtester.py and paper_trader.py. No trading logic
lives here - this is purely a viewer + process launcher for those scripts.

Run with: py -m streamlit run src/dashboard.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from capital_manager import load_capital

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKTEST_CSV = PROJECT_ROOT / "data" / "backtest_results" / "trades.csv"
PAPER_TRADES_CSV = PROJECT_ROOT / "data" / "paper_trades" / "paper_trades.csv"
LIVE_LOG_PATH = PROJECT_ROOT / "data" / "paper_trades" / "live_log.txt"
PAPER_TRADER_SCRIPT = PROJECT_ROOT / "src" / "paper_trader.py"

st.set_page_config(page_title="Share Market Bro", layout="wide")


# ---------- Bot control ----------

def start_bot(max_trades_per_day: int = 1) -> None:
    LIVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(LIVE_LOG_PATH, "w")
    process = subprocess.Popen(
        # -u: unbuffered, so the log file updates live, not just at exit
        [sys.executable, "-u", str(PAPER_TRADER_SCRIPT), "--max-trades-per-day", str(max_trades_per_day)],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
    )
    st.session_state.bot_process = process
    st.session_state.bot_log_fh = log_fh
    st.session_state.bot_start_time = datetime.now()
    st.session_state.bot_max_trades = max_trades_per_day


def stop_bot() -> None:
    process = st.session_state.get("bot_process")
    if process and process.poll() is None:
        process.terminate()
    st.session_state.bot_process = None
    log_fh = st.session_state.get("bot_log_fh")
    if log_fh:
        log_fh.close()
    st.session_state.bot_log_fh = None


def render_bot_control() -> None:
    st.header("Bot Control")

    process = st.session_state.get("bot_process")
    is_running = process is not None and process.poll() is None

    max_trades = st.number_input(
        "Max trades per day",
        min_value=1, max_value=50, value=1, step=1,
        disabled=is_running,
        help=(
            "Default 1 is the intended everyday discipline. Set higher (e.g. 10) only for a few "
            "validation days to see more real results faster - every trade still has to independently "
            "clear the same confidence bar, nothing is ever forced. Switch back to 1 once you've seen enough."
        ),
    )
    if max_trades != 1:
        st.warning(f"Set to {int(max_trades)} - remember to set this back to 1 once you're done validating.")

    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("Start", disabled=is_running, type="primary"):
            start_bot(max_trades_per_day=int(max_trades))
            st.rerun()
    with col2:
        if st.button("Stop", disabled=not is_running):
            stop_bot()
            st.rerun()
    with col3:
        if is_running:
            started = st.session_state.bot_start_time.strftime("%H:%M:%S")
            trades_note = f" (max {st.session_state.bot_max_trades} trades/day)" if st.session_state.get("bot_max_trades", 1) != 1 else ""
            st.success(f"Running since {started}{trades_note}")
        else:
            st.info("Not running")

    st.caption(
        "Note: if you close/restart this dashboard while the bot is running, it loses track of that "
        "process (it keeps running in the background regardless) - close it manually via Task Manager "
        "in that case."
    )

    st.subheader("Live log")
    if LIVE_LOG_PATH.exists():
        lines = LIVE_LOG_PATH.read_text().splitlines()
        st.code("\n".join(lines[-20:]) or "(no output yet)", language=None)
    else:
        st.caption("No log yet - start the bot to see live progress here.")

    auto_refresh = st.checkbox("Auto-refresh every 10 seconds", value=False)
    st.button("Refresh now")
    if auto_refresh:
        time.sleep(10)
        st.rerun()


# ---------- Reports ----------

def load_trades(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def compute_summary(df: pd.DataFrame) -> dict:
    has_no_trade_rows = "direction" in df.columns and (df["direction"] == "NO_TRADE").any()
    trades = df[df["direction"] != "NO_TRADE"] if has_no_trade_rows else df

    summary = {
        "days_evaluated": len(df) if has_no_trade_rows else None,
        "trades_taken": len(trades),
        "wins": int((trades["exit_reason"] == "TARGET").sum()) if len(trades) else 0,
        "losses": int((trades["exit_reason"] == "STOPLOSS").sum()) if len(trades) else 0,
        "eod_closes": int((trades["exit_reason"] == "EOD_CLOSE").sum()) if len(trades) else 0,
        "avg_pct_change": trades["pct_change"].mean() * 100 if len(trades) else 0.0,
    }
    decided = summary["wins"] + summary["losses"]
    summary["win_rate_excl_eod"] = (summary["wins"] / decided * 100) if decided else None
    return summary, trades


def render_report(csv_path: Path, title: str, show_capital: bool = False) -> None:
    if show_capital:
        st.metric("Current Paper Capital", f"Rs {load_capital():,.2f}")

    df = load_trades(csv_path)
    if df.empty:
        st.info(f"No data yet at `{csv_path.relative_to(PROJECT_ROOT)}`.")
        return

    summary, trades = compute_summary(df)

    cols = st.columns(5)
    if summary["days_evaluated"] is not None:
        cols[0].metric("Days evaluated", summary["days_evaluated"])
    cols[1].metric("Trades taken", summary["trades_taken"])
    win_rate = summary["win_rate_excl_eod"]
    cols[2].metric("Win rate (excl. EOD)", f"{win_rate:.1f}%" if win_rate is not None else "n/a")
    cols[3].metric("Avg P&L / trade", f"{summary['avg_pct_change']:+.2f}%")
    cols[4].metric("Wins / Losses / EOD", f"{summary['wins']} / {summary['losses']} / {summary['eod_closes']}")

    if len(trades):
        has_rupee_pnl = "pnl_rupees" in trades.columns
        st.subheader("Equity curve" + (" (rupees)" if has_rupee_pnl else " (cumulative % return across trades)"))
        if has_rupee_pnl:
            st.line_chart(trades["capital_after"].reset_index(drop=True))
        else:
            equity = trades["pct_change"].cumsum() * 100
            st.line_chart(equity.reset_index(drop=True))

        st.subheader("Trade log")
        st.dataframe(trades, width="stretch")


# ---------- Layout ----------

st.title("Share Market Bro")

render_bot_control()

st.divider()

tab1, tab2 = st.tabs(["Backtest Results", "Paper Trading Results"])
with tab1:
    render_report(BACKTEST_CSV, "Backtest Results")
with tab2:
    render_report(PAPER_TRADES_CSV, "Paper Trading Results", show_capital=True)
