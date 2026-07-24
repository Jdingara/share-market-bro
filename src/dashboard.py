"""
Local web dashboard: a control panel (start/stop the live paper trader) plus
report views (backtest results, paper trading results) on top of the CSVs
already produced by backtester.py and paper_trader.py. No trading logic
lives here - this is purely a viewer + process launcher for those scripts.

Run with: py -m streamlit run src/dashboard.py
"""

from __future__ import annotations

import io
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from capital_manager import MAX_CAPITAL_PER_TRADE, load_capital

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKTEST_CSV = PROJECT_ROOT / "data" / "backtest_results" / "trades.csv"
PAPER_TRADES_CSV = PROJECT_ROOT / "data" / "paper_trades" / "paper_trades.csv"
LIVE_LOG_PATH = PROJECT_ROOT / "data" / "paper_trades" / "live_log.txt"
PAPER_TRADER_SCRIPT = PROJECT_ROOT / "src" / "paper_trader.py"

st.set_page_config(page_title="Share Market Bro", page_icon="\U0001F4C8", layout="wide")

# Status colors carry meaning (win/loss/neutral) - always paired with a text
# label nearby, never color alone, per the project's color-accessibility rule.
COLOR_GOOD = "#0ca30c"
COLOR_CRITICAL = "#d03b3b"
COLOR_MUTED = "#8a8f98"
COLOR_BLUE = "#2a78d6"

# Real-world cost estimate, for an honest "what would this actually keep" figure -
# not simulated, these are real published/documented rates:
# - Zerodha's flat F&O brokerage is Rs 20 per executed order (zerodha.com/charges);
#   each trade here is 2 orders (entry + exit), so Rs 40/trade.
# - Kite Connect API costs Rs 500 per 30 days (see PROJECT_STATUS.md), amortized daily.
# NOTE: this does NOT include STT, exchange transaction charges, GST, or stamp duty -
# those would reduce the real net figure further. Brokerage-only estimate, not exact.
BROKERAGE_PER_ORDER = 20.0
ORDERS_PER_TRADE = 2
KITE_MONTHLY_COST = 500.0
KITE_BILLING_DAYS = 30


# ---------- Styling ----------

def inject_css() -> None:
    st.markdown(
        """
        <style>
          .sm-hero {
            background: linear-gradient(135deg, #eef4ff 0%, #f9fafb 100%);
            border: 1px solid #dde3ec; border-radius: 16px;
            padding: 22px 26px; margin-bottom: 18px;
            display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;
          }
          .sm-hero .title { font-size: 26px; font-weight: 700; color: #0b0b0b; margin: 0; }
          .sm-hero .subtitle { font-size: 14px; color: #5b6270; margin-top: 4px; }
          .sm-badge {
            display: inline-flex; align-items: center; gap: 7px;
            padding: 6px 14px; border-radius: 999px; font-size: 13px; font-weight: 600;
          }
          .sm-badge.live { background: #e8f6e8; color: #0ca30c; border: 1px solid #0ca30c; }
          .sm-badge.off { background: #f0f1f3; color: #6b7280; border: 1px solid #d8dbe0; }
          .sm-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
          .sm-dot.live { background: #0ca30c; box-shadow: 0 0 0 3px rgba(12,163,12,0.18); }
          .sm-dot.off { background: #9aa0a8; }

          .sm-stat {
            background: #fcfcfb; border: 1px solid #e1e0d9; border-radius: 12px;
            padding: 14px 16px; height: 100%;
          }
          .sm-stat .label { font-size: 11.5px; color: #6b7280; text-transform: uppercase; letter-spacing: .04em; }
          .sm-stat .value { font-size: 24px; font-weight: 700; margin-top: 3px; color: #0b0b0b; }
          .sm-stat .value.good { color: #0ca30c; }
          .sm-stat .value.bad { color: #d03b3b; }

          section[data-testid="stSidebar"] { display: none; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def stat_tile(col, label: str, value: str, tone: str | None = None) -> None:
    tone_class = f" {tone}" if tone else ""
    col.markdown(
        f"""<div class="sm-stat"><div class="label">{label}</div>
              <div class="value{tone_class}">{value}</div></div>""",
        unsafe_allow_html=True,
    )


# ---------- Bot control ----------

def start_bot(
    max_trades_per_day: int = 1,
    max_capital_per_trade: float = MAX_CAPITAL_PER_TRADE,
    put_only: bool = True,
    split_session: bool = False,
    max_trades_per_session: int = 6,
) -> None:
    LIVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(LIVE_LOG_PATH, "w")
    cmd = [
        sys.executable, "-u", str(PAPER_TRADER_SCRIPT),
        "--max-trades-per-day", str(max_trades_per_day),
        "--max-capital-per-trade", str(max_capital_per_trade),
    ]
    if not put_only:
        cmd.append("--allow-calls")
    if split_session:
        cmd += ["--split-session", "--max-trades-per-session", str(max_trades_per_session)]
    process = subprocess.Popen(
        # -u: unbuffered, so the log file updates live, not just at exit
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
    )
    st.session_state.bot_process = process
    st.session_state.bot_log_fh = log_fh
    st.session_state.bot_start_time = datetime.now()
    st.session_state.bot_max_trades = max_trades_per_day
    st.session_state.bot_put_only = put_only
    st.session_state.bot_max_capital_per_trade = max_capital_per_trade
    st.session_state.bot_split_session = split_session
    st.session_state.bot_max_trades_per_session = max_trades_per_session


def stop_bot() -> None:
    process = st.session_state.get("bot_process")
    if process and process.poll() is None:
        process.terminate()
    st.session_state.bot_process = None
    log_fh = st.session_state.get("bot_log_fh")
    if log_fh:
        log_fh.close()
    st.session_state.bot_log_fh = None


def render_bot_control() -> bool:
    process = st.session_state.get("bot_process")
    is_running = process is not None and process.poll() is None

    with st.container(border=True):
        header_col, badge_col = st.columns([3, 1])
        header_col.subheader("\U0001F916 Bot Control")
        if is_running:
            started = st.session_state.bot_start_time.strftime("%H:%M:%S")
            badge_col.markdown(
                f'<div style="text-align:right; padding-top:8px">'
                f'<span class="sm-badge live"><span class="sm-dot live"></span>Running since {started}</span></div>',
                unsafe_allow_html=True,
            )
        else:
            badge_col.markdown(
                '<div style="text-align:right; padding-top:8px">'
                '<span class="sm-badge off"><span class="sm-dot off"></span>Not running</span></div>',
                unsafe_allow_html=True,
            )

        split_session_mode = st.checkbox(
            "Split into morning/afternoon sessions",
            value=False,
            disabled=is_running,
            help=(
                "Instead of one flat daily cap, use two independent quotas: up to N trades before 1:15 PM "
                "(morning, early-session 5-min model) and up to N more from 1:15 PM onward (afternoon, "
                "primary 15-min model) - up to 2N trades total. If the morning quota fills before 1:15, new "
                "entries pause until the afternoon quota opens rather than ending the day. Added 2026-07-16 "
                "based on morning trades looking better than later-day ones so far - still an early idea, "
                "not yet proven over many days."
            ),
        )

        if split_session_mode:
            max_trades_per_session = st.number_input(
                "Trades per session (morning / afternoon)",
                min_value=1, max_value=50, value=6, step=1,
                disabled=is_running,
                help="Cap for EACH session - morning and afternoon each get this many, so the real daily max is double this number.",
            )
            max_trades = 1  # unused in split-session mode, kept for start_bot()'s shared signature
        else:
            max_trades_per_session = 6  # unused in flat mode, kept for start_bot()'s shared signature
            max_trades = st.number_input(
                "Max trades per day",
                min_value=1, max_value=999, value=1, step=1,
                disabled=is_running,
                help=(
                    "Default 1 is the intended everyday discipline. Set much higher (e.g. 999) to remove the "
                    "cap entirely for a validation stretch - every trade still has to independently clear the "
                    "same confidence bar, nothing is ever forced, so the real number of trades taken is however "
                    "many genuine signals actually show up, not this number. Switch back to 1 once you've seen enough."
                ),
            )
            if max_trades != 1:
                st.warning(f"Set to {int(max_trades)} - remember to set this back to 1 once you're done validating.")

        max_capital = st.number_input(
            "Max capital per trade (Rs)",
            min_value=1000.0, value=MAX_CAPITAL_PER_TRADE, step=10000.0,
            disabled=is_running,
            help=(
                "Never deploy more than this much of the balance on a single trade, no matter how large the "
                f"account grows. Default: Rs {MAX_CAPITAL_PER_TRADE:,.0f}. Anything above the cap stays idle."
            ),
        )

        allow_calls_mode = st.checkbox(
            "Allow CALL trades",
            value=False,
            disabled=is_running,
            help=(
                "Off by default - PUT-only. CALL's confidence collapses at high thresholds in BOTH "
                "the primary and early-session models (confirmed 2026-07-24) - a consistent, "
                "cross-model weakness, not one bad stretch. Check this box only to deliberately "
                "gather more CALL data."
            ),
        )
        if allow_calls_mode:
            st.warning("CALL signals will be taken this run, despite confirmed weak precision in both models.")

        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            if st.button("▶ Start", disabled=is_running, type="primary", width="stretch"):
                start_bot(
                    max_trades_per_day=int(max_trades),
                    max_capital_per_trade=float(max_capital),
                    put_only=not allow_calls_mode,
                    split_session=split_session_mode,
                    max_trades_per_session=int(max_trades_per_session),
                )
                st.rerun()
        with col2:
            if st.button("■ Stop", disabled=not is_running, width="stretch"):
                stop_bot()
                st.rerun()
        with col3:
            if is_running:
                if st.session_state.get("bot_split_session", False):
                    n = st.session_state.get("bot_max_trades_per_session", 6)
                    trades_note = f" · up to {n}+{n} trades (morning+afternoon)"
                else:
                    trades_note = f" · max {st.session_state.bot_max_trades} trades/day" if st.session_state.get("bot_max_trades", 1) != 1 else ""
                cap = st.session_state.get("bot_max_capital_per_trade", MAX_CAPITAL_PER_TRADE)
                capital_note = f" · cap Rs {cap:,.0f}/trade" if cap != MAX_CAPITAL_PER_TRADE else ""
                calls_note = "" if st.session_state.get("bot_put_only", True) else " · CALLs allowed"
                st.caption(f"{trades_note}{capital_note}{calls_note}".lstrip(" ·") or "​")

        st.caption(
            "Note: if you close/restart this dashboard while the bot is running, it loses track of that "
            "process (it keeps running in the background regardless) - close it manually via Task Manager "
            "in that case."
        )

        with st.expander("\U0001F4DC Live log", expanded=is_running):
            if LIVE_LOG_PATH.exists():
                lines = LIVE_LOG_PATH.read_text().splitlines()
                st.code("\n".join(lines[-20:]) or "(no output yet)", language=None)
            else:
                st.caption("No log yet - start the bot to see live progress here.")

            auto_refresh = st.checkbox("Auto-refresh every 10 seconds", value=True)

    return auto_refresh


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


def render_equity_chart(trades: pd.DataFrame, has_rupee_pnl: bool) -> None:
    if has_rupee_pnl:
        series = trades["capital_after"].reset_index(drop=True)
        y_title = "Capital (Rs)"
    else:
        series = (trades["pct_change"].cumsum() * 100).reset_index(drop=True)
        y_title = "Cumulative Return (%)"
    chart_df = pd.DataFrame({"Trade #": range(1, len(series) + 1), y_title: series})

    base = alt.Chart(chart_df).encode(
        x=alt.X("Trade #:Q", axis=alt.Axis(tickMinStep=1, grid=False)),
        y=alt.Y(f"{y_title}:Q", scale=alt.Scale(zero=False)),
        tooltip=["Trade #", alt.Tooltip(f"{y_title}:Q", format=",.2f")],
    )
    area = base.mark_area(opacity=0.12, color=COLOR_BLUE)
    line = base.mark_line(color=COLOR_BLUE, strokeWidth=2.5)
    points = base.mark_circle(color=COLOR_BLUE, size=35)
    st.altair_chart((area + line + points).properties(height=260), width="stretch", theme=None)


def render_outcome_bar(summary: dict) -> None:
    total = summary["wins"] + summary["losses"] + summary["eod_closes"]
    if total == 0:
        return
    data = pd.DataFrame([
        {"Outcome": "Win", "Count": summary["wins"], "order": 0},
        {"Outcome": "Loss", "Count": summary["losses"], "order": 1},
        {"Outcome": "EOD Close", "Count": summary["eod_closes"], "order": 2},
    ])
    domain = ["Win", "Loss", "EOD Close"]
    colors = [COLOR_GOOD, COLOR_CRITICAL, COLOR_MUTED]
    chart = alt.Chart(data).mark_bar(size=34).encode(
        x=alt.X("Count:Q", stack="normalize", axis=None, title=None),
        y=alt.value(0),
        color=alt.Color("Outcome:N", scale=alt.Scale(domain=domain, range=colors),
                         legend=alt.Legend(title=None, orient="bottom")),
        order=alt.Order("order:Q"),
        tooltip=["Outcome", "Count"],
    ).properties(height=70)
    st.altair_chart(chart, width="stretch", theme=None)


def render_daily_pnl_chart(trades: pd.DataFrame) -> None:
    if "date" not in trades.columns:
        return
    value_col = "pnl_rupees" if "pnl_rupees" in trades.columns else "pct_change"
    y_title = "P&L (Rs)" if value_col == "pnl_rupees" else "P&L (%)"
    mult = 1 if value_col == "pnl_rupees" else 100

    daily = trades.groupby("date", as_index=False)[value_col].sum()
    daily[y_title] = daily[value_col] * mult
    daily["Outcome"] = daily[y_title].apply(lambda v: "Profit" if v >= 0 else "Loss")

    bars = alt.Chart(daily).mark_bar(size=22).encode(
        x=alt.X("date:N", title=None, sort=None),
        y=alt.Y(f"{y_title}:Q"),
        color=alt.Color("Outcome:N", scale=alt.Scale(domain=["Profit", "Loss"], range=[COLOR_GOOD, COLOR_CRITICAL]),
                         legend=alt.Legend(title=None, orient="bottom")),
        tooltip=["date", alt.Tooltip(f"{y_title}:Q", format=",.2f")],
    )
    rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=COLOR_MUTED, strokeWidth=1).encode(y="y:Q")
    st.altair_chart((bars + rule).properties(height=220), width="stretch", theme=None)


@st.cache_data(show_spinner=False)
def dataframe_to_pdf_bytes(df: pd.DataFrame, title: str) -> bytes:
    """Renders a table to a printable PDF page via matplotlib (already a project
    dependency - no new library needed, unlike weasyprint/reportlab which would
    need extra system setup on the user's machine). Cached on the table's actual
    content - with auto-refresh reruning this every 10s, an uncached version would
    rebuild the PDF from scratch on every tick even when no new trade has happened;
    this way it only actually re-renders when the underlying data changes (a real
    project-day case: a growing 80+ row trade log made the uncached version
    noticeably slow on every single rerun, not just when downloading)."""
    display_df = df.fillna("")
    row_height = 0.3
    fig_height = max(2.0, row_height * (len(display_df) + 2))
    fig, ax = plt.subplots(figsize=(11.7, fig_height))  # landscape A4 width
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    table = ax.table(
        cellText=display_df.values.astype(str),
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def render_report(csv_path: Path, title: str, show_capital: bool = False) -> None:
    df = load_trades(csv_path)
    if df.empty:
        st.info(f"No data yet at `{csv_path.relative_to(PROJECT_ROOT)}`.")
        return

    summary, trades = compute_summary(df)

    n_stats = 5 if show_capital else (4 if summary["days_evaluated"] is None else 5)
    stat_cols = st.columns(n_stats)
    i = 0
    if show_capital:
        stat_tile(stat_cols[i], "Current Paper Capital", f"Rs {load_capital():,.2f}"); i += 1
    if summary["days_evaluated"] is not None:
        stat_tile(stat_cols[i], "Days Evaluated", str(summary["days_evaluated"])); i += 1
    stat_tile(stat_cols[i], "Trades Taken", str(summary["trades_taken"])); i += 1
    win_rate = summary["win_rate_excl_eod"]
    stat_tile(stat_cols[i], "Win Rate (excl. EOD)", f"{win_rate:.1f}%" if win_rate is not None else "n/a"); i += 1
    avg = summary["avg_pct_change"]
    stat_tile(stat_cols[i], "Avg P&L / Trade", f"{avg:+.2f}%", tone="good" if avg >= 0 else "bad")

    if len(trades):
        st.markdown("")
        chart_col, outcome_col = st.columns([2, 1])
        has_rupee_pnl = "pnl_rupees" in trades.columns
        with chart_col:
            st.markdown("**Equity curve** " + ("(rupees)" if has_rupee_pnl else "(cumulative % return)"))
            render_equity_chart(trades, has_rupee_pnl)
        with outcome_col:
            st.markdown(f"**Outcomes** · {summary['wins']} win / {summary['losses']} loss / {summary['eod_closes']} EOD")
            render_outcome_bar(summary)
            st.caption("Daily P&L")
            render_daily_pnl_chart(trades)

        st.subheader("Trade log")
        display_trades = trades.reset_index(drop=True).copy()
        column_config = {}
        column_order = None
        button_key = f"view_chart_{title.lower().replace(' ', '_')}"
        selected_state_key = f"{button_key}_selected"
        hidden_columns = []

        if "pct_change" in display_trades.columns:
            # pct_change is stored as a fraction (0.1112 = +11.12%) - pre-multiply
            # rather than trust column_config's ambiguous "percent" preset.
            pct_position = display_trades.columns.get_loc("pct_change")
            display_trades.insert(pct_position, "P&L %", display_trades["pct_change"] * 100)
            column_config["P&L %"] = st.column_config.NumberColumn("P&L %", format="%.2f%%")
            hidden_columns.append("pct_change")

        has_chart_column = "chart_path" in display_trades.columns
        if has_chart_column:
            display_trades["view"] = display_trades["chart_path"].apply(
                lambda p: ":material/visibility: View Candle" if p else ""
            )
            hidden_columns.append("chart_path")

        if hidden_columns:
            column_order = [c for c in display_trades.columns if c not in hidden_columns and c != "view"]
            if "view" in display_trades.columns:
                column_order.append("view")

        if "date" in display_trades.columns:
            # Inserting blank subtotal rows forces originally-integer columns (strike,
            # lots) to upcast to float (NaN can't live in an int64 column) - force them
            # back to whole-number display via column_config rather than leaving "1.0".
            for int_col in ("strike", "lots"):
                if int_col in display_trades.columns:
                    column_config[int_col] = st.column_config.NumberColumn(int_col, format="%d")

            grouped_rows = []
            for day, group in display_trades.groupby("date", sort=False):
                grouped_rows.append(group)
                summary = {}
                for col in display_trades.columns:
                    if col == "date":
                        summary[col] = f"{day} - TOTAL"
                    elif col == "P&L %":
                        # Summing per-trade percentages is meaningless when trades have
                        # different position sizes (a naive sum can even show the wrong
                        # sign vs. the real rupee outcome). Weight by invested amount when
                        # we have it (paper trades); fall back to a plain average otherwise
                        # (backtest rows have no position-size data at all).
                        if "invested_amount" in group.columns and "pnl_rupees" in group.columns and group["invested_amount"].sum():
                            summary[col] = round(group["pnl_rupees"].sum() / group["invested_amount"].sum() * 100, 2)
                        else:
                            summary[col] = round(group["P&L %"].mean(), 2)
                    elif col == "pnl_rupees":
                        summary[col] = round(group["pnl_rupees"].sum(), 2)
                    elif pd.api.types.is_numeric_dtype(display_trades[col]):
                        summary[col] = float("nan")
                    else:
                        summary[col] = ""
                grouped_rows.append(pd.DataFrame([summary]))

                if "pnl_rupees" in display_trades.columns:
                    brokerage = len(group) * ORDERS_PER_TRADE * BROKERAGE_PER_ORDER
                    kite_share = KITE_MONTHLY_COST / KITE_BILLING_DAYS
                    net = round(group["pnl_rupees"].sum() - brokerage - kite_share, 2)
                    net_row = {col: (float("nan") if pd.api.types.is_numeric_dtype(display_trades[col]) else "") for col in display_trades.columns}
                    net_row["date"] = f"{day} - NET (after ~Rs {brokerage:,.0f} brokerage + Rs {kite_share:,.2f} Kite/day)"
                    net_row["pnl_rupees"] = net
                    grouped_rows.append(pd.DataFrame([net_row]))
            display_trades = pd.concat(grouped_rows, ignore_index=True)

        if has_chart_column:
            # Bind the click handler to the FINAL display_trades (post day-grouping,
            # same object st.dataframe below actually renders) - click["row"] is a
            # position in that final table, and indexing into an earlier, shorter
            # pre-grouping version was a real bug (wrong/out-of-bounds rows once
            # TOTAL/NET summary rows shifted everything after them).
            def _handle_view_click(key=button_key, sel_key=selected_state_key, df=display_trades):
                click = st.session_state[key]
                st.session_state[sel_key] = df.iloc[click["row"]]["chart_path"]

            column_config["view"] = st.column_config.ButtonColumn(
                "Candle Chart",
                help="Click to view the 2-hour, 5-min candle chart of this trade's option premium around entry",
                on_click=_handle_view_click,
                key=button_key,
            )

        st.dataframe(display_trades, width="stretch", column_config=column_config, column_order=column_order)

        pdf_source = display_trades[column_order] if column_order else display_trades
        pdf_source = pdf_source.drop(columns=["view"], errors="ignore")
        st.download_button(
            "\U0001F5A8 Download as PDF", data=dataframe_to_pdf_bytes(pdf_source, title),
            file_name=f"{title.replace(' ', '_').lower()}.pdf", mime="application/pdf",
            key=f"pdf_{title}",
        )

        selected_path = st.session_state.get(selected_state_key)
        if selected_path:
            if Path(selected_path).exists():
                st.image(selected_path, caption=Path(selected_path).name)
            else:
                st.warning(f"Chart file not found: {selected_path}")


def render_daily_summary(csv_path: Path) -> None:
    df = load_trades(csv_path)
    if df.empty:
        st.info(f"No data yet at `{csv_path.relative_to(PROJECT_ROOT)}`.")
        return
    trades = df[df["direction"] != "NO_TRADE"] if "direction" in df.columns and (df["direction"] == "NO_TRADE").any() else df
    if trades.empty or "date" not in trades.columns:
        st.info("No trades yet.")
        return

    # Filter by a separate parsed series, not the "date" column itself - it stays
    # in its original string form for the chart/table code below, unchanged.
    date_series = pd.to_datetime(trades["date"]).dt.date
    min_date, max_date = date_series.min(), date_series.max()

    period = st.radio(
        "Period", ["All time", "This month", "Custom range"],
        horizontal=True, key="daily_summary_period",
    )
    if period == "This month":
        today = date.today()
        start_date, end_date = today.replace(day=1), today
    elif period == "Custom range":
        col1, col2 = st.columns(2)
        start_date = col1.date_input("From", value=min_date, min_value=min_date, max_value=max_date, key="daily_summary_from")
        end_date = col2.date_input("To", value=max_date, min_value=min_date, max_value=max_date, key="daily_summary_to")
    else:
        start_date, end_date = min_date, max_date

    trades = trades[(date_series >= start_date) & (date_series <= end_date)]
    if trades.empty:
        st.info("No trades in the selected period.")
        return

    has_rupees = "pnl_rupees" in trades.columns
    has_invested = "invested_amount" in trades.columns

    rows = []
    for day, group in trades.groupby("date", sort=True):
        wins = int((group["exit_reason"] == "TARGET").sum())
        losses = int((group["exit_reason"] == "STOPLOSS").sum())
        eod = int((group["exit_reason"] == "EOD_CLOSE").sum())
        decided = wins + losses
        win_rate = wins / decided * 100 if decided else None

        if has_rupees:
            gross = group["pnl_rupees"].sum()
            brokerage = len(group) * ORDERS_PER_TRADE * BROKERAGE_PER_ORDER
            kite_share = KITE_MONTHLY_COST / KITE_BILLING_DAYS
            net = gross - brokerage - kite_share
        else:
            gross = brokerage = net = None

        if has_rupees and has_invested and group["invested_amount"].sum():
            pct_total = gross / group["invested_amount"].sum() * 100
        else:
            pct_total = (group["pct_change"] * 100).mean()

        rows.append({
            "Date": day,
            "Trades": len(group),
            "Win": wins, "Loss": losses, "EOD": eod,
            "Win Rate": f"{win_rate:.1f}%" if win_rate is not None else "n/a",
            "P&L %": f"{pct_total:+.2f}%",
            "Gross P&L": f"Rs {gross:+,.2f}" if gross is not None else "n/a",
            "Brokerage + Kite": f"Rs {brokerage + KITE_MONTHLY_COST / KITE_BILLING_DAYS:,.2f}" if brokerage is not None else "n/a",
            "Net P&L": f"Rs {net:+,.2f}" if net is not None else "n/a",
        })

    summary_df = pd.DataFrame(rows)

    st.markdown("**Daily P&L**")
    render_daily_pnl_chart(trades)

    st.subheader("Per-day breakdown")
    st.dataframe(summary_df, width="stretch", hide_index=True)
    st.download_button(
        "\U0001F5A8 Download as PDF", data=dataframe_to_pdf_bytes(summary_df, "Daily Summary"),
        file_name=f"daily_summary_{start_date}_to_{end_date}.pdf", mime="application/pdf",
        key="pdf_daily_summary",
    )

    if has_rupees:
        total_gross = trades["pnl_rupees"].sum()
        total_days = trades["date"].nunique()
        total_trades = len(trades)
        total_brokerage = total_trades * ORDERS_PER_TRADE * BROKERAGE_PER_ORDER
        total_kite = total_days * (KITE_MONTHLY_COST / KITE_BILLING_DAYS)
        total_net = total_gross - total_brokerage - total_kite
        cols = st.columns(4)
        stat_tile(cols[0], "Days", str(total_days))
        stat_tile(cols[1], "Total Trades", str(total_trades))
        stat_tile(cols[2], "Gross P&L (all days)", f"Rs {total_gross:+,.2f}", tone="good" if total_gross >= 0 else "bad")
        stat_tile(cols[3], "Net P&L (all days, after costs)", f"Rs {total_net:+,.2f}", tone="good" if total_net >= 0 else "bad")


# ---------- Layout ----------

inject_css()

st.markdown(
    """<div class="sm-hero">
         <div>
           <p class="title">\U0001F4C8 Share Market Bro</p>
           <p class="subtitle">Autonomous NIFTY 50 options bot &middot; paper trading &middot; one genuine trade at a time</p>
         </div>
       </div>""",
    unsafe_allow_html=True,
)

auto_refresh = render_bot_control()

st.divider()

tab1, tab2, tab3 = st.tabs(["\U0001F4CA Backtest Results", "\U0001F4B9 Paper Trading Results", "\U0001F4C5 Daily Summary"])
with tab1:
    render_report(BACKTEST_CSV, "Backtest Results")
with tab2:
    render_report(PAPER_TRADES_CSV, "Paper Trading Results", show_capital=True)
with tab3:
    render_daily_summary(PAPER_TRADES_CSV)

# Must be the LAST thing in the script - st.rerun() halts execution immediately,
# so anything placed after the point it's called from never renders. Found live
# 2026-07-17: having this inside render_bot_control() (called before the tabs)
# meant the tabs/tables never rendered at all whenever auto-refresh was on.
if auto_refresh:
    time.sleep(10)
    st.rerun()
