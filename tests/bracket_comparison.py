"""Not an automated test - compares the current +10%/-5% target/stop-loss
bracket against a proposed +5%/-3% one (edit the (target, stop) list below to
try others), holding the entry signal constant so this isolates the exit
rule's effect. Monkey-patches backtester's module-level TARGET_PCT/
STOP_LOSS_PCT at runtime rather than editing the file, since simulate_trade
reads them fresh from the module namespace on every call.

Built 2026-08-15 to answer a real question (would +5%/-3% be more
profitable) - it wasn't: same win/loss split as the current bracket on the
held-out test period, but a worse reward:risk ratio (2:1 vs 1.67:1) meant a
quarter of the total P&L on the exact same trades. See PROJECT_STATUS.md."""

import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

import backtester
from ml_features import attach_vix
from ml_signal import build_labeled_dataset, load_models, generate_ml_signal, time_based_split
from signal_engine import generate_signal

# Same held-out test period used everywhere else in this project - running the
# full cached history instead would include days the model was TRAINED on,
# inflating its apparent performance with in-sample results.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
daily_df = pd.read_csv(PROJECT_ROOT / "data" / "historical" / "NIFTY_50_day.csv")
daily_df["date"] = pd.to_datetime(daily_df["date"])
intraday_df = pd.read_csv(PROJECT_ROOT / "data" / "historical" / "NIFTY_50_15minute.csv")
intraday_df["date"] = pd.to_datetime(intraday_df["date"])
vix_daily_df = pd.read_csv(PROJECT_ROOT / "data" / "historical" / "INDIA_VIX_day.csv")
vix_daily_df["date"] = pd.to_datetime(vix_daily_df["date"])
vix_intraday_df = pd.read_csv(PROJECT_ROOT / "data" / "historical" / "INDIA_VIX_15minute.csv")
vix_intraday_df["date"] = pd.to_datetime(vix_intraday_df["date"])
daily_df = attach_vix(daily_df, vix_daily_df)
intraday_df = attach_vix(intraday_df, vix_intraday_df)
_, test_df = time_based_split(build_labeled_dataset(daily_df, intraday_df))
TEST_START, TEST_END = test_df["date"].min(), test_df["date"].max()
print(f"Held-out test period: {TEST_START} to {TEST_END} ({test_df['date'].nunique()} days)")


def summarize(label: str, results) -> dict:
    trades = [r for r in results if r.direction != "NO_TRADE"]
    wins = [t for t in trades if t.exit_reason == "TARGET"]
    losses = [t for t in trades if t.exit_reason == "STOPLOSS"]
    eod = [t for t in trades if t.exit_reason == "EOD_CLOSE"]
    decided = len(wins) + len(losses)
    win_rate = len(wins) / decided * 100 if decided else float("nan")
    avg_pct = sum(t.pct_change for t in trades) / len(trades) * 100 if trades else float("nan")
    total_pct = sum(t.pct_change for t in trades) * 100
    print(f"{label:>28}: {len(trades):>3} trades | win/loss/eod {len(wins)}/{len(losses)}/{len(eod)} "
          f"| win rate {win_rate:5.1f}% | avg {avg_pct:+6.2f}% | sum {total_pct:+7.2f}%")
    return {"label": label, "trades": len(trades), "win_rate": win_rate, "avg_pct": avg_pct, "total_pct": total_pct}


call_model, put_model, call_threshold, put_threshold = load_models("gradient_boosting")
ml_signal_fn = partial(generate_ml_signal, call_model=call_model, put_model=put_model,
                        call_threshold=call_threshold, put_threshold=put_threshold)

for target, stop, label_suffix in [(0.10, 0.05, "current +10%/-5%"), (0.05, 0.03, "proposed +5%/-3%")]:
    backtester.TARGET_PCT = target
    backtester.STOP_LOSS_PCT = stop
    print(f"\n--- {label_suffix} ---")
    rule_results = backtester.run_backtest(signal_fn=generate_signal, start_date=TEST_START, end_date=TEST_END)
    summarize(f"Rule-based ({label_suffix})", rule_results)
    ml_results = backtester.run_backtest(signal_fn=ml_signal_fn, start_date=TEST_START, end_date=TEST_END)
    summarize(f"Gradient Boosting ({label_suffix})", ml_results)
