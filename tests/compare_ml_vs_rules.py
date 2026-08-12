"""
Not an automated test - the honest side-by-side comparison: replays the SAME
held-out (test) period through the rule-based signal engine and every ML
model type (Random Forest, Logistic Regression, Gradient Boosting/XGBoost),
using the exact same backtesting/bracket-simulation machinery, so all the
trade-frequency/win-rate/P&L numbers are directly comparable.
"""

import sys
from dataclasses import asdict
from functools import partial
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from backtester import run_backtest
from ml_features import attach_vix
from ml_signal import MODEL_TYPES, build_labeled_dataset, load_models, time_based_split, generate_ml_signal
from signal_engine import generate_signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent

daily_df = pd.read_csv(PROJECT_ROOT / "data" / "historical" / "NIFTY_50_day.csv")
daily_df["date"] = pd.to_datetime(daily_df["date"])
daily_df = daily_df.sort_values("date").reset_index(drop=True)

intraday_df = pd.read_csv(PROJECT_ROOT / "data" / "historical" / "NIFTY_50_15minute.csv")
intraday_df["date"] = pd.to_datetime(intraday_df["date"])
intraday_df = intraday_df.sort_values("date").reset_index(drop=True)

# India VIX features (added 2026-08-12) - must be merged in here too, not just at
# training time: the loaded models were fit expecting real vix_level/vix_change/
# vix_pctile values, and without this merge extract_features silently falls back
# to neutral defaults (0.0/0.0/0.5) instead of raising - a comparison run without
# this would quietly evaluate the VIX-aware models on fake, constant VIX inputs.
vix_daily_df = pd.read_csv(PROJECT_ROOT / "data" / "historical" / "INDIA_VIX_day.csv")
vix_daily_df["date"] = pd.to_datetime(vix_daily_df["date"])
daily_df = attach_vix(daily_df, vix_daily_df)

vix_intraday_df = pd.read_csv(PROJECT_ROOT / "data" / "historical" / "INDIA_VIX_15minute.csv")
vix_intraday_df["date"] = pd.to_datetime(vix_intraday_df["date"])
intraday_df = attach_vix(intraday_df, vix_intraday_df)

# Recompute the exact same chronological split used during training, so we
# know precisely which days are genuinely held-out (never seen in training).
labeled_df = build_labeled_dataset(daily_df, intraday_df)
_, test_df = time_based_split(labeled_df)
test_start, test_end = test_df["date"].min(), test_df["date"].max()
print(f"Held-out test period: {test_start} to {test_end} ({test_df['date'].nunique()} days)\n")


def summarize(label: str, results) -> dict:
    trades = [r for r in results if r.direction != "NO_TRADE"]
    wins = [t for t in trades if t.exit_reason == "TARGET"]
    losses = [t for t in trades if t.exit_reason == "STOPLOSS"]
    eod = [t for t in trades if t.exit_reason == "EOD_CLOSE"]
    decided = len(wins) + len(losses)
    win_rate = len(wins) / decided * 100 if decided else float("nan")
    avg_pct = sum(t.pct_change for t in trades) / len(trades) * 100 if trades else float("nan")

    print(f"=== {label} ===")
    print(f"Trades taken: {len(trades)}  |  Win/Loss/EOD: {len(wins)}/{len(losses)}/{len(eod)}")
    print(f"Win rate (excl. EOD): {win_rate:.1f}%  |  Average P&L per trade: {avg_pct:+.2f}%\n")
    return {"label": label, "trades": len(trades), "wins": len(wins), "losses": len(losses),
            "eod": len(eod), "win_rate": win_rate, "avg_pct": avg_pct}


summary_rows = []

rule_results = run_backtest(signal_fn=generate_signal, start_date=test_start, end_date=test_end)
summary_rows.append(summarize("RULE-BASED", rule_results))
pd.DataFrame([asdict(r) for r in rule_results]).to_csv(PROJECT_ROOT / "data" / "backtest_results" / "rule_based_test_period.csv", index=False)

for model_type in MODEL_TYPES:
    call_model, put_model, call_threshold, put_threshold = load_models(model_type)
    ml_signal_fn = partial(generate_ml_signal, call_model=call_model, put_model=put_model,
                            call_threshold=call_threshold, put_threshold=put_threshold)
    results = run_backtest(signal_fn=ml_signal_fn, start_date=test_start, end_date=test_end)
    summary_rows.append(summarize(model_type.upper().replace("_", " "), results))
    pd.DataFrame([asdict(r) for r in results]).to_csv(PROJECT_ROOT / "data" / "backtest_results" / f"ml_{model_type}_test_period.csv", index=False)

print("=== SUMMARY TABLE ===")
print(pd.DataFrame(summary_rows).to_string(index=False))
