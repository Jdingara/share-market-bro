"""
Not an automated test - a CALL-specific confidence-threshold scan, isolated
from PUT entirely (unlike compare_ml_vs_rules.py's blended win-rate table).

This recreates the analysis that found (2026-07-24) CALL's precision collapses
at high confidence thresholds in both the primary (15-min) and early-session
(5-min) models - the reason put_only became the default. Re-run this any time
the feature set changes (e.g. adding VIX/Greeks/FII-DII, 2026-08-12 onward) to
see whether CALL's calibration actually improved, rather than assuming it did.

Trains fresh on the TRAINING split only (never peeks at test data to pick a
threshold - same discipline as ml_signal.calibrate_threshold), then reports
precision/support at every threshold from 0.30 to 0.85 against the held-out
TEST split, for CALL only, for every model type and both candle resolutions.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ml_features import attach_vix
from ml_signal import MODEL_TYPES, _build_estimator, build_labeled_dataset, time_based_split
from ml_features import FEATURE_NAMES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS = np.arange(0.30, 0.85, 0.05)


def _load(interval: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_df = pd.read_csv(PROJECT_ROOT / "data" / "historical" / "NIFTY_50_day.csv")
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df = daily_df.sort_values("date").reset_index(drop=True)

    intraday_df = pd.read_csv(PROJECT_ROOT / "data" / "historical" / f"NIFTY_50_{interval}.csv")
    intraday_df["date"] = pd.to_datetime(intraday_df["date"])
    intraday_df = intraday_df.sort_values("date").reset_index(drop=True)

    vix_daily_df = pd.read_csv(PROJECT_ROOT / "data" / "historical" / "INDIA_VIX_day.csv")
    vix_daily_df["date"] = pd.to_datetime(vix_daily_df["date"])
    daily_df = attach_vix(daily_df, vix_daily_df)

    vix_intraday_df = pd.read_csv(PROJECT_ROOT / "data" / "historical" / f"INDIA_VIX_{interval}.csv")
    vix_intraday_df["date"] = pd.to_datetime(vix_intraday_df["date"])
    intraday_df = attach_vix(intraday_df, vix_intraday_df)

    return daily_df, intraday_df


def scan_call_precision(label: str, interval: str, model_type: str) -> None:
    daily_df, intraday_df = _load(interval)
    labeled_df = build_labeled_dataset(daily_df, intraday_df)
    train_df, test_df = time_based_split(labeled_df)

    call_model = _build_estimator(model_type).fit(train_df[FEATURE_NAMES], train_df["call_label"])
    test_proba = call_model.predict_proba(test_df[FEATURE_NAMES])[:, 1]
    baseline = test_df["call_label"].mean() * 100

    print(f"\n=== {label} ({model_type}, {interval} candles) ===")
    print(f"Test period: {test_df['date'].min()} to {test_df['date'].max()} ({test_df['date'].nunique()} days) | "
          f"unconditional CALL win rate: {baseline:.1f}% (n={len(test_df)})")
    print(f"{'threshold':>10} {'precision':>10} {'n':>6}")
    for t in THRESHOLDS:
        mask = test_proba >= t
        n = int(mask.sum())
        precision = test_df.loc[mask, "call_label"].mean() * 100 if n else float("nan")
        print(f"{t:>10.2f} {precision:>9.1f}% {n:>6}")


def main() -> None:
    for model_type in MODEL_TYPES:
        scan_call_precision(model_type.upper().replace("_", " "), "15minute", model_type)
    scan_call_precision("GRADIENT BOOSTING (early-session)", "5minute", "gradient_boosting")


if __name__ == "__main__":
    main()
