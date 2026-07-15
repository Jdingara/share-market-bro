"""
Trains a second Gradient Boosting model specifically on 5-minute candles, so
paper_trader.py can generate signals earlier in the day.

Why this exists: the standard model (ml_signal.py) is trained on 15-minute
candles, and its features (RSI etc.) need 16 candles' worth of history before
they're computable - a hard 4-hour wait from market open (confirmed against
all real trading days so far: first signal never earlier than ~13:15). At
5-minute candles, that same 16-candle minimum only takes ~1h20m. The model
can't just be pointed at faster data live, though - it learned patterns from
how RSI/momentum look at 15-minute resolution, which is statistically
different from 5-minute resolution. This trains a dedicated model instead.

Saved under model_type "gradient_boosting_5min" (reuses ml_signal.py's
save_models/load_models - "model_type" is just a filename component, not
enforced against the ModelType Literal at runtime).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_signal import build_labeled_dataset, save_models, time_based_split, train_models

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_TYPE_5MIN = "gradient_boosting_5min"


def main() -> None:
    daily_df = pd.read_csv(PROJECT_ROOT / "data" / "historical" / "NIFTY_50_day.csv")
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df = daily_df.sort_values("date").reset_index(drop=True)

    intraday_df = pd.read_csv(PROJECT_ROOT / "data" / "historical" / "NIFTY_50_5minute.csv")
    intraday_df["date"] = pd.to_datetime(intraday_df["date"])
    intraday_df = intraday_df.sort_values("date").reset_index(drop=True)

    print("Building labeled dataset from 5-minute candles (this simulates every candle's hypothetical CALL/PUT outcome)...")
    labeled_df = build_labeled_dataset(daily_df, intraday_df)
    print(f"Labeled {len(labeled_df)} candles across {labeled_df['date'].nunique()} days.")
    print(f"Call win rate overall: {labeled_df['call_label'].mean():.1%} | Put win rate overall: {labeled_df['put_label'].mean():.1%}\n")

    train_df, test_df = time_based_split(labeled_df)
    print(f"Train: {train_df['date'].nunique()} days ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"Test:  {test_df['date'].nunique()} days ({test_df['date'].min()} to {test_df['date'].max()})\n")

    print(f"--- Training {MODEL_TYPE_5MIN} ---")
    call_model, put_model, call_threshold, put_threshold = train_models(train_df, "gradient_boosting")
    save_models(MODEL_TYPE_5MIN, call_model, put_model, call_threshold, put_threshold)
    print(f"Calibrated thresholds (from training data only): call={call_threshold:.2f}  put={put_threshold:.2f}")

    # Honest test-set read, same as the original 15-min comparison did.
    from ml_features import FEATURE_NAMES
    X_test = test_df[FEATURE_NAMES]
    call_proba = call_model.predict_proba(X_test)[:, 1]
    put_proba = put_model.predict_proba(X_test)[:, 1]
    call_mask = call_proba >= call_threshold
    put_mask = put_proba >= put_threshold
    call_n, put_n = call_mask.sum(), put_mask.sum()
    call_prec = test_df.loc[call_mask, "call_label"].mean() * 100 if call_n else float("nan")
    put_prec = test_df.loc[put_mask, "put_label"].mean() * 100 if put_n else float("nan")
    print(f"\nHeld-out test-set precision at calibrated threshold: CALL {call_prec:.1f}% (n={call_n})  PUT {put_prec:.1f}% (n={put_n})")
    print(f"Models saved to {PROJECT_ROOT / 'data' / 'models'}")


if __name__ == "__main__":
    main()
