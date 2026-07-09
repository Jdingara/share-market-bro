# NIFTY Options Auto-Trading Bot — Project Status

This file is the single source of truth for this project's goal, decisions, and current progress. Any AI assistant (Claude, Codex, or otherwise) or human picking this up should read this file first before making changes. Keep it updated as the project progresses — update the "Current Status" and "Open Decisions" sections whenever a phase completes or a new decision is made.

## Goal

Build an autonomous bot that trades NIFTY 50 index options (calls and puts) through Zerodha Kite Connect, for the account owner's personal use. The bot must analyze the market itself and decide when/what to trade — it is not a manual-signal tool.

## Core Trading Rules (decided, stable — do not change without explicit user request)

- **One trade per day, maximum.** The bot waits for its best setup, takes one entry, exits, then stops trading for the rest of that day — win or lose, no re-entry, no second attempt.
- **Exit bracket:** close the position at **+10% profit** (the goal) or **-10% loss** (the hard stop), whichever is hit first.
- **Direction:** both calls and puts are in scope. The bot decides bullish vs. bearish from its own analysis, and **sits out entirely** if no setup is clear ("no trade today" is a valid, expected outcome — never force a trade).
- **Underlying:** NIFTY 50 index options first; Bank Nifty is an explicit later addition, not in scope yet.
- **Position sizing:** NOT YET DECIDED — see Open Decisions below. Do not assume a specific model (fixed amount vs. daily-compounding reinvestment) without confirming with the user first.
- **Validation order, strictly in this sequence — do not skip ahead:**
  1. Backtest the strategy against historical data first, to measure a real historical win rate (never assume a strategy works without measuring it).
  2. Paper trading — simulated orders, real live market data, running on the user's own laptop during market hours (9:15am–3:30pm IST). No real money at risk.
  3. Live trading — only after paper trading has proven out over a meaningful sample of trading days.

## Broker & Account Setup

- **Broker:** Zerodha Kite Connect. Chosen over Angel One (genuinely free) and Upstox/Groww specifically because Zerodha has by far the most mature documentation and community for this kind of Python/algo-trading build — worth the small recurring cost.
- **Billing:** Kite Connect API access is credit-based: 500 credits ≈ ₹500 (inclusive of GST), valid for 30 days per app, auto-renewing because the user linked their Zerodha account for billing. The "Connect" plan tier is required (not "Personal" — Personal excludes historical data and live market quotes/WebSockets, which this project needs).
- **Zerodha app name:** `NiftyOptionsBot`. API key and Zerodha Client ID are stored only in `.env` (gitignored) - not written here on purpose, to keep this file safe to store in git/GitHub.
- **Credentials location:** `.env` in the project root (gitignored, never committed). Contains `KITE_API_KEY`, `KITE_API_SECRET`, `KITE_USER_ID`, `KITE_PASSWORD`, `KITE_TOTP_SECRET`. See `.env.example` for the template/field names.
- **TOTP 2FA is enabled** on the Zerodha account (required for automated daily login — Kite access tokens expire every day).
- **Credit renewal reminder:** the app's 500 credits were purchased 2026-07-07 and are valid ~30 days — expect to need renewal around **2026-08-06**, via the Billing page on developers.kite.trade.

## Non-Obvious Technical Findings (read before debugging auth issues)

1. **Kite Connect's login flow requires a one-time manual browser consent.** The first time a given API key + Zerodha account pair is used, Zerodha shows an "authorize this app" consent screen that a plain HTTP redirect chain cannot get past (tried `skip_session=true` — did not work). This had to be resolved by manually visiting `https://kite.zerodha.com/connect/login?api_key=<key>&v=3` in a real browser once, logging in, and clicking Authorize. After that one-time consent, the fully-automated `requests`-based login in `src/auth.py` works correctly with no browser involved. **If auth ever mysteriously breaks after being fine, check whether a NEW Kite Connect app was created (each new app needs its own one-time consent) before assuming the code is broken.**
2. **Kite Connect access tokens expire daily** — this is expected, not a bug. `src/auth.py` automates the login (password + TOTP) daily via `requests` calls to Zerodha's web login endpoints, since there is no officially documented headless login API. This is a widely-used community pattern, not part of Kite's official public API — if it starts failing, Zerodha's internal login endpoints may have changed shape.
3. **`pandas-ta` was deliberately dropped from dependencies** — it depends on `numba`, which does not support the Python 3.14 installed on the dev machine. EMA and RSI are simple enough to hand-implement directly in pandas; do not re-add `pandas-ta` or another indicator library without checking Python version compatibility first.
4. **SEBI's static-IP mandate (effective 2026-04-01)** only restricts *order placement* API calls, not login/data-fetching. It does not block anything in Phases 0–4 (auth, historical data, signal engine, backtesting, paper trading). It only becomes relevant at the live-trading phase (Phase 5), when real orders start being placed and a static IP must be registered with Zerodha.
5. **Index instruments (like NIFTY 50) show `volume: 0`** in historical data — this is normal, indices don't carry their own traded volume, only individual stocks/derivatives do.
6. **Expired option contracts are completely unavailable via Kite Connect** — confirmed directly by checking the live instrument list (earliest listed expiry is always today's date). There is no way to fetch real historical option premiums for backtesting; `src/options_pricing.py` simulates them via Black-Scholes instead. This is a real, permanent constraint of the API, not a bug to fix later.
7. **Transient network errors WILL happen during a 6+ hour unattended run and will kill the process if not handled** — confirmed by a real crash (`ConnectionResetError(10054)`) that silently killed `paper_trader.py` for nearly an hour before being noticed. `paper_trader.py` now retries network calls with backoff and has an outer catch-all that logs and continues rather than dying (see Current Status). If writing any other long-running unattended script (e.g. for Phase 5), apply the same pattern - don't assume a bare API call in a loop is safe unattended.
8. **RandomForestClassifier's `predict_proba` rarely gets close to 0 or 1 on noisy financial data** — on this project's data, max predicted confidence was only ~0.6-0.7 even on training data. Don't guess a confidence threshold (an initial guess of 0.6 turned out higher than the model's entire test-set output range, silently producing zero trades) - always inspect the actual predicted-probability distribution and training-set precision-at-threshold first.
9. **NSE changed NIFTY's weekly options expiry day from Thursday to Tuesday at some point before 2026-07-08** — confirmed live against Kite's instrument list, after a genuine live trading signal on 2026-07-08 failed to find a matching option contract (was computing expiry=Thursday 2026-07-16, which was never a real listed contract; real expiries were all Tuesdays: 2026-07-14, 07-21, 07-28...). **This wasted that day's one trade opportunity.** Fixed two ways: (1) `options_pricing.WEEKLY_EXPIRY_WEEKDAY` updated to Tuesday for backtesting/simulation (where no real instrument list exists for past dates, so a weekday guess is unavoidable); (2) live trading (`paper_trader.py`) no longer guesses a weekday at all — `option_lookup.find_nearest_valid_expiry()` now queries the actual live instrument list for the real nearest valid expiry, making it robust to NSE changing this convention again in the future without needing another code fix. All 3 ML models were retrained with the corrected assumption (see Phase 6 section - the comparison numbers shifted somewhat as a result, XGBoost remains the best performer).

## Full Roadmap

| Phase | Goal | Status |
|---|---|---|
| **0. Project setup & auth** | Python scaffold, Kite Connect credentials, automated daily TOTP login | ✅ Done, verified end-to-end with real account (2026-07-07) |
| **1. Historical data pipeline** | Pull & cache NIFTY 50 historical candles for backtesting | ✅ Done, verified — real daily candles pulled to `data/historical/` |
| **2. Signal engine (rule-based)** | EMA trend filter + Fibonacci retracement zones + RSI + candlestick confirmation, combined into one confluence rule producing a daily call/put/no-trade decision | ✅ Done, verified — see `src/indicators.py`, `src/signal_engine.py` |
| **3. Backtesting engine** | Replay the signal engine over historical data, simulate the +10%/-10% bracket, measure real win rate; tune thresholds until there's a genuine measured edge | ✅ Done, verified — see `src/options_pricing.py`, `src/backtester.py`. Result after fixing a 0DTE root cause: **+1.05%/trade, 63.6% win rate** over 18 trades (small sample, promising not proven — see below) |
| **4. Paper trading engine** | Real-time version: live data (REST polling, not WebSocket - see notes), same signal + bracket logic, simulated (not real) orders using REAL quoted premiums, full daily logging | ✅ Built and smoke-tested live — see `src/paper_trader.py`, `src/option_lookup.py`. Not yet run through a full trading day or multiple days |
| **5. Live trading** | Real Kite order API calls, safety guards (kill switch, daily loss circuit breaker), static-IP registration | ⬜ Not started |
| **6. ML enhancement** | Feature-engineer OHLCV + indicators, train classifiers (Random Forest, Logistic Regression, Gradient Boosting) on historical labeled outcomes, compare against the rule-based signal | ✅ Built, honestly 4-way compared, and **now the live default** — see below |

## Current Status (last updated: 2026-07-07)

Phases 0 through 4 and Phase 6 are complete. Phase 4 (paper trading) is built and smoke-tested but **not yet run through a full live trading day** - it now runs on the Phase 6 Gradient Boosting (XGBoost) signal by default (see the 2026-07-07 decision below), so tomorrow's run will be the first real test of that combination.

- **Phase 0/1**: `src/auth.py` (daily login) and `src/data_fetch.py` (historical candle fetch with correct per-interval pagination) verified live against the user's real Zerodha account.
- **Phase 2**: `src/indicators.py` (EMA, RSI, Fibonacci levels, 4 candlestick patterns) has 11 passing unit tests in `tests/test_indicators.py` against hand-computed/analytically-known values. `src/signal_engine.py` implements the confluence rule (`daily_trend_bias`, `fib_zones_for_today`, `generate_signal`) — see its module docstring for the important lookahead-bias contract callers must follow (`daily_df` = strictly prior days only, `intraday_df` = one day's candles only).
- **Phase 3**: Confirmed directly against the live Kite instrument list that **expired option contracts are not retrievable at all** (earliest listed expiry is always today) — so `src/options_pricing.py` simulates option premiums via Black-Scholes (historical/realized volatility standing in for implied vol) rather than replaying real historical option prices. 10 unit tests in `tests/test_options_pricing.py`, including an exact put-call-parity check. `src/backtester.py` replays `generate_signal` day-by-day over 132 cached trading days and simulates the +10%/-10% bracket via the pricing model.

  **Backtest result, initial (132 days, 2025-12 through 2026-07): 124 no-trade days, only 8 trades taken.** 2 hit +10%, 3 hit -10%, 3 closed at EOD. Win rate 40% (excl. EOD) / 25% (incl.). Average P&L -3.18%/trade.

  **Tuning round 1 — widened Fibonacci proximity (0.15%→0.3%) and relaxed candlestick confirmation to a 2-candle lookback window (`CANDLE_CONFIRM_LOOKBACK`):** trade count rose to 18, but win rate stayed at exactly 40% and P&L worsened slightly (-3.72%) — a consistent negative result across a bigger sample, not just noise. Diagnostic funnel analysis (`tests/diagnose_signal_funnel.py`) found the candlestick relaxation was a complete no-op (all 8 original trades' timing was unchanged) — the real bottleneck is the rarity of Fibonacci+RSI co-occurring at all (only 53 of ~1000 candles across the dataset), not candlestick pattern strictness.

  **Tuning round 2 — added Bollinger Bands** (`indicators.bollinger_bands`, window=20, 2 std-dev) as a second, independent "price at an extreme" trigger alongside Fibonacci, on the theory that a dynamic volatility-based zone might catch genuinely different setups. **Result: zero effect — all 18 trades still trace back to a Fibonacci zone (confirmed via the new `zone` field in `backtester.TradeResult`); Bollinger Bands never fired once in 132 days with these parameters.** Not a useful addition as configured.

  **Methodology fix — realistic intra-candle bracket checking:** the original backtester only checked the simulated premium at each 15-minute candle's *close*, letting some trades overshoot far past +/-10% (one hit -94.78%) before being caught, while other genuine intra-candle stop-outs were invisible if the candle's close happened to recover by the time it was checked. Fixed to check each candle's high/low range and exit AT the +10%/-10% threshold price the moment it's crossed (conservative assumption: if both target and stop were crossable within one candle, assume the stop hit first, since intra-candle order isn't knowable from OHLC data alone).

  **Root cause found — 0-2 days-to-expiry (0DTE) noise, not a bad entry signal:** checked days-to-expiry for each of the 18 trades and found every single one had 0-2 days left (median ~1 day), because `next_weekly_expiry()` always picked the *nearest* Thursday. Cross-checked underlying-index move vs. premium move per trade and found cases like the underlying moving +0.03% (favorably!) while the premium still swung -10% — a hallmark of extreme gamma/theta noise dominating near expiry, unrelated to whether the directional call was actually right.

  **Fix: `next_weekly_expiry()` now requires a minimum 3 days to expiry** (`MIN_DAYS_TO_EXPIRY` constant in `options_pricing.py`), rolling to the following week's expiry if the nearest one is too close. **Result after this single fix: win rate jumped to 63.6% (excl. EOD closes: 7 targets, 4 stops), and average P&L flipped positive: +1.05%/trade.** Individual trades now look sane (entry premiums ₹120-320, consistent with real time value; EOD closes range only -8.1% to +4.1%, no more wild overshoots). This is the current best result and a genuinely encouraging one — but from a still-small sample (18 trades, 11 of which are non-EOD), so treat as promising, not proven.

  **This means Phase 4 (paper trading) is no longer structurally blocked** — the core problem (0DTE noise swamping the signal) had a real, well-diagnosed fix, not just more threshold guessing.

- **Phase 4** is built: `src/paper_trader.py` runs the unmodified `generate_signal()` live during market hours via simple REST polling (not a WebSocket ticker - deliberate simplicity choice, more than sufficient for a one-trade-per-day strategy). `src/option_lookup.py` finds the real tradable option contract (confirmed NIFTY lot size is **65**, not 50) and fetches its **real quoted premium** via `kite.quote()` - no more Black-Scholes simulation from here on. No real orders are ever placed (`kite.place_order` is not called anywhere in this phase). 6 unit tests for the pure exit-decision logic (`tests/test_paper_trader.py`) and 4 for option lookup (`tests/test_option_lookup.py`).

  **Live smoke test succeeded** (2026-07-07, ~10:20am, 3-minute run): real login, real 65-day daily history fetch, real 34,860-instrument NFO dump, three clean 60-second polling cycles against real live data, correctly returned NO_TRADE each time (expected - early in the day, no confluence yet) with no errors.

  **Not yet done: a full trading day (or several) of paper trading to see a real entry/exit cycle happen live.** The smoke test proves the plumbing works, not that a live trade has actually been observed end-to-end yet.

**Next step: run the bot for one or more full trading days** (via the dashboard's Start button, or `py src/paper_trader.py` directly) to see real paper trades accumulate in `data/paper_trades/paper_trades.csv`, and compare that real performance against the backtest's +1.05%/trade expectation.

- **Dashboard added** (2026-07-07): `src/dashboard.py`, a local Streamlit web control panel — Start/Stop buttons for the paper trader (launched as a background subprocess, output redirected to `data/paper_trades/live_log.txt`), a live log viewer, and two report tabs (Backtest Results, Paper Trading Results) showing summary stats, an equity curve, and the full trade table, reading directly from the existing CSVs. Run via `py -m streamlit run src/dashboard.py`, opens at `http://localhost:8501`.
  - **Locked to localhost only** via `.streamlit/config.toml` — by default Streamlit also advertises a network/external URL, which would have made this reachable from outside the machine; since it can start/stop a bot tied to a real Zerodha account, that was closed off deliberately.
  - **Found and fixed a real bug during verification**: the background subprocess's output wasn't appearing in the live log for several seconds because Python buffers stdout differently when it's redirected to a file vs. a terminal. Fixed by launching with `-u` (unbuffered) so the log updates in real time, verified directly (log showed real progress within ~6 seconds after the fix, vs. staying empty before it).
  - Verified against real data: the dashboard's own summary computation matches `backtester.py`'s printed output exactly (18 trades, 63.6% win rate, +1.05% avg).

- **Real crash found and fixed during an actual live run** (2026-07-07, ~11:46am-12:40pm): a transient network blip (`requests.exceptions.ConnectionError` / `ConnectionResetError(10054)`) during a routine API call killed the entire `paper_trader.py` process with no recovery - it silently sat dead for nearly an hour before being noticed, having taken no trade. **Fixed in `paper_trader.py`**: added `_call_with_retry()` (retries any network call up to 5 times with a 5s backoff before giving up) wrapping every Kite API call in the loop, plus an outer try/except around the whole polling loop body that catches any remaining unexpected exception, logs it, pauses 30s, and keeps going rather than letting the process die. Verified with a fresh 2-minute live smoke test after the fix - runs clean. **This class of bug (unattended process dying silently on a transient error) is exactly the kind of thing to watch for once real money is involved (Phase 5) - worth extra scrutiny there, not just here.**

- **Phase 6 (ML signal engine, done ahead of the original sequence)**: built at the user's request, motivated by a sound observation — the rule-based engine only trades ~13.6% of days (18/132) because it needs one rigid, hand-picked pattern to align exactly, likely missing genuine opportunities that are "close but not exact." `src/ml_features.py` engineers 16 features per candle (RSI, Fibonacci/Bollinger distances, EMA trend ratio, momentum, realized vol, candlestick flags, time-of-day, day-of-week) - reuses the same indicator building blocks as the rule-based engine so the comparison is meaningful. `src/ml_signal.py` labels every historical candle by reusing `backtester.simulate_trade` (promoted from private to public specifically for this reuse) to simulate "would a CALL/PUT here have hit +10% before -10%," trains two conservative `RandomForestClassifier`s (shallow, min 10 samples/leaf - deliberately resistant to overfitting on ~130 days of data), and splits **chronologically** (train on first 70% of days, test on last 30%) rather than randomly, to avoid leaking correlated nearby days between train/test.

  **Confidence threshold had to be recalibrated, not guessed**: an initial arbitrary guess of 0.6 turned out to exceed the model's actual max output on the test period entirely (zero trades). Fixed by inspecting precision at various thresholds **using only the training set** (never peeking at test data to choose it) - 0.5 gives ~89% training precision vs. a 35-39% unconditional baseline, with a large enough sample (100+) to trust.

  **Extended same day (user's request): added Logistic Regression and Gradient Boosting (XGBoost) as two more model types**, trained and calibrated with the exact same rigor as Random Forest - `ml_signal.py` was generalized (`_build_estimator(model_type)`, one model per type) and threshold selection was formalized into `calibrate_threshold()` (scans training-set-only precision at each threshold, picks the lowest one clearing 70% precision with 30+ supporting samples) instead of eyeballed by hand per model. This also improved Random Forest's own result versus the first pass (which had used a manually-eyeballed 0.5 threshold instead of the formal 70%-precision rule).

  **Honest 4-way result on the held-out 40-day test period (2026-05-08 to 2026-07-06), identical bracket simulation for all** (first pass, before the expiry-weekday bug below was found and fixed):

  | Approach | Trades | Win/Loss/EOD | Win rate (excl. EOD) | Avg P&L |
  |---|---|---|---|---|
  | Rule-based | 7 | 3/0/4 | 100%* | +2.81% |
  | Random Forest | 29 | 19/10/0 | 65.5% | +3.10% |
  | Logistic Regression | 34 | 15/17/2 | 46.9% | **-0.63%** |
  | **Gradient Boosting (XGBoost)** | 36 | 27/8/1 | **77.1%** | **+5.45%** |

  *Rule-based's 100% is only 3 decided trades - too small to trust on its own.

  **Read honestly:** Logistic Regression (the simplest model, a straight linear boundary) came out **negative** - this is actually a valuable, deliberate sanity check, not a bad result: it shows the real relationship between these features and outcomes is **nonlinear**, which is why both tree-based approaches (Random Forest, XGBoost) outperform it - they're capturing real structure a straight line can't, not just adding complexity for its own sake. **Gradient Boosting is currently the strongest candidate by every metric** (most trades, highest win rate, best average P&L) - but it's also historically the model type most prone to overfitting small data if under-regularized, so this needs to be read as encouraging, not conclusive, until validated on more than one 40-day window (same caveat as always: candles within a day are correlated, so effective independent sample size is smaller than raw row counts suggest).

  **Re-run after fixing the expiry-weekday bug (see Non-Obvious Technical Finding #9) and retraining all 3 models with the corrected assumption:**

  | Approach | Trades | Win/Loss/EOD | Win rate (excl. EOD) | Avg P&L |
  |---|---|---|---|---|
  | Rule-based | 7 | 2/2/3 | 50.0% | -0.38% |
  | Random Forest | 36 | 21/14/1 | 60.0% | +2.08% |
  | Logistic Regression | 15 | 9/6/0 | 60.0% | +2.00% |
  | **Gradient Boosting (XGBoost)** | 38 | 24/14/0 | **63.2%** | **+2.63%** |

  Numbers shifted meaningfully once the simulation used the correct expiry day (makes sense - every simulated trade's time-to-expiry changed). **XGBoost remains the best performer on every metric**, though by a smaller margin than the first pass suggested, and Logistic Regression no longer looks as clearly bad - both good reminders that these numbers are sensitive to getting the underlying assumptions right, and that a single 40-day comparison shouldn't be over-read either way. The decision to run live with XGBoost (below) still stands on the corrected numbers.

  **Decision made (2026-07-07): `paper_trader.py` now defaults to Gradient Boosting (XGBoost)** for live paper trading, per the user's call after seeing the comparison. `paper_trader.py` gained a `--signal-source` flag (`rule_based`, `random_forest`, `logistic_regression`, or `gradient_boosting`, defaulting to `gradient_boosting`) via `_build_signal_fn()`, which wraps `ml_signal.generate_ml_signal` with the calibrated thresholds via `functools.partial` - matches `generate_signal`'s exact interface so the rest of the loop (entry, position-watching, exit, logging) is completely unchanged. `PaperTrade` now also logs which `signal_source` produced each trade. Smoke-tested after hours (market closed, so no live signal check, but confirmed model loading + startup wiring both work for `gradient_boosting` and `rule_based`). **First real market-hours run was 2026-07-08** - found the expiry-weekday bug (finding #9) when a genuine CALL signal fired but failed at the option-lookup step; fixed and verified against live data same day (see Current Status for the corrected comparison numbers). Compare via `tests/compare_ml_vs_rules.py` (loops over all `MODEL_TYPES`). Retrain all three via `py src/ml_signal.py`.

  **First-ever completed live paper trade (2026-07-08, 13:58-13:59): a genuine WIN.** After the expiry-bug fix, a real PUT signal (75% confidence) fired, found `NIFTY2671424200PE`, entered at Rs 265.25, and hit the +10% target within 2 minutes, exiting at Rs 298.20 (+12.42%). Full pipeline (signal -> option lookup -> entry -> monitoring -> exit -> logging -> correct one-trade-per-day stop) worked end-to-end for the first time.

  **Capital-based position sizing added same day** (`src/capital_manager.py`) - resolves the "position sizing" open decision from the very start of the project (see Open Decisions below for full detail). Starting capital Rs 20,000; the first real trade above correctly compounded it to **Rs 22,141.75**. `paper_trades.csv`'s existing row was retroactively updated with the new `lots`/`invested_amount`/`pnl_rupees`/`capital_after` columns for consistency, and `capital_state.json` was initialized to the post-trade balance so tomorrow's run picks up correctly rather than resetting.

  **`--max-trades-per-day` added to `paper_trader.py` (2026-07-08), default stays 1.** User's idea: rather than waiting 5-10 calendar days to gather 5-10 live trade results (one per day), temporarily raise this (e.g. to 20) to gather a real validation sample within a day or two, then switch back to 1 for the actual live-discipline track record. Implemented as a trade-slot counter (`trades_taken_today`) replacing the old boolean flag - each slot (successful trade, failed option lookup, or insufficient-capital skip) still counts against the limit and updates `capital` in-memory before the next slot, so multi-trade days compound correctly within the same run. Prints a visible warning when run with anything other than the default, as a reminder this is a validation setting, not the intended production behavior. Smoke-tested both modes after hours. **Important: raising this never lowers the bar for any individual trade** - every trade slot, whether the 1st or the 10th, independently has to clear the exact same calibrated confidence threshold. Nothing is ever forced just to fill slots.

  **Dashboard control added same day**: `render_bot_control()` in `dashboard.py` now has a "Max trades per day" number input (default 1) next to the Start button, passed through to `paper_trader.py --max-trades-per-day` when launched - so the next few validation days (agreed plan: run with a higher cap for ~2-3 days, review the win/loss sample, then drop back to 1) can be done entirely from the dashboard, matching the user's preference for the visual/GUI workflow over the command line.

  **Max capital per trade cap added same day (2026-07-08), user's own idea: "fix the max investment limit is 2 lakh."** Even though `capital_manager.py`'s compounding is otherwise uncapped (grows every win, shrinks every loss with no ceiling), the user wanted a hard ceiling on how much of that growing balance can ever be risked on a *single* trade - so a lucky streak that compounds the account to, say, ₹10 lakh doesn't suddenly mean one bad trade risks ₹10 lakh too. `deployable_capital(capital, max_per_trade=MAX_CAPITAL_PER_TRADE)` returns `min(capital, max_per_trade)` and is applied right before `calculate_affordable_lots()` in `paper_trader.py` - so lot sizing is always based on the capped amount, and any balance above ₹2,00,000 simply sits idle and untouched (verified: `apply_trade_pnl()` needed no changes at all, since it already only moves the invested portion). Wired through everywhere the trade-count cap was: a `--max-capital-per-trade` CLI flag on `paper_trader.py` (default ₹2,00,000), and a matching "Max capital per trade (Rs)" number input in the dashboard's Bot Control panel, both mirroring the existing `--max-trades-per-day` pattern. 3 new unit tests (`test_deployable_capital_*`) cover below-cap/above-cap/exactly-at-cap. Full suite now 53 tests, all passing.

**Real bug found and fixed on today's first 10-trade validation day (2026-07-09): rapid meaningless trades right at market close.** With `max_trades_per_day=10`, the loop kept accepting new entries all the way up to `MARKET_CLOSE_TIME` (3:30 PM), but any position opened after `FORCE_CLOSE_TIME` (3:25 PM) gets force-closed within the very next 15-second monitoring poll - it never gets a real chance to move toward +10%/-10%. This produced 6 near-instant back-to-back "trades" between 15:21-15:26, each measuring only a few seconds of random premium noise, inflating the trade count with results that don't reflect real signal performance. **Fixed**: the main loop in `paper_trader.py` now also stops looking for *new* entries once `now.time() >= FORCE_CLOSE_TIME`, even if trade slots remain unused - already-open positions are still force-closed exactly as before if still running at 3:25. Confirmed with the full test suite (53 passed).

**Agreed plan (2026-07-09): next 5 trading days (through ~2026-07-16) run with the trade-count cap effectively removed** (`--max-trades-per-day` set very high, e.g. 999, via the dashboard's now-uncapped "Max trades per day" input, `max_value` raised from 50 to 999) - so the real number of trades taken each day is whatever the ML confidence threshold (currently 33%) genuinely produces, not an artificial ceiling. Purpose: observe whether the confidence-based decision logic behaves sensibly under realistic, unrestricted conditions - zero trades on a quiet day is still a fully acceptable outcome, not a problem to fix. After this 5-day window, review the results together and decide: tune thresholds (see the CALL/PUT precision finding above), or move toward Phase 5 (real money) if the track record looks solid.

## Open Decisions (resolve before relevant phase starts)

- ~~Should a signal replace the rule-based one in paper trading, and which one?~~ **Resolved 2026-07-07**: paper trading now defaults to Gradient Boosting (XGBoost). Still open: how many real trading days of live paper trading are "enough" before trusting this for Phase 5 (see below) - this decision was made from one 40-day backtest window, so the live track record matters more than usual here.
- **XGBoost threshold tuning - deferred, not urgent (raised 2026-07-09).** During today's live run (multi-trade validation day, no trade fired for several hours), analyzed real win rate at different confidence thresholds on the 40-day held-out test set: PUT is well-calibrated (win rate rises cleanly from 56.6% at 0.20 to ~60-63% at 0.40-0.70, more confidence = more reliable), but **CALL is not** - win rate stays flat around 33-42% regardless of threshold and never climbs the way it should, meaning the CALL model's calibrated 70%-training-precision target didn't generalize to the test period. Neither side reliably crosses 70% win rate at a trustworthy sample size (only tiny, noisy samples do, e.g. n=2-6). A small nudge (33%->35%) tested and found to change almost nothing (CALL 40.8%->39.6%, PUT 58.1%->60.4%) - not worth doing on its own. **User's call: leave thresholds untouched for now, keep watching live results one more day (2026-07-10), and only revisit tuning - likely CALL-specific (e.g. a higher/separate threshold, or recalibrating with more data) - in a calm review session, not reactively mid-trading-day.** Also confirmed same day: when both CALL and PUT clear 33% simultaneously (happened twice today, ~13:15 and 13:30), the bot correctly treats it as ambiguous and skips - user explicitly chose to keep this conservative behavior rather than auto-picking the stronger side.

- ~~Position sizing / capital compounding~~ **Resolved 2026-07-08**: `src/capital_manager.py` (new module) tracks a running paper-trading balance, starting at **₹20,000**, persisted across days in `data/paper_trades/capital_state.json` (not reset each morning). Before every trade, `calculate_affordable_lots()` works out how many whole lots (lot size confirmed live: **65**) the current balance can afford at that day's real premium, rounding down - if that's 0, the bot logs why and sits out for the day rather than forcing a fractional position. `apply_trade_pnl()` updates the balance after each trade closes (only the invested portion moves; idle leftover capital that wasn't enough for one more lot is untouched). This compounds up on wins and down on losses, exactly as discussed - not a one-way ratchet. 8 unit tests, including a hand-verified match against the actual real trade from earlier today (₹20,000 → ₹22,141.75 on the +12.42% win). `paper_trader.py` and `dashboard.py` both updated to show real rupee P&L and lot counts, not just percentages.
- ~~Max capital risked per single trade as the balance compounds up~~ **Resolved 2026-07-08**: hard-capped at **₹2,00,000 (2 lakh)** per the user's explicit instruction. See Current Status above for implementation detail (`deployable_capital()`).
- **How many days/weeks of live paper trading is "enough" before considering Phase 5 (live trading)?** Not yet decided. Given the backtest edge was found on only 18 trades, real paper-trading confirmation should probably span several weeks of actual trading days (not just one) before trusting it with real money - exact threshold not yet agreed with the user.

## Repo Structure

```
Share_Market_Bro/
  PROJECT_STATUS.md     # this file - read first
  README.md             # setup/usage instructions
  .env                   # real credentials (gitignored, never commit)
  .env.example           # credential template
  requirements.txt
  src/
    auth.py              # daily Kite Connect login (TOTP-based)
    data_fetch.py         # historical candle fetcher
    indicators.py         # EMA, RSI, Fibonacci levels, candlestick patterns
    signal_engine.py       # confluence rule -> daily CALL/PUT/NO_TRADE decision
    options_pricing.py     # Black-Scholes premium simulation (no real historical option data available)
    backtester.py           # day-by-day replay + bracket simulation + reporting
    option_lookup.py         # finds real option contracts + real quoted premiums (Phase 4+)
    paper_trader.py          # live paper trading loop (Phase 4) - simulated orders, real quotes
    dashboard.py             # local web control panel + reports (Streamlit)
    ml_features.py            # feature engineering for the ML signal engine (Phase 6)
    ml_signal.py               # labeling, training (3 model types), generate_ml_signal (Phase 6)
    capital_manager.py          # paper-trading capital tracking + lot-size position sizing
  .streamlit/config.toml     # locks the dashboard to localhost only - do not remove
  tests/
    test_indicators.py                # automated unit tests (pytest)
    test_options_pricing.py            # automated unit tests (pytest), incl. put-call parity check
    test_option_lookup.py               # automated unit tests (pytest)
    test_paper_trader.py                # automated unit tests (pytest) for exit-decision logic
    test_ml_features.py                  # automated unit tests (pytest)
    test_capital_manager.py               # automated unit tests (pytest)
    manual_signal_sanity_check.py     # manual eyeball check against real data, not automated
    diagnose_signal_funnel.py          # one-off diagnostic: which confluence condition is the bottleneck
    compare_ml_vs_rules.py              # honest side-by-side backtest comparison, not automated
  data/
    historical/            # cached candle CSVs
    backtest_results/       # trades.csv from the last backtester.py run, plus *_test_period.csv comparisons
    paper_trades/           # paper_trades.csv (real trade log incl. lots/rupee P&L), capital_state.json (running balance)
    models/                 # trained ML models per type: call_model_<model_type>.joblib, put_model_<model_type>.joblib,
                             #   thresholds_<model_type>.joblib - gitignored, derived artifacts (model_type = random_forest /
                             #   logistic_regression / gradient_boosting)
```
