# Share Market Bro

> **See `PROJECT_STATUS.md` for the full project state** — goal, trading rules, current phase, broker/account setup, and technical gotchas. Read that first, especially after a break or when picking this up with a different AI assistant.

Autonomous NIFTY 50 options trading bot, built in phases:

0. Project setup & auth — **done**
1. Historical data pipeline — **done**
2. Rule-based signal engine (EMA trend + Fibonacci + RSI + candlestick confluence) — **done**
3. Backtesting engine — **done** (see `PROJECT_STATUS.md` for the current result)
4. Paper trading engine — **done, running live during real market hours across multiple real trading days now (see `PROJECT_STATUS.md` for results)**
5. Live trading — not started
6. ML enhancement — **done, built and honestly compared against the rule-based engine (see `PROJECT_STATUS.md`); Gradient Boosting (XGBoost) is now the live paper-trading default**

## Setup

1. Install dependencies:
   ```
   py -m pip install -r requirements.txt
   ```

2. Get your Zerodha Kite Connect credentials:
   - Subscribe to the Kite Connect API plan at https://developers.kite.trade
   - Create an app there to get an **API key** and **API secret**
   - Make sure TOTP-based 2FA is enabled on your Zerodha account (Kite app/web > Settings > Account > 2FA), and save the TOTP secret shown during setup

3. Copy `.env.example` to `.env` and fill in your real values:
   ```
   copy .env.example .env
   ```

## Usage

**Log in and check it works:**
```
py src/auth.py
```
If this fails with a CAPTCHA error, Zerodha has put a CAPTCHA on the login page for your account (this can happen after a burst of automated logins) - the automated flow in `auth.py` can't solve it. Run this instead, once, and follow the prompts (you'll log in through a real browser yourself):
```
py src/manual_login.py
```
It opens your browser to the login page automatically. Log in (solving the CAPTCHA yourself), then copy the URL you land on afterward and paste it back into the terminal when asked - the whole thing takes under a minute. It caches a fresh token that `auth.py`/`paper_trader.py` will pick up automatically afterward - no other changes needed, and any already-running bot process recovers on its own within its normal retry cycle, no restart required.

There's also a `--auto` mode (`py src/manual_login.py --auto`) that tries to capture the redirect itself with no copy-paste at all - it's not the default because it's confirmed unreliable on at least one machine so far (likely a Windows Firewall rule blocking the browser from reaching the local capture server). Worth trying again after checking Windows Security -> Firewall & network protection -> Allow an app through firewall for `python.exe`.

**Fetch historical data** (cached to `data/historical/` as CSV, so repeated runs don't re-hit the API):
```
py src/data_fetch.py --symbol "NIFTY 50" --interval day --days 150
py src/data_fetch.py --symbol "NIFTY 50" --interval 15minute --days 200
```

**Run the backtester** (replays the signal engine over cached historical data, simulating premiums via Black-Scholes since real historical option data isn't available — see `PROJECT_STATUS.md`):
```
py src/backtester.py
```
Prints a summary and writes every simulated trade to `data/backtest_results/trades.csv`.

**Run the paper trader** (live, during real NSE market hours 9:15am–3:30pm IST — uses real quoted option premiums, but never places a real order):
```
py src/paper_trader.py
```
Defaults to the **Gradient Boosting (XGBoost)** signal — the strongest candidate from the Phase 6 comparison. Add `--signal-source rule_based` (or `random_forest` / `logistic_regression`) to use a different one. Add `--max-minutes N` to stop after N minutes (useful for a quick smoke test rather than running the whole day). Trades are logged to `data/paper_trades/paper_trades.csv`, including real lot counts and rupee P&L (starting capital ₹20,000, tracked in `data/paper_trades/capital_state.json` and compounded automatically - see `src/capital_manager.py`).

Add `--max-trades-per-day N` (default **1**, the intended discipline) to allow more than one trade slot in a single day - useful for quickly gathering a validation sample (e.g. `--max-trades-per-day 20`) rather than waiting many calendar days for one trade each. Switch back to the default of 1 once you've gathered enough data. **This is also available directly on the dashboard** (a "Max trades per day" box next to the Start button) - no need to use the command line for this.

Add `--max-capital-per-trade N` (default **₹2,00,000**) to cap how much of the balance is ever risked on a single trade, no matter how large the account has compounded to - anything above the cap simply stays idle/untouched. **Also available directly on the dashboard** (a "Max capital per trade (Rs)" box next to the Start button).

**Go-live safety net (added 2026-08-12, first piece of the agreed go-live plan, already active in paper mode):**
- **Kill switch** - an emergency stop that halts new entries and force-closes any open position within one poll interval (up to ~15s), independent of the dashboard or which terminal the bot was launched from. Manage it with `py src/kill_switch.py on` / `off` / `status`, or the dashboard's Kill Switch button.
- **Daily loss circuit breaker** - stops taking new trades for the rest of the day once today's realized loss reaches `--max-daily-loss-pct` (default **10%**, i.e. 2x the single-trade stop-loss) of the capital the day started with. Any position already open when it trips still runs its normal exit. Survives a bot restart mid-day - it re-sums today's already-logged trades from `paper_trades.csv` on startup rather than resetting to zero.

**PUT-only by default** - CALL's confidence collapses at high thresholds in both the primary and early-session models (confirmed 2026-07-24), a consistent cross-model weakness. Add `--allow-calls` to take CALL signals too. **Also available on the dashboard** ("Allow CALL trades" checkbox, unchecked by default).

Add `--split-session` to use two independent trade quotas instead of one flat daily cap: up to `--max-trades-per-session N` (default **6**) trades before 1:15 PM (morning, early-session model), then up to N more from 1:15 PM onward (afternoon, primary model) - up to 2N trades/day total. If the morning quota fills before 1:15, new entries pause (not end the day) until the afternoon quota opens. Overrides `--max-trades-per-day` when set. **Also available on the dashboard** ("Split into morning/afternoon sessions" checkbox).

By default, the bot can't generate any signal for the first ~4 hours after market open (RSI needs 16 candles' worth of history) - but a second model trained specifically on 5-minute candles (`gradient_boosting` signal source only) kicks in automatically until then, letting it signal from as early as ~1h20m after open instead. No flag needed - this is automatic once the 5-min model is trained (see below).

Every real trade also gets a **Max Pain / Open Interest check logged alongside it, in shadow mode** - it does not affect trading decisions yet, purely diagnostic (see `PROJECT_STATUS.md` for the full plan). Computes the strike that would minimize total payout to option holders for that trade's expiry (`src/option_lookup.py`'s `compute_max_pain()`, using real Open Interest via `kite.quote()`), and logs whether it agrees with the signal's direction as two extra columns in `paper_trades.csv`: `max_pain_strike`, `max_pain_agreed`.

**Train the ML signal engine** (labels historical candles, trains all 3 model types - Random Forest, Logistic Regression, Gradient Boosting - with calibrated thresholds, saves to `data/models/`):
```
py src/ml_signal.py
```

**Train the early-session (5-minute candle) model** (lets `gradient_boosting` signal earlier in the day - see above; needs `data/historical/NIFTY_50_5minute.csv`, fetch it first via `py src/data_fetch.py --interval 5minute --days 210`):
```
py src/train_5min_model.py
```

**Compare the ML signal against the rule-based one** (honest side-by-side on the same held-out test period):
```
py tests/compare_ml_vs_rules.py
```

**Run the test suite:**
```
py -m pytest tests/ -v
```

**Open the dashboard** (a local webpage with a Start/Stop control for the bot, plus backtest, paper-trading, and daily-summary reports — trade log, win rate, P&L, equity curve):
```
py -m streamlit run src/dashboard.py
```
Opens at `http://localhost:8501` (locked to your machine only, per `.streamlit/config.toml`). From here you can click **Start** to launch the paper trader in the background instead of running it from a terminal, watch its live log, and click **Stop** to end it early. Note: if you close the dashboard while the bot is running, the dashboard loses track of it (it keeps running in the background) — close it manually via Task Manager in that case.

Three tabs: **Backtest Results**, **Paper Trading Results**, and **Daily Summary** (a per-day win/loss/P&L breakdown, including real Zerodha brokerage + Kite subscription costs netted out - not just the gross figure). The Daily Summary tab has a **Period filter** (All time / This month / Custom range) so you can pull up a report for just a specific window instead of always seeing every day.

Each row in the trade log has a **"View Candle" button** — click it to see a 2-hour, 5-minute-candle chart of that trade's actual option premium (1 hour before/after entry), with entry and exit marked. Useful for eyeballing whether a stop-loss would have recovered, or a target exit left more upside on the table. Charts are generated automatically at trade-close time (`src/trade_chart.py`) and saved to `data/paper_trades/charts/` — this must happen soon after each trade, since Kite Connect permanently loses historical data once an option contract expires.

Every table also has a **"Download as PDF" button** next to Streamlit's built-in "Download as CSV" one, for a printable version of the same report.

## Notes

- Kite access tokens expire daily. `auth.py` automates the login (password + TOTP) since there's no official headless login API - this uses the same login steps Zerodha's own web login uses, automated via `requests`. It's a widely-used community pattern, not an officially documented API, so if it starts failing, Zerodha's login internals may have changed.
- Never commit `.env` or anything under `.cache/` - both are gitignored.
