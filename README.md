# Trend Trading Toolkit

Backtest and trade **any trend rule** on the S&P 500. Each month the chosen rule scores every
stock, the top N are held equal-weighted, and the portfolio is rebalanced. Rules are pluggable:
four are built in, and you can add your own in a few lines.

| File | Purpose |
|---|---|
| `signals.py` | Trend rules (momentum, ma-cross, price-vs-ma, breakout) and the `@signal` registry |
| `strategy.py` | Runs any signal: eligibility, ranking, trend filter, shared CLI options |
| `data.py` | Point-in-time S&P 500 membership, current list (Wikipedia), cached yfinance prices |
| `backtest.py` | Monthly-rebalance simulation, transaction costs, metrics vs SPY |
| `live_trade.py` | Rebalances an Alpaca account to the current picks (dry run by default) |
| `examples/my_rules.py` | Example custom rule (`dual-momentum`) |

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps alpaca-trade-api==3.2.0
pip install websocket-client msgpack aiohttp PyYAML deprecation
```

## Trend rules
```bash
python backtest.py --list-signals
```
| Signal | Score (higher = stronger trend; > 0 = uptrend) | Params |
|---|---|---|
| `momentum` | 12-month return (skipping last month) ÷ volatility | `lookback_months=12 skip_months=1` |
| `ma-cross` | fast moving average ÷ slow moving average − 1 | `fast=50 slow=200` |
| `price-vs-ma` | price ÷ N-day moving average − 1 | `days=200` |
| `breakout` | position in N-day high/low range (−1 low … +1 high) | `days=252` |

**Trend filter** (`--trend-filter` / `--no-trend-filter`): when on, stocks with score ≤ 0 are
never bought and their slots stay in cash. On by default for the trend rules, off for momentum.

### Adding your own rule
A rule gets a gap-free price window (rows = days, columns = tickers, last row = today) and
returns a DataFrame with a `score` column:
```python
# my_rules.py
import pandas as pd
from signals import signal

@signal("roc", history=lambda p: p["days"], trend_filter=True)
def rate_of_change(window, days=126):
    """N-day price change."""
    return pd.DataFrame({"score": window.iloc[-1] / window.iloc[-days - 1] - 1})
```
```bash
python backtest.py --signal-file my_rules.py --signal roc --param days=63
```
`history` tells the engine how many trading days of prices the rule needs.

## Backtest
```bash
python backtest.py                                          # momentum, point-in-time S&P 500
python backtest.py --signal ma-cross --param fast=20 --param slow=100 --top-n 20
python backtest.py --universe XLK,XLE,XLV,XLF,XLI,XLY,XLP,XLU,XLB --signal price-vs-ma --top-n 3
python backtest.py --universe sp500     # today's members only (survivorship-biased, for comparison)
```
Outputs go to `--out` (default `results/`): `equity_curve.csv`, `holdings.csv`, `turnover.csv`,
`coverage.csv`, `summary.txt` (add `--plot` with matplotlib installed for a chart).

Built-in rules, point-in-time S&P 500, top 10, 10 bps costs, Jan 2015 – Nov 2026:

| Signal | CAGR | Sharpe | Max drawdown | Monthly turnover |
|---|---|---|---|---|
| momentum | 13.9% | 0.60 | −43.3% | 88% |
| ma-cross | 23.8% | 0.84 | −37.8% | 71% |
| price-vs-ma | 16.3% | 0.66 | −39.2% | 94% |
| breakout | 5.7% | 0.40 | −25.4% | 185% |
| dual-momentum (example) | 16.0% | 0.64 | −44.4% | 92% |
| SPY | 13.8% | 0.83 | −33.7% | — |

Picking the best of several rules on the same period overstates how well it will do next;
treat these as a starting point for out-of-sample testing, not a forecast.

## Paper / live trading (Alpaca)
```bash
export APCA_API_KEY_ID=... APCA_API_SECRET_KEY=...
python live_trade.py                          # dry run: prints planned orders
python live_trade.py --signal ma-cross        # any signal, same options as backtest.py
python live_trade.py --execute                # submits to the PAPER account
python live_trade.py --execute --live         # real money; requires the explicit flag
```
Run monthly, e.g. cron on the 1st at 10:00 (machine's local time):
`0 10 1 * * cd /path/to/momentum-trader && .venv/bin/python live_trade.py --execute`
(cron doesn't see keys exported in your shell — set them in the crontab line.)

## Caveats
- **Survivorship bias**: by default each rebalance only considers stocks that were S&P 500
  members *on that date* (history from [fja05680/sp500](https://github.com/fja05680/sp500),
  topped up with today's Wikipedia list). Renamed tickers are mapped in `TICKER_RENAMES`.
  Still unfixed: ~11% of past members (mostly acquired/bankrupt) have no data on Yahoo, so
  they can never be picked — `coverage.csv` shows how many per month. Using today's list
  instead inflated momentum's CAGR from 13.9% to 31.7%.
- On a 500-stock universe at least 10 stocks are nearly always in an uptrend, so the
  per-stock trend filter rarely moves the portfolio to cash. A market-level filter would.
- Trades are filled at the same close used for scoring; no slippage beyond `--cost-bps`.
- `breakout` scores many stocks at exactly +1 (at their high); ties make it churn.
- `alpaca-trade-api` is deprecated (successor: `alpaca-py`); its REST client still works.
- Not investment advice.
