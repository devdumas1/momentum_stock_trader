# Momentum Trading Algorithm

Ranks stocks by **risk-adjusted momentum** (annualized 12-1 month return ÷ annualized
volatility), holds the top N equal-weighted, and rebalances monthly.

| File | Purpose |
|---|---|
| `data.py` | S&P 500 universe (Wikipedia) + cached yfinance price downloads |
| `strategy.py` | `MomentumConfig`, `momentum_scores`, `select_top` |
| `backtest.py` | Monthly-rebalance simulation, transaction costs, metrics vs SPY |
| `live_trade.py` | Rebalances an Alpaca account to the current top N (dry run by default) |

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps alpaca-trade-api==3.2.0
pip install websocket-client msgpack aiohttp PyYAML deprecation
```

## Backtest
```bash
python backtest.py --start 2015-01-01 --top-n 10 --cost-bps 10
python backtest.py --universe AAPL,MSFT,NVDA,AMZN,META --top-n 2
python backtest.py --require-positive   # sit in cash rather than hold negative-momentum stocks
```
Outputs go to `results/`: `equity_curve.csv`, `holdings.csv`, `turnover.csv`, `summary.txt`
(add `--plot` with matplotlib installed for a chart).

## Paper / live trading (Alpaca)
```bash
export APCA_API_KEY_ID=... APCA_API_SECRET_KEY=...
python live_trade.py                    # dry run: prints planned orders
python live_trade.py --execute          # submits to the PAPER account
python live_trade.py --execute --live   # real money; requires the explicit flag
```
Run monthly, e.g. cron on the 1st at 10:00 ET:
`0 10 1 * * cd /path/to/momentum-trader && .venv/bin/python live_trade.py --execute`

## Caveats
- **Survivorship bias**: the universe is *today's* S&P 500, so the backtest only picks
  from companies that survived and grew into the index. Real returns would be much lower.
- Trades are filled at the same close used for scoring; no slippage beyond `--cost-bps`.
- `alpaca-trade-api` is deprecated (successor: `alpaca-py`); its REST client still works.
- Not investment advice.
