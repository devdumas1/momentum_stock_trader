"""Rebalance an Alpaca account to the current top-N momentum portfolio.

Run this once a month (e.g. the first trading day). By default it is a DRY RUN
against the PAPER endpoint: it prints the orders it would send and sends nothing.

    export APCA_API_KEY_ID=...        # from https://app.alpaca.markets
    export APCA_API_SECRET_KEY=...
    python live_trade.py                      # dry run, paper account
    python live_trade.py --execute            # send orders to paper account
    python live_trade.py --execute --live     # REAL MONEY (live endpoint)
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import pandas as pd
from alpaca_trade_api.rest import REST, APIError

from data import download_prices, load_universe
from strategy import MomentumConfig, momentum_scores, select_top

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"


def to_alpaca(symbol: str) -> str:
    return symbol.replace("-", ".")  # BRK-B (Yahoo) -> BRK.B (Alpaca)


def to_yahoo(symbol: str) -> str:
    return symbol.replace(".", "-")


def target_portfolio(universe: list[str], cfg: MomentumConfig) -> tuple[list[str], pd.DataFrame, pd.Series]:
    start = (pd.Timestamp.today() - pd.DateOffset(months=cfg.lookback_months + 2)).strftime("%Y-%m-%d")
    prices = download_prices(universe, start, use_cache=False)
    as_of = prices.index[-1]
    scores = momentum_scores(prices, as_of, cfg)
    picks = select_top(scores, cfg)
    print(f"Scores as of close {as_of.date()} ({len(scores)} eligible stocks)")
    return picks, scores, prices.iloc[-1]


def main() -> None:
    p = argparse.ArgumentParser(description="Alpaca momentum rebalancer")
    p.add_argument("--universe", default="sp500")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--lookback", type=int, default=12)
    p.add_argument("--skip", type=int, default=1)
    p.add_argument("--require-positive", action="store_true")
    p.add_argument("--cash-buffer", type=float, default=0.02,
                   help="fraction of equity left uninvested (covers slippage)")
    p.add_argument("--min-trade", type=float, default=50.0,
                   help="skip adjustments smaller than this many dollars")
    p.add_argument("--fractional", action="store_true",
                   help="use notional (fractional-share) orders instead of whole shares")
    p.add_argument("--execute", action="store_true", help="actually submit orders")
    p.add_argument("--live", action="store_true", help="use the LIVE (real money) endpoint")
    args = p.parse_args()

    key, secret = os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        sys.exit("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables.")
    base_url = LIVE_URL if args.live else os.getenv("APCA_API_BASE_URL", PAPER_URL)
    if "paper" not in base_url and not args.live:
        sys.exit(f"APCA_API_BASE_URL points at {base_url}; pass --live to confirm real-money trading.")

    api = REST(key, secret, base_url, api_version="v2")
    account = api.get_account()
    clock = api.get_clock()
    mode = "LIVE" if "paper" not in base_url else "PAPER"
    print(f"[{mode}] account {account.account_number}  equity ${float(account.equity):,.2f}  "
          f"market {'OPEN' if clock.is_open else 'CLOSED (orders will queue for next open)'}")

    cfg = MomentumConfig(lookback_months=args.lookback, skip_months=args.skip,
                         top_n=args.top_n, require_positive=args.require_positive)
    picks, scores, last_px = target_portfolio(load_universe(args.universe), cfg)

    # Drop anything Alpaca can't trade.
    tradable = []
    for tk in picks:
        try:
            asset = api.get_asset(to_alpaca(tk))
            if asset.tradable:
                tradable.append(tk)
            else:
                print(f"  skip {tk}: not tradable on Alpaca")
        except APIError:
            print(f"  skip {tk}: unknown to Alpaca")
    picks = tradable

    print("\nTarget portfolio:")
    print(scores.loc[picks, ["return", "volatility", "score"]].round(3).to_string())

    equity = float(account.equity)
    per_stock = equity * (1 - args.cash_buffer) / cfg.top_n
    positions = {to_yahoo(pos.symbol): pos for pos in api.list_positions()}
    target_values = {tk: per_stock for tk in picks}

    orders = []  # (symbol, side, qty or None, notional or None)
    for tk, pos in positions.items():  # exit names that dropped out
        if tk not in target_values:
            orders.append((tk, "sell", abs(float(pos.qty)), None))

    for tk, tgt in target_values.items():
        current = float(positions[tk].market_value) if tk in positions else 0.0
        diff = tgt - current
        if abs(diff) < args.min_trade:
            continue
        side = "buy" if diff > 0 else "sell"
        if args.fractional:
            orders.append((tk, side, None, round(abs(diff), 2)))
        else:
            qty = math.floor(abs(diff) / float(last_px[tk]))
            if qty > 0:
                orders.append((tk, side, qty, None))

    orders.sort(key=lambda o: o[1] != "sell")  # sells first to free up cash
    print(f"\nPlanned orders ({len(orders)}):")
    for sym, side, qty, notional in orders:
        size = f"{qty:g} sh" if qty is not None else f"${notional:,.2f}"
        print(f"  {side.upper():4} {sym:6} {size}")

    if not args.execute:
        print("\nDry run — no orders sent. Re-run with --execute to submit.")
        return

    for sym, side, qty, notional in orders:
        try:
            if qty is not None and side == "sell" and sym not in target_values:
                api.close_position(to_alpaca(sym))
            else:
                api.submit_order(symbol=to_alpaca(sym), side=side, type="market",
                                 time_in_force="day", qty=qty, notional=notional)
            print(f"  sent {side} {sym}")
        except APIError as exc:
            print(f"  FAILED {side} {sym}: {exc}")


if __name__ == "__main__":
    main()
