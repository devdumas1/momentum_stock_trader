"""Trend rules ("signals") the backtest and live trader can run.

A signal is a function that receives a clean price window (rows = trading days,
columns = tickers, no gaps, last row = the rebalance date) and returns a
DataFrame indexed by ticker with a "score" column: higher = stronger trend.
By convention score > 0 means "in an uptrend", which the optional trend filter
uses to decide between holding a stock and holding cash. Any extra columns are
kept in holdings.csv for reference.

To add your own rule, write a function in a file and register it:

    from signals import signal

    @signal("my-rule", history=lambda p: p["days"], trend_filter=True)
    def my_rule(window, days=100):
        \"\"\"One-line description shown by --list-signals.\"\"\"
        return pd.DataFrame({"score": window.iloc[-1] / window.iloc[-days] - 1})

then run:  python backtest.py --signal-file my_rules.py --signal my-rule --param days=60
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 21


@dataclass(frozen=True)
class Signal:
    name: str
    fn: Callable[..., pd.DataFrame]
    history: Callable[[dict], int]  # trading days of history needed, given resolved params
    trend_filter: bool              # default for "only hold stocks with score > 0"
    description: str


SIGNALS: dict[str, Signal] = {}


def signal(name: str, history: Callable[[dict], int], trend_filter: bool = False):
    """Decorator that registers a signal function under `name`."""
    def register(fn):
        doc = (fn.__doc__ or "").strip().splitlines()
        SIGNALS[name] = Signal(name, fn, history, trend_filter, doc[0] if doc else "")
        return fn
    return register


@signal("momentum", history=lambda p: p["lookback_months"] * TRADING_DAYS_PER_MONTH)
def momentum(window: pd.DataFrame, lookback_months: int = 12, skip_months: int = 1) -> pd.DataFrame:
    """Return over the lookback (skipping the latest months) divided by volatility."""
    w = window.iloc[: len(window) - skip_months * TRADING_DAYS_PER_MONTH]
    n_days = len(w) - 1
    total_ret = w.iloc[-1] / w.iloc[0] - 1
    ann_ret = (1 + total_ret) ** (TRADING_DAYS_PER_YEAR / n_days) - 1
    ann_vol = w.pct_change().iloc[1:].std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    return pd.DataFrame({
        "return": total_ret,
        "volatility": ann_vol,
        "score": ann_ret / ann_vol.replace(0, np.nan),
    })


@signal("ma-cross", history=lambda p: p["slow"], trend_filter=True)
def ma_cross(window: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """Fast vs. slow moving average: positive while the fast MA is above the slow one."""
    fast_ma = window.iloc[-fast:].mean()
    slow_ma = window.iloc[-slow:].mean()
    return pd.DataFrame({"fast_ma": fast_ma, "slow_ma": slow_ma, "score": fast_ma / slow_ma - 1})


@signal("price-vs-ma", history=lambda p: p["days"], trend_filter=True)
def price_vs_ma(window: pd.DataFrame, days: int = 200) -> pd.DataFrame:
    """Price vs. its N-day moving average: positive while price is above the average."""
    ma = window.iloc[-days:].mean()
    return pd.DataFrame({"ma": ma, "score": window.iloc[-1] / ma - 1})


@signal("breakout", history=lambda p: p["days"], trend_filter=True)
def breakout(window: pd.DataFrame, days: int = 252) -> pd.DataFrame:
    """Position in the N-day high/low range: +1 at the high, -1 at the low."""
    w = window.iloc[-(days + 1) :]
    high, low = w.max(), w.min()
    position = (w.iloc[-1] - low) / (high - low).replace(0, np.nan)
    return pd.DataFrame({"high": high, "low": low, "score": 2 * position - 1})
