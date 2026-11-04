"""Runs any registered trend signal (see signals.py) and picks the portfolio.

The engine is signal-agnostic: each rebalance it hands the chosen signal a clean
price window, ranks stocks by the returned score, optionally drops stocks that
aren't in an uptrend (score <= 0, the rest stays in cash), and holds the top N
equal-weighted.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from signals import SIGNALS, TRADING_DAYS_PER_YEAR, Signal  # noqa: F401  (re-exported)


@dataclass
class StrategyConfig:
    signal: str = "momentum"
    params: dict = field(default_factory=dict)  # overrides for the signal's defaults
    top_n: int = 10
    trend_filter: bool | None = None  # None -> the signal's own default
    max_missing_frac: float = 0.05    # drop stocks missing more data than this in the window

    def __post_init__(self) -> None:
        if self.signal not in SIGNALS:
            raise ValueError(f"Unknown signal {self.signal!r}. Available: {', '.join(SIGNALS)}")
        defaults = signal_defaults(self.sig)
        unknown = set(self.params) - set(defaults)
        if unknown:
            raise ValueError(f"Signal {self.signal!r} has no parameter(s) {sorted(unknown)}; "
                             f"it takes {sorted(defaults)}")
        self.params = {**defaults, **self.params}

    @property
    def sig(self) -> Signal:
        return SIGNALS[self.signal]

    @property
    def history_days(self) -> int:
        return int(self.sig.history(self.params))

    @property
    def use_trend_filter(self) -> bool:
        return self.sig.trend_filter if self.trend_filter is None else self.trend_filter

    def describe(self) -> str:
        params = " ".join(f"{k}={v}" for k, v in self.params.items())
        return (f"signal={self.signal} ({params})  top_n={self.top_n}  "
                f"trend_filter={'on' if self.use_trend_filter else 'off'}")


def signal_defaults(sig: Signal) -> dict:
    params = list(inspect.signature(sig.fn).parameters.values())[1:]  # skip `window`
    return {p.name: p.default for p in params}


def score_stocks(prices: pd.DataFrame, as_of: pd.Timestamp, cfg: StrategyConfig) -> pd.DataFrame:
    """Signal output for every eligible stock using prices up to `as_of`, best first."""
    hist = prices.loc[:as_of]
    need = cfg.history_days
    if len(hist) < need + 1:
        return pd.DataFrame(columns=["score"])

    window = hist.iloc[-(need + 1) :]
    # Eligible: priced at both ends of the window and mostly complete in between.
    complete = window.notna().mean() >= 1 - cfg.max_missing_frac
    eligible = complete & window.iloc[0].notna() & window.iloc[-1].notna()
    window = window.loc[:, eligible].ffill()
    if window.shape[1] == 0:
        return pd.DataFrame(columns=["score"])

    out = cfg.sig.fn(window, **cfg.params).replace([np.inf, -np.inf], np.nan)
    return out.dropna(subset=["score"]).sort_values("score", ascending=False)


def select_top(scores: pd.DataFrame, cfg: StrategyConfig) -> list[str]:
    """Pick the top-N tickers by score (only uptrending ones if the trend filter is on)."""
    picks = scores[scores["score"] > 0] if cfg.use_trend_filter else scores
    return picks.head(cfg.top_n).index.tolist()


def history_start(date, cfg: StrategyConfig) -> str:
    """Calendar date early enough to have `cfg.history_days` trading days before `date`."""
    days = int(cfg.history_days * 365 / TRADING_DAYS_PER_YEAR) + 45
    return (pd.Timestamp(date) - pd.Timedelta(days=days)).strftime("%Y-%m-%d")


def load_signal_file(path: str) -> None:
    """Import a user file so its @signal-decorated functions get registered."""
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load signal file {path}")
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


# --- shared command-line options for backtest.py and live_trade.py ---------------

def add_strategy_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("strategy")
    g.add_argument("--signal", default="momentum", help="trend rule to run (see --list-signals)")
    g.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                   help="signal parameter, repeatable (e.g. --param fast=20 --param slow=100)")
    g.add_argument("--signal-file", action="append", default=[],
                   help="python file defining extra @signal rules")
    g.add_argument("--list-signals", action="store_true", help="list available signals and exit")
    g.add_argument("--top-n", type=int, default=10)
    g.add_argument("--trend-filter", action=argparse.BooleanOptionalAction, default=None,
                   help="hold cash instead of stocks with score <= 0 (default: per signal)")


def _parse_value(text: str):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def config_from_args(args: argparse.Namespace) -> StrategyConfig:
    for path in args.signal_file:
        load_signal_file(path)
    if args.list_signals:
        for sig in SIGNALS.values():
            params = " ".join(f"{k}={v}" for k, v in signal_defaults(sig).items())
            filt = "trend filter on" if sig.trend_filter else "trend filter off"
            print(f"{sig.name:12} {sig.description}\n{'':12} params: {params}  ({filt} by default)")
        raise SystemExit(0)
    params = {}
    for item in args.param:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--param expects KEY=VALUE, got {item!r}")
        params[key.strip()] = _parse_value(value.strip())
    try:
        return StrategyConfig(signal=args.signal, params=params, top_n=args.top_n,
                              trend_filter=args.trend_filter)
    except ValueError as exc:
        raise SystemExit(str(exc))
