"""Momentum scoring and stock selection.

Score = annualized return / annualized volatility, measured over a lookback
window that skips the most recent month (the classic "12-1" momentum, which
avoids the short-term reversal effect). It is a risk-adjusted momentum: a
stock that rose 40% smoothly ranks above one that rose 40% on wild swings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 21


@dataclass
class MomentumConfig:
    lookback_months: int = 12      # total window used to measure momentum
    skip_months: int = 1           # most recent months excluded from the window
    top_n: int = 10                # number of stocks held
    max_missing_frac: float = 0.05  # drop stocks missing more data than this in the window
    require_positive: bool = False  # if True, only hold stocks with positive return (rest in cash)

    @property
    def lookback_days(self) -> int:
        return self.lookback_months * TRADING_DAYS_PER_MONTH

    @property
    def skip_days(self) -> int:
        return self.skip_months * TRADING_DAYS_PER_MONTH


def momentum_scores(prices: pd.DataFrame, as_of: pd.Timestamp, cfg: MomentumConfig) -> pd.DataFrame:
    """Return a DataFrame (return, volatility, score) for every eligible stock,
    using only prices up to and including `as_of`, sorted best first."""
    hist = prices.loc[:as_of]
    if len(hist) < cfg.lookback_days + 1:
        return pd.DataFrame(columns=["return", "volatility", "score"])

    window = hist.iloc[-(cfg.lookback_days + 1) : len(hist) - cfg.skip_days]
    # Eligibility: priced on both ends and mostly complete over the window.
    complete = window.notna().mean() >= 1 - cfg.max_missing_frac
    eligible = complete & window.iloc[0].notna() & window.iloc[-1].notna()
    window = window.loc[:, eligible].ffill()

    n_days = len(window) - 1
    total_ret = window.iloc[-1] / window.iloc[0] - 1
    ann_ret = (1 + total_ret) ** (TRADING_DAYS_PER_YEAR / n_days) - 1
    ann_vol = window.pct_change().iloc[1:].std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    out = pd.DataFrame({"return": total_ret, "volatility": ann_vol})
    out["score"] = ann_ret / ann_vol.replace(0, np.nan)
    return out.dropna().sort_values("score", ascending=False)


def select_top(scores: pd.DataFrame, cfg: MomentumConfig) -> list[str]:
    """Pick the top-N tickers by score."""
    picks = scores
    if cfg.require_positive:
        picks = picks[picks["return"] > 0]
    return picks.head(cfg.top_n).index.tolist()
