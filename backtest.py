"""Monthly-rebalanced momentum backtest.

Usage:
    python backtest.py --start 2015-01-01 --top-n 10
    python backtest.py --universe fallback --start 2018-01-01 --plot
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from data import download_prices, load_universe
from strategy import TRADING_DAYS_PER_YEAR, MomentumConfig, momentum_scores, select_top


@dataclass
class BacktestResult:
    equity: pd.Series
    holdings: pd.DataFrame
    turnover: pd.Series
    benchmark: pd.Series | None = None
    config: MomentumConfig = field(default_factory=MomentumConfig)

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().dropna()


def month_end_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Last trading day of each month present in `index` (excluding the final
    data point, since there's no holding period after it)."""
    ends = index.to_series().groupby(index.to_period("M")).max()
    return [d for d in ends if d != index[-1]]


def run_backtest(
    prices: pd.DataFrame,
    cfg: MomentumConfig,
    start: str | None = None,
    cost_bps: float = 10.0,
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    """Simulate an equal-weight top-N momentum portfolio rebalanced at each month end.

    At each rebalance date the scores use closes up to that date, the portfolio
    trades at that close, then holds (weights drift with prices) until the next
    month end. Transaction cost = turnover * cost_bps.
    """
    rebal = month_end_dates(prices.index)
    first_ok = prices.index[min(cfg.lookback_days, len(prices.index) - 1)]
    rebal = [d for d in rebal if d >= first_ok and (start is None or d >= pd.Timestamp(start))]
    if not rebal:
        raise ValueError("Not enough history for the requested lookback/start date.")

    equity = pd.Series(np.nan, index=prices.loc[rebal[0] :].index, dtype=float)
    equity.iloc[0] = initial_capital
    value = initial_capital
    drifted = pd.Series(dtype=float)  # weights just before the next rebalance
    holding_rows, turnover = [], {}

    for i, t in enumerate(tqdm(rebal, desc="Rebalancing", unit="month")):
        t_next = rebal[i + 1] if i + 1 < len(rebal) else prices.index[-1]
        scores = momentum_scores(prices, t, cfg)
        picks = select_top(scores, cfg)
        target = pd.Series(1.0 / cfg.top_n, index=picks, dtype=float)

        trade = target.sub(drifted, fill_value=0.0).abs().sum()
        turnover[t] = trade
        value -= value * trade * cost_bps / 1e4

        for rank, tk in enumerate(picks, 1):
            row = scores.loc[tk]
            holding_rows.append(
                {"date": t, "rank": rank, "ticker": tk, "score": row["score"],
                 "lookback_return": row["return"], "volatility": row["volatility"]}
            )

        period = prices.loc[t:t_next]
        cash_w = 1.0 - target.sum()
        if picks:
            px = period[picks].ffill()
            rel = (px / px.iloc[0]).fillna(1.0)
            growth = rel.mul(target, axis=1).sum(axis=1) + cash_w
            drifted = rel.iloc[-1] * target / growth.iloc[-1]
        else:
            growth = pd.Series(1.0, index=period.index)
            drifted = pd.Series(dtype=float)

        path = value * growth
        equity.loc[path.index[1:]] = path.iloc[1:].values
        value = path.iloc[-1]

    return BacktestResult(
        equity=equity.dropna(),
        holdings=pd.DataFrame(holding_rows),
        turnover=pd.Series(turnover, name="turnover"),
        config=cfg,
    )


def performance_metrics(equity: pd.Series, rf: float = 0.0) -> dict[str, float]:
    rets = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    vol = rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    excess = rets - rf / TRADING_DAYS_PER_YEAR
    sharpe = excess.mean() / rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if rets.std() else np.nan
    downside = rets[rets < 0].std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sortino = excess.mean() * TRADING_DAYS_PER_YEAR / downside if downside else np.nan
    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()
    monthly = equity.resample("ME").last().pct_change().dropna()
    return {
        "Total Return": total,
        "CAGR": cagr,
        "Annual Volatility": vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_dd,
        "Calmar Ratio": cagr / abs(max_dd) if max_dd else np.nan,
        "Positive Months": (monthly > 0).mean(),
        "Best Month": monthly.max(),
        "Worst Month": monthly.min(),
    }


def format_metrics(table: pd.DataFrame) -> str:
    ratio_rows = {"Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"}
    out = table.copy().astype(object)
    for idx in out.index:
        fmt = "{:.2f}" if idx in ratio_rows else "{:.2%}"
        out.loc[idx] = [fmt.format(v) if pd.notna(v) else "n/a" for v in table.loc[idx]]
    return out.to_string()


def main() -> None:
    p = argparse.ArgumentParser(description="Momentum strategy backtest")
    p.add_argument("--universe", default="sp500",
                   help="'sp500', 'fallback', a ticker file, or comma-separated tickers")
    p.add_argument("--start", default="2015-01-01", help="first rebalance date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="last date of data (default: today)")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--lookback", type=int, default=12, help="lookback months")
    p.add_argument("--skip", type=int, default=1, help="most recent months to skip")
    p.add_argument("--cost-bps", type=float, default=10.0, help="cost per unit turnover, in bps")
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--require-positive", action="store_true",
                   help="hold cash instead of stocks with negative lookback return")
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--out", default="results")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--plot", action="store_true", help="save equity_curve.png (needs matplotlib)")
    args = p.parse_args()

    cfg = MomentumConfig(lookback_months=args.lookback, skip_months=args.skip,
                         top_n=args.top_n, require_positive=args.require_positive)

    tickers = load_universe(args.universe)
    print(f"Universe: {len(tickers)} tickers")
    data_start = (pd.Timestamp(args.start) - pd.DateOffset(months=args.lookback + 2)).strftime("%Y-%m-%d")
    prices = download_prices(tickers, data_start, args.end, use_cache=not args.no_cache)
    print(f"Prices: {prices.shape[1]} tickers, {prices.index[0].date()} -> {prices.index[-1].date()}")

    result = run_backtest(prices, cfg, start=args.start, cost_bps=args.cost_bps,
                          initial_capital=args.capital)

    curves = {"Momentum": result.equity}
    bench = download_prices([args.benchmark], data_start, args.end, use_cache=not args.no_cache)
    if args.benchmark in bench:
        b = bench[args.benchmark].reindex(result.equity.index).ffill()
        result.benchmark = b / b.iloc[0] * args.capital
        curves[args.benchmark] = result.benchmark

    table = pd.DataFrame({name: performance_metrics(eq) for name, eq in curves.items()})
    summary = (
        f"Momentum backtest  {result.equity.index[0].date()} -> {result.equity.index[-1].date()}\n"
        f"top_n={cfg.top_n}  lookback={cfg.lookback_months}m  skip={cfg.skip_months}m  "
        f"cost={args.cost_bps}bps  avg monthly turnover={result.turnover.mean():.1%}\n\n"
        f"{format_metrics(table)}\n\n"
        f"Final value: ${result.equity.iloc[-1]:,.0f} (from ${args.capital:,.0f})\n"
    )
    latest = result.holdings[result.holdings["date"] == result.holdings["date"].max()]
    summary += f"\nLatest holdings ({latest['date'].iloc[0].date()}):\n"
    summary += latest[["rank", "ticker", "score", "lookback_return", "volatility"]].to_string(index=False)
    print("\n" + summary)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    pd.DataFrame(curves).to_csv(out / "equity_curve.csv")
    result.holdings.to_csv(out / "holdings.csv", index=False)
    result.turnover.to_csv(out / "turnover.csv")
    (out / "summary.txt").write_text(summary)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping plot.")
        else:
            ax = pd.DataFrame(curves).plot(figsize=(11, 6), logy=True, title="Momentum vs benchmark")
            ax.set_ylabel("Portfolio value ($, log scale)")
            plt.tight_layout()
            plt.savefig(out / "equity_curve.png", dpi=120)
    print(f"\nSaved results to {out.resolve()}/")


if __name__ == "__main__":
    main()
