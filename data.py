"""Market data helpers: universe lookup and price downloads via yfinance."""

from __future__ import annotations

import hashlib
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CACHE_DIR = Path(__file__).parent / ".cache"

# Used when Wikipedia can't be reached.
FALLBACK_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "AVGO", "TSLA", "JPM",
    "LLY", "V", "UNH", "XOM", "MA", "JNJ", "PG", "HD", "COST", "MRK",
    "ABBV", "CVX", "PEP", "KO", "ADBE", "WMT", "CRM", "BAC", "NFLX", "AMD",
    "TMO", "MCD", "CSCO", "ACN", "LIN", "ABT", "ORCL", "DHR", "INTU", "DIS",
    "WFC", "TXN", "QCOM", "CAT", "AMGN", "IBM", "GE", "HON", "NOW", "SPGI",
]


def get_sp500_tickers() -> list[str]:
    """Current S&P 500 constituents from Wikipedia (yfinance-style symbols)."""
    try:
        resp = requests.get(SP500_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        resp.raise_for_status()
        table = pd.read_html(StringIO(resp.text))[0]
        tickers = table["Symbol"].astype(str).str.replace(".", "-", regex=False)
        return sorted(tickers.unique().tolist())
    except Exception as exc:  # network / parsing failure
        print(f"[data] Could not fetch S&P 500 list ({exc}); using fallback universe.")
        return list(FALLBACK_UNIVERSE)


def load_universe(spec: str) -> list[str]:
    """`spec` is 'sp500', 'fallback', a path to a file of tickers, or a comma list."""
    if spec == "sp500":
        return get_sp500_tickers()
    if spec == "fallback":
        return list(FALLBACK_UNIVERSE)
    path = Path(spec)
    if path.exists():
        text = path.read_text().replace(",", "\n")
        return sorted({t.strip().upper() for t in text.splitlines() if t.strip()})
    return sorted({t.strip().upper() for t in spec.split(",") if t.strip()})


def download_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    chunk_size: int = 50,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Daily split/dividend-adjusted closes, one column per ticker."""
    key = hashlib.md5(f"{sorted(tickers)}|{start}|{end}".encode()).hexdigest()[:12]
    cache_file = CACHE_DIR / f"prices_{key}.pkl"
    if use_cache and cache_file.exists():
        return pd.read_pickle(cache_file)

    frames = []
    chunks = [tickers[i : i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    for chunk in tqdm(chunks, desc="Downloading prices", unit="batch"):
        raw = yf.download(
            chunk,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if raw is None or raw.empty:
            continue
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        if isinstance(close, pd.Series):
            close = close.to_frame(chunk[0])
        if list(close.columns) == ["Close"]:
            close.columns = chunk[:1]
        frames.append(close)

    if not frames:
        raise RuntimeError("No price data downloaded — check tickers / network.")

    prices = pd.concat(frames, axis=1).sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated()].dropna(axis=1, how="all")
    prices.index = pd.to_datetime(prices.index).tz_localize(None)

    if use_cache:
        CACHE_DIR.mkdir(exist_ok=True)
        prices.to_pickle(cache_file)
    return prices
