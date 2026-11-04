"""Market data helpers: universe lookup and price downloads via yfinance."""

from __future__ import annotations

import hashlib
import logging
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# Point-in-time S&P 500 constituents (https://github.com/fja05680/sp500).
SP500_HISTORY_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
CACHE_DIR = Path(__file__).parent / ".cache"

# The membership history uses the ticker a company had at the time; Yahoo lists
# a renamed company's full history only under its current ticker.
TICKER_RENAMES = {
    "FB": "META", "ANTM": "ELV", "ABC": "COR", "BLL": "BALL", "PKI": "RVTY",
    "FLT": "CPAY", "RE": "EG", "WLTW": "WTW", "NLOK": "GEN", "SYMC": "GEN",
    "FBHS": "FBIN", "HFC": "DINO", "CTL": "LUMN", "GPS": "GAP", "ADS": "BFH",
    "PEAK": "DOC", "HCP": "DOC", "JEC": "J", "TMK": "GL", "UTX": "RTX",
    "DWDP": "DD", "HRS": "LHX", "BK": "BNY", "BHGE": "BKR", "FI": "FISV",
    "ARNC": "HWM",
}

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


def _yahoo_symbol(ticker: str) -> str:
    ticker = ticker.strip().replace(".", "-")
    return TICKER_RENAMES.get(ticker, ticker)


def load_sp500_membership(use_cache: bool = True) -> pd.Series:
    """Historical S&P 500 constituents: a Series indexed by date whose values are
    frozensets of tickers, each valid until the next date in the index."""
    cache_file = CACHE_DIR / "sp500_membership.csv"
    fresh = cache_file.exists() and time.time() - cache_file.stat().st_mtime < 7 * 86400
    if not (use_cache and fresh):
        resp = requests.get(SP500_HISTORY_URL, timeout=30)
        resp.raise_for_status()
        CACHE_DIR.mkdir(exist_ok=True)
        cache_file.write_text(resp.text)
    raw = pd.read_csv(cache_file, parse_dates=["date"])
    members = {
        row.date: frozenset(_yahoo_symbol(t) for t in row.tickers.split(","))
        for row in raw.itertuples()
    }
    # The history file lags; today's Wikipedia list covers changes since its last date.
    if current := _fetch_current_members():
        members[pd.Timestamp.today().normalize()] = current
    return pd.Series(members).sort_index()


def _fetch_current_members() -> frozenset[str] | None:
    try:
        resp = requests.get(SP500_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        resp.raise_for_status()
        symbols = pd.read_html(StringIO(resp.text))[0]["Symbol"].astype(str)
        return frozenset(_yahoo_symbol(t) for t in symbols)
    except Exception:
        return None


def members_on(membership: pd.Series, date) -> frozenset[str]:
    """Index constituents as of `date` (the most recent snapshot on or before it)."""
    pos = membership.index.searchsorted(pd.Timestamp(date), side="right") - 1
    return membership.iloc[pos] if pos >= 0 else frozenset()


def tickers_since(membership: pd.Series, start) -> list[str]:
    """Every ticker that was in the index at any point from `start` onward."""
    start = pd.Timestamp(start)
    return sorted(frozenset().union(members_on(membership, start), *membership.loc[start:]))


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

    # Delisted tickers fail to download; that's expected, so keep yfinance quiet.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    frames = []
    chunks =[tickers[i : i + chunk_size] for i in range(0, len(tickers), chunk_size)]
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
