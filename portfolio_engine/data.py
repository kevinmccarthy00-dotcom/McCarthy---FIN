"""Market data loading: historical ETF stats via yfinance, with a
hardcoded fallback so the engine still works when yfinance is unreachable
or returns unusable data.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_engine.assets import (
    ASSET_CLASSES,
    ASSET_KEYS,
    FALLBACK_CORRELATION,
    TICKER_BY_KEY,
)

logger = logging.getLogger(__name__)

TRADING_MONTHS_PER_YEAR = 12
MIN_MONTHS_REQUIRED = 24  # need at least ~2 years of monthly data to trust it


@dataclass
class MarketData:
    expected_returns: pd.Series  # annualized, indexed by asset key
    cov_matrix: pd.DataFrame  # annualized, indexed/columned by asset key
    source: str  # "yfinance" or "fallback"
    note: str = ""


def _fallback_market_data(note: str = "") -> MarketData:
    vols = np.array([a.fallback_volatility for a in ASSET_CLASSES])
    corr = np.array(FALLBACK_CORRELATION)
    cov = np.outer(vols, vols) * corr

    expected_returns = pd.Series(
        [a.fallback_return for a in ASSET_CLASSES], index=ASSET_KEYS
    )
    cov_matrix = pd.DataFrame(cov, index=ASSET_KEYS, columns=ASSET_KEYS)
    return MarketData(
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        source="fallback",
        note=note,
    )


def fetch_price_history(period: str = "10y", interval: str = "1mo") -> pd.DataFrame:
    """Download adjusted-close price history for all baseline tickers.

    Raises on any failure (network error, missing package, empty/short
    response) so the caller can decide to fall back.
    """
    import yfinance as yf  # imported lazily so the package is optional at import time

    tickers = [a.ticker for a in ASSET_CLASSES]
    raw = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw is None or raw.empty:
        raise ValueError("yfinance returned no data")

    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    prices = prices.dropna(how="all")

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise ValueError(f"yfinance response missing tickers: {missing}")

    prices = prices[tickers].dropna()
    if len(prices) < MIN_MONTHS_REQUIRED:
        raise ValueError(
            f"only {len(prices)} usable price rows, need at least {MIN_MONTHS_REQUIRED}"
        )

    prices = prices.rename(columns={t: k for k, t in TICKER_BY_KEY.items()})
    return prices


def compute_annualized_stats(prices: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Turn a monthly price history into annualized expected returns + covariance."""
    monthly_returns = prices.pct_change().dropna(how="all").dropna()
    if monthly_returns.empty:
        raise ValueError("no returns could be computed from price history")

    expected_returns = monthly_returns.mean() * TRADING_MONTHS_PER_YEAR
    cov_matrix = monthly_returns.cov() * TRADING_MONTHS_PER_YEAR
    return expected_returns[ASSET_KEYS], cov_matrix.loc[ASSET_KEYS, ASSET_KEYS]


def get_market_data(period: str = "10y") -> MarketData:
    """Best-effort live market data via yfinance, else fallback assumptions."""
    try:
        prices = fetch_price_history(period=period)
        expected_returns, cov_matrix = compute_annualized_stats(prices)
        return MarketData(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            source="yfinance",
        )
    except Exception as exc:  # noqa: BLE001 - any failure means "use fallback"
        logger.warning("Falling back to hardcoded market assumptions: %s", exc)
        return _fallback_market_data(note=str(exc))
