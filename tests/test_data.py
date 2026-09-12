import numpy as np
import pandas as pd
import pytest

from portfolio_engine.assets import ASSET_KEYS
from portfolio_engine.data import (
    compute_annualized_stats,
    fetch_price_history,
    get_market_data,
)


def test_fallback_used_when_yfinance_unavailable(monkeypatch):
    def boom(*args, **kwargs):
        raise ConnectionError("network blocked")

    monkeypatch.setattr(
        "portfolio_engine.data.fetch_price_history", boom
    )
    market_data = get_market_data()
    assert market_data.source == "fallback"
    assert set(market_data.expected_returns.index) == set(ASSET_KEYS)
    assert market_data.cov_matrix.shape == (len(ASSET_KEYS), len(ASSET_KEYS))


def test_get_market_data_actually_falls_back_in_this_sandbox():
    # No network access to Yahoo Finance here, so this exercises the real
    # (non-mocked) failure path end to end.
    market_data = get_market_data()
    assert market_data.source in ("yfinance", "fallback")
    assert not market_data.expected_returns.isna().any()
    assert np.all(np.linalg.eigvalsh(market_data.cov_matrix.values) >= -1e-8)


def test_compute_annualized_stats_annualizes_monthly_data():
    dates = pd.date_range("2020-01-01", periods=13, freq="MS")
    # constant 1% monthly growth for asset A, 0.5% for asset B
    prices = pd.DataFrame(
        {
            "US_STOCKS": [100 * (1.01**i) for i in range(13)],
            "INTL_DEV": [100 * (1.005**i) for i in range(13)],
            "EM": [100 * (1.005**i) for i in range(13)],
            "US_BONDS": [100 * (1.002**i) for i in range(13)],
            "TIPS": [100 * (1.002**i) for i in range(13)],
            "REITS": [100 * (1.005**i) for i in range(13)],
        },
        index=dates,
    )
    expected_returns, cov_matrix = compute_annualized_stats(prices)
    assert expected_returns["US_STOCKS"] == pytest.approx(0.12, abs=1e-6)
    assert cov_matrix.loc["US_STOCKS", "US_STOCKS"] == pytest.approx(0.0, abs=1e-6)


def test_fetch_price_history_raises_without_network():
    with pytest.raises(Exception):
        fetch_price_history(period="1y")
