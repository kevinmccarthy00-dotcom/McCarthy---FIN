import pytest

from portfolio_engine.assets import ASSET_KEYS
from portfolio_engine.engine import build_lifecycle_portfolio, build_mvo_portfolio
from portfolio_engine.lifecycle import RiskTolerance


def test_lifecycle_portfolio_end_to_end():
    result = build_lifecycle_portfolio(age=30, risk_tolerance=RiskTolerance.MODERATE, horizon_years=25)
    assert result.method == "lifecycle"
    assert result.weights.sum() == pytest.approx(1.0)
    assert (result.weights >= 0).all()
    assert result.expected_volatility > 0
    assert result.data_source in ("yfinance", "fallback")
    tickers = set(result.weights_by_ticker().keys())
    assert len(tickers) == len(ASSET_KEYS)


def test_mvo_portfolio_end_to_end():
    result = build_mvo_portfolio()
    assert result.method == "mean_variance"
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (result.weights >= -1e-9).all()
    assert result.expected_volatility > 0


def test_older_investor_gets_less_equity_than_younger_investor():
    from portfolio_engine.assets import EQUITY_KEYS
    from portfolio_engine.data import get_market_data

    market_data = get_market_data()
    young = build_lifecycle_portfolio(age=25, market_data=market_data)
    old = build_lifecycle_portfolio(age=65, market_data=market_data)
    young_equity = young.weights[EQUITY_KEYS].sum()
    old_equity = old.weights[EQUITY_KEYS].sum()
    assert young_equity > old_equity


def test_result_to_dict_is_json_serializable_shape():
    import json

    result = build_lifecycle_portfolio(age=45)
    payload = result.to_dict()
    json.dumps(payload)  # should not raise
    assert "weights" in payload and "expected_return" in payload
