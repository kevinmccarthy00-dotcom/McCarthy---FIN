import pytest

from portfolio_engine.assets import ASSET_KEYS
from portfolio_engine.engine import (
    build_lifecycle_portfolio,
    build_mvo_portfolio,
    build_research_informed_portfolio,
)
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


def test_mvo_portfolio_responds_to_risk_tolerance():
    from portfolio_engine.assets import EQUITY_KEYS
    from portfolio_engine.data import get_market_data

    market_data = get_market_data()
    conservative = build_mvo_portfolio(
        risk_tolerance=RiskTolerance.CONSERVATIVE, horizon_years=20, market_data=market_data
    )
    aggressive = build_mvo_portfolio(
        risk_tolerance=RiskTolerance.AGGRESSIVE, horizon_years=20, market_data=market_data
    )
    assert aggressive.weights[EQUITY_KEYS].sum() > conservative.weights[EQUITY_KEYS].sum()
    assert aggressive.expected_volatility > conservative.expected_volatility


def test_mvo_portfolio_responds_to_horizon():
    from portfolio_engine.assets import EQUITY_KEYS
    from portfolio_engine.data import get_market_data

    market_data = get_market_data()
    short_horizon = build_mvo_portfolio(
        risk_tolerance=RiskTolerance.MODERATE, horizon_years=3, market_data=market_data
    )
    long_horizon = build_mvo_portfolio(
        risk_tolerance=RiskTolerance.MODERATE, horizon_years=30, market_data=market_data
    )
    assert long_horizon.weights[EQUITY_KEYS].sum() > short_horizon.weights[EQUITY_KEYS].sum()


def test_two_standard_test_clients_get_different_mvo_portfolios():
    # Regression guard: same risk tolerance (moderate) but different
    # horizons (30y vs 20y) must not collapse to an identical portfolio.
    from portfolio_engine.data import get_market_data

    market_data = get_market_data()
    client_a = build_mvo_portfolio(
        risk_tolerance=RiskTolerance.MODERATE, horizon_years=30, market_data=market_data
    )
    client_b = build_mvo_portfolio(
        risk_tolerance=RiskTolerance.MODERATE, horizon_years=20, market_data=market_data
    )
    assert not client_a.weights.equals(client_b.weights)


def test_research_informed_portfolio_end_to_end():
    result = build_research_informed_portfolio(
        age=40, risk_tolerance=RiskTolerance.MODERATE, horizon_years=20,
        initial_investment=250_000, monthly_contribution=500,
    )
    assert result.method == "research_informed"
    assert result.weights.sum() == pytest.approx(1.0)
    assert (result.weights >= 0).all()
    assert result.expected_volatility > 0
    tickers = set(result.weights_by_ticker().keys())
    assert len(tickers) == len(ASSET_KEYS)


def test_research_informed_standard_clients():
    from portfolio_engine.assets import EQUITY_KEYS
    from portfolio_engine.data import get_market_data

    market_data = get_market_data()
    client_a = build_research_informed_portfolio(
        age=35, risk_tolerance=RiskTolerance.MODERATE, horizon_years=30,
        initial_investment=100_000, monthly_contribution=2_000, market_data=market_data,
    )
    client_b = build_research_informed_portfolio(
        age=68, risk_tolerance=RiskTolerance.MODERATE, horizon_years=20,
        initial_investment=1_500_000, monthly_contribution=0, market_data=market_data,
    )
    client_b_equity = client_b.weights[EQUITY_KEYS].sum()
    assert 0.55 <= client_b_equity <= 0.65
    assert client_a.weights[EQUITY_KEYS].sum() > client_b_equity


def test_adding_research_informed_does_not_change_baseline_methods():
    # Regression guard: lifecycle and MVO must be untouched by the new
    # third method.
    from portfolio_engine.data import get_market_data

    market_data = get_market_data()
    lifecycle = build_lifecycle_portfolio(
        age=68, risk_tolerance=RiskTolerance.MODERATE, horizon_years=20, market_data=market_data
    )
    mvo = build_mvo_portfolio(
        risk_tolerance=RiskTolerance.MODERATE, horizon_years=20, market_data=market_data
    )
    # values pinned from before research.py existed
    assert lifecycle.expected_return == pytest.approx(0.0556, abs=1e-3)
    assert mvo.expected_return == pytest.approx(0.0500, abs=1e-3)
