"""High-level, UI-agnostic entry points for building a portfolio.

This is the module the Gradio app (or anything else) should import from.
It never touches yfinance directly and has no UI concerns - it just wires
together data -> allocation method -> metrics.
"""

from dataclasses import dataclass, field

import pandas as pd

from portfolio_engine.assets import ASSET_CLASSES, TICKER_BY_KEY
from portfolio_engine.data import MarketData, get_market_data
from portfolio_engine.lifecycle import RiskTolerance, lifecycle_allocation
from portfolio_engine.metrics import portfolio_return, portfolio_volatility, sharpe_ratio
from portfolio_engine.optimizer import efficient_frontier, mean_variance_optimize

DEFAULT_RISK_FREE_RATE = 0.02


@dataclass
class PortfolioResult:
    method: str
    weights: pd.Series  # indexed by asset key, sums to 1.0
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    data_source: str  # "yfinance" or "fallback"
    data_note: str = ""
    inputs: dict = field(default_factory=dict)

    def weights_by_ticker(self) -> dict[str, float]:
        return {TICKER_BY_KEY[key]: w for key, w in self.weights.items()}

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "weights": self.weights.to_dict(),
            "weights_by_ticker": self.weights_by_ticker(),
            "expected_return": self.expected_return,
            "expected_volatility": self.expected_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "data_source": self.data_source,
            "data_note": self.data_note,
            "inputs": self.inputs,
        }


def _summarize(
    method: str,
    weights: pd.Series,
    market_data: MarketData,
    risk_free_rate: float,
    inputs: dict,
) -> PortfolioResult:
    exp_return = portfolio_return(weights, market_data.expected_returns)
    exp_vol = portfolio_volatility(weights, market_data.cov_matrix)
    sharpe = sharpe_ratio(exp_return, exp_vol, risk_free_rate)
    return PortfolioResult(
        method=method,
        weights=weights,
        expected_return=exp_return,
        expected_volatility=exp_vol,
        sharpe_ratio=sharpe,
        data_source=market_data.source,
        data_note=market_data.note,
        inputs=inputs,
    )


def build_lifecycle_portfolio(
    age: int,
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE,
    horizon_years: float = 20,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    market_data: MarketData | None = None,
) -> PortfolioResult:
    """Age-based ("110 minus age") heuristic allocation."""
    weights = lifecycle_allocation(age, risk_tolerance, horizon_years)
    market_data = market_data or get_market_data()
    inputs = {
        "age": age,
        "risk_tolerance": RiskTolerance(risk_tolerance).value,
        "horizon_years": horizon_years,
        "risk_free_rate": risk_free_rate,
    }
    return _summarize("lifecycle", weights, market_data, risk_free_rate, inputs)


def build_mvo_portfolio(
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    market_data: MarketData | None = None,
) -> PortfolioResult:
    """Long-only, max-Sharpe mean-variance optimized allocation."""
    market_data = market_data or get_market_data()
    weights = mean_variance_optimize(
        market_data.expected_returns, market_data.cov_matrix, risk_free_rate
    )
    inputs = {"risk_free_rate": risk_free_rate}
    return _summarize("mean_variance", weights, market_data, risk_free_rate, inputs)


def compute_efficient_frontier(
    market_data: MarketData | None = None, num_points: int = 25
) -> pd.DataFrame:
    """Long-only efficient frontier for the baseline asset universe."""
    market_data = market_data or get_market_data()
    return efficient_frontier(
        market_data.expected_returns, market_data.cov_matrix, num_points
    )


def asset_universe() -> list[dict]:
    """Metadata for the baseline asset universe, handy for a future UI."""
    return [
        {
            "key": a.key,
            "name": a.name,
            "ticker": a.ticker,
            "fallback_return": a.fallback_return,
            "fallback_volatility": a.fallback_volatility,
        }
        for a in ASSET_CLASSES
    ]
