import pytest

from portfolio_engine.data import get_market_data
from portfolio_engine.optimizer import (
    efficient_frontier,
    minimize_variance_for_target_return,
)


@pytest.fixture(scope="module")
def market_data():
    return get_market_data()


def test_frontier_returns_points_spanning_min_to_max_asset_return(market_data):
    frontier = efficient_frontier(market_data.expected_returns, market_data.cov_matrix, num_points=15)
    assert len(frontier) > 0
    assert frontier["expected_return"].min() == pytest.approx(
        market_data.expected_returns.min(), abs=1e-3
    )
    assert frontier["expected_return"].max() == pytest.approx(
        market_data.expected_returns.max(), abs=1e-3
    )


def test_frontier_volatility_is_non_negative(market_data):
    frontier = efficient_frontier(market_data.expected_returns, market_data.cov_matrix, num_points=15)
    assert (frontier["volatility"] >= 0).all()


def test_target_return_below_min_asset_is_infeasible(market_data):
    too_low = market_data.expected_returns.min() - 0.10
    weights = minimize_variance_for_target_return(
        market_data.expected_returns, market_data.cov_matrix, too_low
    )
    assert weights is None


def test_target_return_at_a_feasible_point_sums_to_one(market_data):
    mid = (market_data.expected_returns.min() + market_data.expected_returns.max()) / 2
    weights = minimize_variance_for_target_return(
        market_data.expected_returns, market_data.cov_matrix, mid
    )
    assert weights is not None
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (weights >= -1e-9).all()
