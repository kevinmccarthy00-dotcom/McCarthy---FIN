import numpy as np
import pandas as pd
import pytest

from portfolio_engine.assets import ASSET_KEYS
from portfolio_engine.data import get_market_data
from portfolio_engine.optimizer import mean_variance_optimize


@pytest.fixture(scope="module")
def market_data():
    return get_market_data()  # exercises the real fallback path in this sandbox


def test_weights_sum_to_one_and_no_shorting(market_data):
    weights = mean_variance_optimize(market_data.expected_returns, market_data.cov_matrix)
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (weights >= -1e-9).all()
    assert set(weights.index) == set(ASSET_KEYS)


def test_optimizer_prefers_higher_sharpe_asset():
    # Two uncorrelated assets, second has a much better risk-adjusted return.
    # The optimizer should overweight it relative to an equal-weight split.
    expected_returns = pd.Series({"A": 0.05, "B": 0.10}, index=["A", "B"])
    cov_matrix = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.01]], index=["A", "B"], columns=["A", "B"]
    )
    weights = mean_variance_optimize(expected_returns, cov_matrix, risk_free_rate=0.0)
    assert weights["B"] > weights["A"]


def test_no_short_selling_even_with_negative_expected_return():
    expected_returns = pd.Series({"A": 0.08, "B": -0.05}, index=["A", "B"])
    cov_matrix = pd.DataFrame(
        [[0.03, 0.0], [0.0, 0.02]], index=["A", "B"], columns=["A", "B"]
    )
    weights = mean_variance_optimize(expected_returns, cov_matrix, risk_free_rate=0.0)
    assert weights["B"] >= -1e-9
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)
