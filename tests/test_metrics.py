import pandas as pd
import pytest

from portfolio_engine.metrics import (
    portfolio_return,
    portfolio_volatility,
    sharpe_ratio,
)


def test_portfolio_return_is_weighted_average():
    weights = pd.Series({"A": 0.6, "B": 0.4})
    expected_returns = pd.Series({"A": 0.10, "B": 0.05})
    assert portfolio_return(weights, expected_returns) == pytest.approx(0.08)


def test_portfolio_volatility_matches_manual_quadratic_form():
    weights = pd.Series({"A": 0.5, "B": 0.5})
    cov_matrix = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]], index=["A", "B"], columns=["A", "B"]
    )
    expected = (0.5**2 * 0.04 + 0.5**2 * 0.09 + 2 * 0.5 * 0.5 * 0.01) ** 0.5
    assert portfolio_volatility(weights, cov_matrix) == pytest.approx(expected)


def test_sharpe_ratio_basic():
    assert sharpe_ratio(0.10, 0.20, 0.02) == pytest.approx(0.4)


def test_sharpe_ratio_zero_volatility_is_zero_not_inf():
    assert sharpe_ratio(0.10, 0.0, 0.02) == 0.0
