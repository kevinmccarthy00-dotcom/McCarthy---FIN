"""Portfolio-level return/risk metrics shared by both allocation methods."""

import numpy as np
import pandas as pd


def portfolio_return(weights: pd.Series, expected_returns: pd.Series) -> float:
    return float(np.dot(weights.values, expected_returns.values))


def portfolio_volatility(weights: pd.Series, cov_matrix: pd.DataFrame) -> float:
    variance = weights.values @ cov_matrix.values @ weights.values
    return float(np.sqrt(max(variance, 0.0)))


def sharpe_ratio(
    expected_return: float, volatility: float, risk_free_rate: float
) -> float:
    if volatility == 0:
        return 0.0
    return (expected_return - risk_free_rate) / volatility
