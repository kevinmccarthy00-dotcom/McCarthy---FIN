"""Mean-variance optimization: long-only max-Sharpe (tangency) portfolio."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from portfolio_engine.metrics import portfolio_return, portfolio_volatility


def mean_variance_optimize(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.02,
) -> pd.Series:
    """Long-only, fully-invested portfolio that maximizes the Sharpe ratio.

    Constraints: weights sum to 1, each weight in [0, 1] (no short selling).
    """
    assets = list(expected_returns.index)
    n = len(assets)
    mu = expected_returns.values
    cov = cov_matrix.loc[assets, assets].values

    def negative_sharpe(w: np.ndarray) -> float:
        ret = float(np.dot(w, mu))
        vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
        if vol == 0:
            return 0.0
        return -(ret - risk_free_rate) / vol

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    initial_guess = np.full(n, 1.0 / n)

    result = minimize(
        negative_sharpe,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not result.success:
        raise RuntimeError(f"mean-variance optimization failed: {result.message}")

    weights = np.clip(result.x, 0.0, None)
    weights = weights / weights.sum()  # clean up tiny negatives / float drift
    return pd.Series(weights, index=assets)


def portfolio_stats(
    weights: pd.Series,
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.02,
) -> tuple[float, float]:
    ret = portfolio_return(weights, expected_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    return ret, vol
