"""Mean-variance optimization: long-only max-Sharpe (tangency) portfolio,
plus a risk-aversion-parameterized utility optimizer used to pick a
client-appropriate point on the efficient frontier.
"""

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


def maximize_utility(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_aversion: float,
) -> pd.Series:
    """Long-only, fully-invested portfolio maximizing mean-variance utility:

        U(w) = w . mu  -  0.5 * risk_aversion * w' Sigma w

    A higher risk_aversion pulls the result toward the global minimum-
    variance portfolio; a lower risk_aversion pulls it toward the
    highest-return corner. This is how the optimizer is made client-
    specific: risk tolerance and horizon are translated into a
    risk_aversion coefficient (see risk_profile.py) instead of always
    solving for the single max-Sharpe portfolio.
    """
    assets = list(expected_returns.index)
    n = len(assets)
    mu = expected_returns.values
    cov = cov_matrix.loc[assets, assets].values

    def negative_utility(w: np.ndarray) -> float:
        ret = float(np.dot(w, mu))
        var = float(w @ cov @ w)
        return -(ret - 0.5 * risk_aversion * var)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    initial_guess = np.full(n, 1.0 / n)

    result = minimize(
        negative_utility,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not result.success:
        raise RuntimeError(f"utility-maximizing optimization failed: {result.message}")

    weights = np.clip(result.x, 0.0, None)
    weights = weights / weights.sum()
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


def minimize_variance_for_target_return(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    target_return: float,
) -> pd.Series | None:
    """Long-only, fully-invested min-variance portfolio for a given target
    return. Returns None if no feasible portfolio hits that target.
    """
    assets = list(expected_returns.index)
    n = len(assets)
    mu = expected_returns.values
    cov = cov_matrix.loc[assets, assets].values

    def variance(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w: np.dot(w, mu) - target_return},
    )
    bounds = tuple((0.0, 1.0) for _ in range(n))
    initial_guess = np.full(n, 1.0 / n)

    result = minimize(
        variance,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not result.success:
        return None

    weights = np.clip(result.x, 0.0, None)
    total = weights.sum()
    if total <= 0:
        return None
    return pd.Series(weights / total, index=assets)


def efficient_frontier(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    num_points: int = 25,
) -> pd.DataFrame:
    """Long-only efficient frontier as (volatility, return) points.

    With no short selling, achievable portfolio returns are bounded by the
    lowest- and highest-returning individual asset, so the target grid
    spans that range.
    """
    mu = expected_returns.values
    low, high = float(mu.min()), float(mu.max())
    targets = np.linspace(low, high, num_points)

    rows = []
    for target in targets:
        weights = minimize_variance_for_target_return(
            expected_returns, cov_matrix, target
        )
        if weights is None:
            continue
        ret = portfolio_return(weights, expected_returns)
        vol = portfolio_volatility(weights, cov_matrix)
        rows.append({"expected_return": ret, "volatility": vol})

    return pd.DataFrame(rows)
