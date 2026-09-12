"""Wealth projection over time given a contribution schedule and an
annual return assumption. Pure math, no UI concerns - the Gradio layer
just calls this and plots the result.
"""

import pandas as pd

MIN_ANNUAL_RETURN = -0.95  # guards against a fractional power of a negative base


def project_wealth_path(
    initial_investment: float,
    monthly_contribution: float,
    annual_return: float,
    years: int,
) -> pd.Series:
    """Year-end portfolio value for 0..years, given a constant annual return.

    Contributions are applied at the end of each month; growth compounds monthly.
    """
    annual_return = max(annual_return, MIN_ANNUAL_RETURN)
    monthly_rate = (1 + annual_return) ** (1 / 12) - 1

    values = {0: float(initial_investment)}
    balance = float(initial_investment)
    for month in range(1, years * 12 + 1):
        balance = balance * (1 + monthly_rate) + monthly_contribution
        if month % 12 == 0:
            values[month // 12] = balance

    return pd.Series(values).sort_index()


def wealth_scenarios(
    initial_investment: float,
    monthly_contribution: float,
    expected_return: float,
    volatility: float,
    years: int,
) -> dict[str, pd.Series]:
    """Expected / optimistic / pessimistic wealth paths.

    Optimistic and pessimistic scenarios use +/- one standard deviation
    (the portfolio's expected volatility) around the expected annual return -
    a simple illustrative band, not a statistical confidence interval.
    """
    return {
        "expected": project_wealth_path(
            initial_investment, monthly_contribution, expected_return, years
        ),
        "optimistic": project_wealth_path(
            initial_investment,
            monthly_contribution,
            expected_return + volatility,
            years,
        ),
        "pessimistic": project_wealth_path(
            initial_investment,
            monthly_contribution,
            expected_return - volatility,
            years,
        ),
    }
