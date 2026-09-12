"""Lifecycle / heuristic allocation: age-driven equity glide path, adjusted
for risk tolerance and investment horizon, then split across asset classes.
"""

from enum import Enum

import pandas as pd

from portfolio_engine.assets import (
    ASSET_KEYS,
    EQUITY_SLEEVE_SPLIT,
    FIXED_INCOME_SLEEVE_SPLIT,
)

MIN_EQUITY_PCT = 0.10
MAX_EQUITY_PCT = 0.95


class RiskTolerance(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


# Shared with the research-informed method (research.py) so both heuristics
# apply the same risk-tolerance/horizon nudges to their own age-driven
# baseline instead of each defining their own numbers.
RISK_TOLERANCE_ADJUSTMENT = {
    RiskTolerance.CONSERVATIVE: -0.10,
    RiskTolerance.MODERATE: 0.0,
    RiskTolerance.AGGRESSIVE: 0.10,
}


def risk_tolerance_adjustment(risk_tolerance: RiskTolerance) -> float:
    return RISK_TOLERANCE_ADJUSTMENT[RiskTolerance(risk_tolerance)]


def horizon_adjustment(horizon_years: float) -> float:
    if horizon_years >= 20:
        return 0.05
    if horizon_years >= 10:
        return 0.0
    if horizon_years >= 5:
        return -0.05
    return -0.15


def lifecycle_equity_pct(
    age: int, risk_tolerance: RiskTolerance, horizon_years: float
) -> float:
    """"110 minus age" baseline, nudged by risk tolerance and horizon, clamped."""
    base = (110 - age) / 100
    adjusted = (
        base
        + risk_tolerance_adjustment(risk_tolerance)
        + horizon_adjustment(horizon_years)
    )
    return min(max(adjusted, MIN_EQUITY_PCT), MAX_EQUITY_PCT)


def lifecycle_allocation(
    age: int,
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE,
    horizon_years: float = 20,
) -> pd.Series:
    """Full asset-class weight vector for the lifecycle heuristic."""
    if age < 0 or age > 100:
        raise ValueError("age must be between 0 and 100")
    if horizon_years < 0:
        raise ValueError("horizon_years must be non-negative")

    equity_pct = lifecycle_equity_pct(age, risk_tolerance, horizon_years)
    fixed_income_pct = 1.0 - equity_pct

    weights = {}
    for key, share in EQUITY_SLEEVE_SPLIT.items():
        weights[key] = equity_pct * share
    for key, share in FIXED_INCOME_SLEEVE_SPLIT.items():
        weights[key] = fixed_income_pct * share

    series = pd.Series(weights).reindex(ASSET_KEYS).fillna(0.0)
    series = series / series.sum()  # guard against float drift
    return series
