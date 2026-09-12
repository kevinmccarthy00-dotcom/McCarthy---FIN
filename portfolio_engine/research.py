"""Research-informed allocation: a hump-shaped, human-capital-motivated
age glide path, adjusted for risk tolerance, investment horizon, and the
client's wealth and ongoing savings capacity - not just age, unlike the
baseline lifecycle heuristic in lifecycle.py.

Rationale (the baseline app's own synthesis of well-known glide-path
research themes, not a transcription of one specific published formula -
swap in the assignment's exact prescribed curve/coefficients if it
specifies one):

- Human capital theory motivates a HUMP rather than a straight-line
  "N minus age" decline. Early career risk capacity is tempered by thin
  savings, higher debt, and career/income uncertainty even though the
  time horizon is long; risk capacity then rises through the 30s-40s as
  earnings and savings stabilize, peaking around the mid-40s when human
  capital is still substantial and retirement withdrawals are far off.
  It declines approaching retirement as human capital runs down and
  sequence-of-returns risk grows near the withdrawal phase (Blanchett,
  2007 "Dynamic Allocation Strategies for Distribution Portfolios";
  Kitces & Pfau's glide-path research). Consistent with the "rising
  equity glide path in retirement" literature (Pfau & Kitces, 2014), the
  curve does not collapse to a low equity floor right at retirement.
- Wealth and ongoing contribution capacity (used here as a proxy for
  income, since the baseline app does not collect income directly) are
  folded in via Blanchett's "ability to take risk" framing: a larger
  cushion - either banked wealth or a strong ongoing savings rate -
  supports a somewhat higher equity allocation than age alone would
  suggest, and a thin one supports somewhat less.
"""

import numpy as np
import pandas as pd

from portfolio_engine.assets import (
    ASSET_KEYS,
    EQUITY_SLEEVE_SPLIT,
    FIXED_INCOME_SLEEVE_SPLIT,
)
from portfolio_engine.lifecycle import (
    MAX_EQUITY_PCT,
    MIN_EQUITY_PCT,
    RiskTolerance,
    horizon_adjustment,
    risk_tolerance_adjustment,
)

# Hump-shaped baseline equity % by age: rises through early/mid career,
# peaks in the mid-40s, then declines - but levels off well above zero
# rather than continuing to fall through retirement.
_AGE_CONTROL_POINTS = [18, 25, 35, 45, 55, 65, 75, 80]
_EQUITY_CONTROL_POINTS = [0.55, 0.68, 0.80, 0.85, 0.75, 0.58, 0.45, 0.42]

_WEALTH_TILT_BREAKPOINTS = [50_000, 250_000, 1_000_000]
_WEALTH_TILTS = [-0.05, 0.0, 0.02, 0.03]

_CONTRIBUTION_TILT_BREAKPOINTS = [200, 1_000]
_CONTRIBUTION_TILTS = [-0.02, 0.0, 0.03]


def _hump_shaped_baseline_equity_pct(age: float) -> float:
    return float(np.interp(age, _AGE_CONTROL_POINTS, _EQUITY_CONTROL_POINTS))


def _wealth_tilt(initial_investment: float) -> float:
    for breakpoint, tilt in zip(_WEALTH_TILT_BREAKPOINTS, _WEALTH_TILTS):
        if initial_investment < breakpoint:
            return tilt
    return _WEALTH_TILTS[-1]


def _income_proxy_tilt(monthly_contribution: float) -> float:
    # No ongoing contribution is treated as neutral (e.g. a retiree drawing
    # down savings), not as a sign of low income - that's already captured
    # by the wealth tilt above.
    if monthly_contribution <= 0:
        return 0.0
    for breakpoint, tilt in zip(_CONTRIBUTION_TILT_BREAKPOINTS, _CONTRIBUTION_TILTS):
        if monthly_contribution < breakpoint:
            return tilt
    return _CONTRIBUTION_TILTS[-1]


def research_equity_pct(
    age: float,
    risk_tolerance: RiskTolerance,
    horizon_years: float,
    initial_investment: float,
    monthly_contribution: float,
) -> float:
    baseline = _hump_shaped_baseline_equity_pct(age)
    adjusted = (
        baseline
        + risk_tolerance_adjustment(risk_tolerance)
        + horizon_adjustment(horizon_years)
        + _wealth_tilt(initial_investment)
        + _income_proxy_tilt(monthly_contribution)
    )
    return min(max(adjusted, MIN_EQUITY_PCT), MAX_EQUITY_PCT)


def research_informed_allocation(
    age: float,
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE,
    horizon_years: float = 20,
    initial_investment: float = 10_000,
    monthly_contribution: float = 0,
) -> pd.Series:
    """Full asset-class weight vector for the research-informed heuristic."""
    if age < 0 or age > 100:
        raise ValueError("age must be between 0 and 100")
    if horizon_years < 0:
        raise ValueError("horizon_years must be non-negative")
    if initial_investment < 0:
        raise ValueError("initial_investment must be non-negative")
    if monthly_contribution < 0:
        raise ValueError("monthly_contribution must be non-negative")

    equity_pct = research_equity_pct(
        age, risk_tolerance, horizon_years, initial_investment, monthly_contribution
    )
    fixed_income_pct = 1.0 - equity_pct

    weights = {}
    for key, share in EQUITY_SLEEVE_SPLIT.items():
        weights[key] = equity_pct * share
    for key, share in FIXED_INCOME_SLEEVE_SPLIT.items():
        weights[key] = fixed_income_pct * share

    series = pd.Series(weights).reindex(ASSET_KEYS).fillna(0.0)
    series = series / series.sum()  # guard against float drift
    return series
