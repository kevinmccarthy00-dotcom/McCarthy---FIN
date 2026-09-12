import pytest

from portfolio_engine.lifecycle import RiskTolerance
from portfolio_engine.risk_profile import (
    MAX_RISK_AVERSION,
    MIN_RISK_AVERSION,
    risk_aversion_for_profile,
)


def test_conservative_is_more_risk_averse_than_aggressive_at_same_horizon():
    conservative = risk_aversion_for_profile(RiskTolerance.CONSERVATIVE, 20)
    moderate = risk_aversion_for_profile(RiskTolerance.MODERATE, 20)
    aggressive = risk_aversion_for_profile(RiskTolerance.AGGRESSIVE, 20)
    assert conservative > moderate > aggressive


def test_shorter_horizon_increases_risk_aversion():
    short = risk_aversion_for_profile(RiskTolerance.MODERATE, 3)
    long = risk_aversion_for_profile(RiskTolerance.MODERATE, 30)
    assert short > long


def test_risk_aversion_is_clamped_to_bounds():
    for tolerance in RiskTolerance:
        for horizon in (1, 5, 10, 20, 30):
            value = risk_aversion_for_profile(tolerance, horizon)
            assert MIN_RISK_AVERSION <= value <= MAX_RISK_AVERSION


def test_different_horizons_give_different_risk_aversion_for_same_tolerance():
    # Regression guard for the two standard test clients: same risk
    # tolerance (moderate) but different horizons (30y vs 20y) must not
    # collapse to the same risk-aversion coefficient.
    a = risk_aversion_for_profile(RiskTolerance.MODERATE, 30)
    b = risk_aversion_for_profile(RiskTolerance.MODERATE, 20)
    assert a != b
