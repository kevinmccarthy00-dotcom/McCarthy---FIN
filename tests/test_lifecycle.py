import pytest

from portfolio_engine.assets import ASSET_KEYS
from portfolio_engine.lifecycle import (
    RiskTolerance,
    lifecycle_allocation,
    lifecycle_equity_pct,
)


def test_equity_pct_matches_110_minus_age_for_moderate_mid_horizon():
    # moderate + 10-20yr horizon has zero adjustment, so this is the pure
    # "110 minus age" baseline.
    pct = lifecycle_equity_pct(30, RiskTolerance.MODERATE, 15)
    assert pct == pytest.approx(0.80)


def test_conservative_lowers_and_aggressive_raises_equity():
    conservative = lifecycle_equity_pct(40, RiskTolerance.CONSERVATIVE, 15)
    moderate = lifecycle_equity_pct(40, RiskTolerance.MODERATE, 15)
    aggressive = lifecycle_equity_pct(40, RiskTolerance.AGGRESSIVE, 15)
    assert conservative < moderate < aggressive


def test_longer_horizon_raises_equity_allocation():
    short = lifecycle_equity_pct(40, RiskTolerance.MODERATE, 3)
    long = lifecycle_equity_pct(40, RiskTolerance.MODERATE, 25)
    assert long > short


def test_equity_pct_is_clamped():
    very_old_conservative_short = lifecycle_equity_pct(95, RiskTolerance.CONSERVATIVE, 1)
    very_young_aggressive_long = lifecycle_equity_pct(5, RiskTolerance.AGGRESSIVE, 30)
    assert 0.10 <= very_old_conservative_short <= 1.0
    assert 0.0 <= very_young_aggressive_long <= 0.95


def test_allocation_weights_sum_to_one_and_are_non_negative():
    weights = lifecycle_allocation(35, RiskTolerance.MODERATE, 20)
    assert set(weights.index) == set(ASSET_KEYS)
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0).all()


def test_invalid_age_raises():
    with pytest.raises(ValueError):
        lifecycle_allocation(-1)
    with pytest.raises(ValueError):
        lifecycle_allocation(150)
