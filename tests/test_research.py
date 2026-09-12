import pytest

from portfolio_engine.assets import ASSET_KEYS, EQUITY_KEYS
from portfolio_engine.lifecycle import RiskTolerance
from portfolio_engine.research import research_equity_pct, research_informed_allocation


def test_equity_pct_is_hump_shaped_with_age():
    ages = [18, 25, 35, 45, 55, 65, 75, 80]
    pcts = [
        research_equity_pct(age, RiskTolerance.MODERATE, 20, 100_000, 0) for age in ages
    ]
    peak_index = pcts.index(max(pcts))
    # peak should be somewhere in the middle of the age range, not at either end
    assert 1 <= peak_index <= len(ages) - 2
    # rising into the peak
    assert pcts[:peak_index + 1] == sorted(pcts[:peak_index + 1])
    # falling after the peak
    assert pcts[peak_index:] == sorted(pcts[peak_index:], reverse=True)


def test_higher_wealth_increases_equity_allocation_all_else_equal():
    low_wealth = research_equity_pct(50, RiskTolerance.MODERATE, 20, 20_000, 0)
    high_wealth = research_equity_pct(50, RiskTolerance.MODERATE, 20, 2_000_000, 0)
    assert high_wealth > low_wealth


def test_stronger_contribution_increases_equity_allocation_all_else_equal():
    low_contribution = research_equity_pct(40, RiskTolerance.MODERATE, 20, 100_000, 50)
    high_contribution = research_equity_pct(40, RiskTolerance.MODERATE, 20, 100_000, 2_000)
    assert high_contribution > low_contribution


def test_zero_contribution_is_not_penalized_like_a_low_contribution():
    zero_contribution = research_equity_pct(68, RiskTolerance.MODERATE, 20, 1_500_000, 0)
    tiny_contribution = research_equity_pct(68, RiskTolerance.MODERATE, 20, 1_500_000, 50)
    assert zero_contribution >= tiny_contribution


def test_allocation_weights_sum_to_one_and_are_non_negative():
    weights = research_informed_allocation(45, RiskTolerance.AGGRESSIVE, 20, 500_000, 1_000)
    assert set(weights.index) == set(ASSET_KEYS)
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0).all()


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        research_informed_allocation(-1)
    with pytest.raises(ValueError):
        research_informed_allocation(150)
    with pytest.raises(ValueError):
        research_informed_allocation(40, initial_investment=-1)
    with pytest.raises(ValueError):
        research_informed_allocation(40, monthly_contribution=-1)


def test_client_a_profile_lands_high_equity():
    # Age 35, moderate, 30yr horizon, $100,000 initial, $2,000/mo, retirement
    weights = research_informed_allocation(35, RiskTolerance.MODERATE, 30, 100_000, 2_000)
    equity_pct = weights[EQUITY_KEYS].sum()
    assert equity_pct > 0.75  # young, long horizon, strong savings rate


def test_client_b_profile_lands_in_55_to_65_percent_equity():
    # Age 68, moderate, 20yr horizon, $1,500,000 initial, $0/mo, retirement
    weights = research_informed_allocation(68, RiskTolerance.MODERATE, 20, 1_500_000, 0)
    equity_pct = weights[EQUITY_KEYS].sum()
    assert 0.55 <= equity_pct <= 0.65
