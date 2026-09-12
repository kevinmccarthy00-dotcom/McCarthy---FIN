import math

import pandas as pd
import pytest

from portfolio_engine.assets import ASSET_KEYS, EQUITY_KEYS
from portfolio_engine.lifecycle import RiskTolerance
from portfolio_engine.research import (
    human_capital,
    merton_equity_share,
    research_equity_pct,
    research_informed_allocation,
)


def test_human_capital_matches_paper_worked_example():
    # Choi, Liu & Liu (2025) Section 3.3: age 55, risk aversion 7, log
    # risk-free rate 2%, log equity premium 2%, replacement rate 40%,
    # college-graduate income risk, $100,000 income through age 66 then a
    # (lambda-implied) $40,000 retirement benefit. The paper reports
    # H = $924,805 and a resulting equity share of 30% (vs. their exact
    # optimum of 33%, since this is a fitted approximation, not an exact
    # formula - reproducing their number validates our transcription of
    # their regression coefficients).
    H = human_capital(
        age=55,
        annual_income=100_000,
        risk_aversion=7,
        log_equity_premium=0.02,
        log_risk_free=0.02,
    )
    assert H == pytest.approx(924_805, rel=1e-3)


def test_merton_share_matches_paper_worked_example():
    # Same scenario: alpha* = 15.5% per the paper.
    alpha_star = merton_equity_share(
        risk_aversion=7, log_equity_premium=0.02, sigma_log_equity=0.185
    )
    assert alpha_star == pytest.approx(0.155, abs=1e-3)


def test_full_worked_example_equity_share():
    # alpha* * (1 + H/W) with W = $1,000,000 should land at ~30%, matching
    # the paper (their true numerical optimum for this case is 33%).
    alpha_star = merton_equity_share(7, 0.02, 0.185)
    H = human_capital(55, 100_000, 7, 0.02, 0.02)
    equity_share = min(max(alpha_star * (1 + H / 1_000_000), 0.0), 1.0)
    assert equity_share == pytest.approx(0.30, abs=0.01)


def test_higher_wealth_relative_to_income_lowers_equity_share():
    from portfolio_engine.data import get_market_data

    market_data = get_market_data()
    low_wealth = research_equity_pct(45, 100_000, 50_000, RiskTolerance.MODERATE, market_data)
    high_wealth = research_equity_pct(45, 100_000, 5_000_000, RiskTolerance.MODERATE, market_data)
    assert low_wealth > high_wealth


def test_higher_income_relative_to_wealth_raises_equity_share():
    from portfolio_engine.data import get_market_data

    market_data = get_market_data()
    low_income = research_equity_pct(45, 20_000, 500_000, RiskTolerance.MODERATE, market_data)
    high_income = research_equity_pct(45, 300_000, 500_000, RiskTolerance.MODERATE, market_data)
    assert high_income > low_income


def test_higher_risk_aversion_lowers_equity_share():
    from portfolio_engine.data import get_market_data

    market_data = get_market_data()
    conservative = research_equity_pct(45, 100_000, 500_000, RiskTolerance.CONSERVATIVE, market_data)
    aggressive = research_equity_pct(45, 100_000, 500_000, RiskTolerance.AGGRESSIVE, market_data)
    assert aggressive > conservative


def test_equity_share_bounded_between_zero_and_one():
    from portfolio_engine.data import get_market_data

    market_data = get_market_data()
    tiny_wealth_share = research_equity_pct(25, 200_000, 100, RiskTolerance.AGGRESSIVE, market_data)
    huge_wealth_share = research_equity_pct(80, 20_000, 50_000_000, RiskTolerance.CONSERVATIVE, market_data)
    assert 0.0 <= tiny_wealth_share <= 1.0
    assert 0.0 <= huge_wealth_share <= 1.0


def test_retiree_human_capital_declines_with_age():
    # Holding the (already-retired) benefit level fixed, human capital
    # should fall as fewer years of remaining benefit payments are left.
    H_70 = human_capital(70, 50_000, 7, 0.02, 0.02)
    H_90 = human_capital(90, 50_000, 7, 0.02, 0.02)
    assert H_70 > H_90 > 0


def test_allocation_weights_sum_to_one_and_are_non_negative():
    from portfolio_engine.data import get_market_data

    market_data = get_market_data()
    weights = research_informed_allocation(45, 150_000, 500_000, RiskTolerance.AGGRESSIVE, market_data)
    assert set(weights.index) == set(ASSET_KEYS)
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0).all()


def test_invalid_inputs_raise():
    from portfolio_engine.data import get_market_data

    market_data = get_market_data()
    with pytest.raises(ValueError):
        research_informed_allocation(10, 50_000, 100_000, RiskTolerance.MODERATE, market_data)
    with pytest.raises(ValueError):
        research_informed_allocation(150, 50_000, 100_000, RiskTolerance.MODERATE, market_data)
    with pytest.raises(ValueError):
        research_informed_allocation(40, -1, 100_000, RiskTolerance.MODERATE, market_data)
    with pytest.raises(ValueError):
        research_informed_allocation(40, 50_000, -1, RiskTolerance.MODERATE, market_data)
