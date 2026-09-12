"""Research-informed allocation: a practical implementation of the
human-capital view of lifecycle investing from Duarte, Fonseca, Goodman &
Parker (2021) ("DFGP"), Choi, Liu & Liu (2025) ("CLL"), and the underlying
Cocco, Gomes & Maenhout (2005) ("CGM") optimal-control model, motivated by
Choi (2022)'s finding that popular heuristics like "100 minus age" diverge
from what this literature actually recommends.

WHY THIS IS AN "IMPLEMENTATION OF THE INSIGHT," NOT A LITERAL FORMULA
-----------------------------------------------------------------------
CLL's own contribution is a simple closed-form approximation,
alpha = clip(0, 1, alpha_star * (1 + H/W)) (their eq. 9), fit by
regression to CGM's full numerical solution. It is validated against
CGM's numerical model - but CLL fit and tested it only at conservative,
below-historical equity premia (2-4% log) specifically as a stress test
of the approximation's robustness, and DFGP's own richer numerical model
(deep reinforcement learning, not a regression fit) shows the plain
formula overshoots for the young - DFGP attribute the young's low actual
equity share (under 30% at 25, despite enormous human capital) to
liquidity constraints and borrowing limits the simple formula doesn't
capture, not to low human capital. In other words: CLL's formula is a
faithful map of CGM's solution *inside* CGM's own parameterization, but
naively plugging in a household's numbers under a different (here, more
conservative) capital-market assumption set can push the formula's H/W
term into territory the two papers don't actually claim it handles well.

So instead of transcribing alpha_star * (1 + H/W) as the final answer,
this module uses DFGP's own *reported* age-conditional equity shares -
their real, published numbers - as the anchor, and uses the human-capital
machinery (still CLL's discount-rate approximation, unchanged) only as a
bounded adjustment on top: how much a given client's own pension-like
income and wealth should move them above or below that empirical anchor,
not to replace it outright. Risk tolerance is applied once, as a bounded
personal-preference tilt (Choi 2022's "willingness to take risk"),
because it is one of the app's collected inputs and DFGP's model has no
free preference parameter beyond risk aversion (which we hold fixed for
computing H - see below).

1. AGE ANCHOR (Duarte et al., 2021, Section 4 and its footnote 2):
   "the share of financial wealth that a household should hold in stocks
   is hump-shaped over the working life, peaking around age 45 at 80%
   and declining to a stable 60% at and during retirement" and "the
   average optimal share in equity declines linearly to about 60% at
   retirement, after which it is roughly constant." Those four numbers
   (25% at the start of working life [DFGP: "below 30%"], 80% at 45,
   60% at the retirement threshold, 60% flat thereafter) are DFGP's own
   reported figures, not fit by us; DUARTE_ANCHOR_AGES/_SHARES below is a
   piecewise-linear read of that description (linear between 45 and 66,
   per DFGP's own word "linearly"), not a numerical solution we re-ran.

2. HUMAN-CAPITAL ADJUSTMENT (bounded, not multiplicative): human_capital()
   below is unchanged CLL machinery - the same regression-based discount
   rates (their Tables 1-2), validated against their own worked example
   to the dollar (see tests/test_research.py). Rather than feeding H
   straight into an unbounded alpha*(1+H/W), we use it as a capped nudge,
   HUMAN_CAPITAL_ADJUSTMENT_CAP * tanh(H/W), around the DFGP anchor. Two
   reasons for a bounded, saturating shape instead of the raw ratio:
   (a) DFGP's own cross-sectional finding is that WEALTHIER households
   hold a MORE steeply declining, LOWER late-life equity share than
   poorer ones at the same age (their Section 4.2 / Figure VI.a: "23
   percentage points more [equity]... 26 percentage points less" between
   the top and bottom total-financial-wealth quartiles at age 65) - i.e.
   more relative pension/human-capital coverage (equivalently, lower
   wealth for a given income) pushes equity UP, consistent in direction
   with H/W, giving us a real, cited empirical magnitude to calibrate
   against; (b) an unbounded H/W blows past what DFGP's own frictioned
   model would recommend for extreme cases (very young, or very wealthy
   relative to income) - exactly the liquidity-constraint gap noted
   above. The cap is set to 12 percentage points, roughly half of DFGP's
   ~23-26 point top-vs-bottom-quartile spread, since we are nudging a
   single client away from one anchor rather than differentiating four
   quartiles; this specific number is our own calibration choice, not a
   figure printed in either paper, and is documented here rather than
   left implicit.

3. RISK-TOLERANCE TILT: a fixed +/-7.5 percentage point shift for
   aggressive/conservative clients (moderate: no shift), smaller than the
   age and human-capital effects since those are the ones DFGP and CLL
   actually document; risk aversion itself is held fixed at CLL's tested
   midpoint (gamma=7) when computing H, so a client's stated risk
   tolerance affects the result exactly once, not doubled through both
   the discount-rate calculation and this tilt.

Deliberate simplifications versus the papers (documented, not hidden):
- CGM/CLL fit a cubic, education-specific deterministic age-earnings
  profile from PSID data. This app only collects a single "current annual
  income" figure, so future expected income is modeled as flat in real
  terms (no assumed raises) rather than following that fitted curve. This
  is not an ad hoc shortcut: it is exactly the scenario CLL use in their
  own worked numerical example (Section 3.3: constant $100,000 income
  through the last working year, then a fixed retirement benefit) - this
  module's human-capital calculation reproduces that example to the
  dollar (see tests/test_research.py).
- Income-shock variances (sigma_v, sigma_eps) and the Social Security
  replacement rate (lambda) are fixed at CLL's "college graduate" /
  baseline calibration rather than collected from the client, since this
  app does not ask about education or expected retirement benefits.
- Retirement age is fixed at CGM/CLL's 66 (benefits begin at 67, the
  current Social Security full retirement age) - but per the anchor
  above, a retiree's equity share does NOT keep declining after this
  point the way the baseline lifecycle heuristic's age rule does; it
  holds near 60%, exactly DFGP's point that retirees still draw a
  long-horizon, pension-like income stream that keeps supporting risk-
  taking rather than a naive "years until you need the money" framing.
- The "risk-free asset" and "the stock market" in CLL's binary risky/safe
  framing are proxied here by this engine's existing fixed-income sleeve
  (US bonds + TIPS) and equity sleeve (US/international/EM stocks +
  REITs) respectively, blended using the same weights the baseline
  lifecycle heuristic already uses (EQUITY_SLEEVE_SPLIT /
  FIXED_INCOME_SLEEVE_SPLIT) - the papers only model two assets, so
  splitting each sleeve into the app's six asset classes is this app's
  own extension, not something the papers specify.
- This method's equity share depends on age, income, wealth, and risk
  tolerance, NOT on the app's generic "investment horizon" slider, which
  represents a goal timeline (e.g. saving for a house in 5 years) rather
  than years-to-retirement. Horizon is intentionally not an input here,
  unlike the baseline lifecycle heuristic.
"""

import math

import numpy as np
import pandas as pd

from portfolio_engine.assets import (
    ASSET_KEYS,
    EQUITY_SLEEVE_SPLIT,
    FIXED_INCOME_SLEEVE_SPLIT,
)
from portfolio_engine.data import MarketData
from portfolio_engine.lifecycle import RiskTolerance
from portfolio_engine.metrics import portfolio_return, portfolio_volatility

# CGM/CLL lifecycle parameterization.
LAST_WORKING_AGE = 66  # retirement benefits begin at 67
MAX_AGE = 100
RETIREMENT_REPLACEMENT_RATE = 0.40  # CLL's baseline example / lowest tested value

# CLL's "college graduate" income-risk calibration (their Table of
# parameters), used as a single default since this app does not collect
# education level.
SIGMA_PERMANENT_INCOME_SHOCK = 0.130
SIGMA_TRANSITORY_INCOME_SHOCK = 0.242

# Risk aversion held fixed at CLL's tested midpoint (their range is 4-10)
# when computing human capital, so a client's risk tolerance affects the
# result exactly once - via RISK_TOLERANCE_TILT below - not twice.
REFERENCE_RISK_AVERSION = 7.0

# Duarte, Fonseca, Goodman & Parker (2021), Section 4 and footnote 2: the
# equity share of financial wealth starts below 30% at the beginning of
# working life (age 25), rises to a peak of 80% around age 45, "declines
# linearly to about 60% at retirement" (age 66), "after which it is
# roughly constant." These four points are DFGP's own reported figures.
DUARTE_ANCHOR_AGES = [25, 45, LAST_WORKING_AGE, MAX_AGE]
DUARTE_ANCHOR_SHARES = [0.25, 0.80, 0.60, 0.60]

# Cap on the human-capital adjustment applied around the Duarte anchor:
# roughly half of DFGP's own reported ~23-26 percentage point spread
# between the top and bottom total-financial-wealth quartiles at a given
# age (Section 4.2 / Figure VI.a) - halved because we are nudging one
# client away from a single anchor, not differentiating four quartiles.
# This specific number is our own calibration choice (documented, not a
# figure printed in either paper).
HUMAN_CAPITAL_ADJUSTMENT_CAP = 0.12

# Bounded personal-preference tilt (Choi 2022's "willingness to take
# risk"), smaller than the age/human-capital effects since those are the
# ones DFGP and CLL actually document.
RISK_TOLERANCE_TILT: dict[RiskTolerance, float] = {
    RiskTolerance.CONSERVATIVE: -0.075,
    RiskTolerance.MODERATE: 0.0,
    RiskTolerance.AGGRESSIVE: 0.075,
}

# CLL Table 1, column 3: one-year-ahead discount rate for labor income
# when next period is still in working life.
_WORK_COEF_RISK_AVERSION = 0.087  # on risk_aversion / 10
_WORK_COEF_LOG_EQUITY_PREMIUM = -0.267
_WORK_COEF_LOG_RISK_FREE = 1.132
_WORK_COEF_SIGMA_PERMANENT_SQ = 4.332
_WORK_COEF_SIGMA_TRANSITORY_SQ = 0.028
_WORK_COEF_REPLACEMENT_RATE = 0.010
_WORK_COEF_AGE = -0.149  # on age / 100
_WORK_COEF_AGE_SQ = 0.142  # on (age / 100)^2
_WORK_INTERCEPT = -0.020

# CLL Table 2, column 3: one-year-ahead discount rate when next period is
# in retirement (income is the risk-free retirement benefit).
_RET_COEF_RISK_AVERSION = 0.0003
_RET_COEF_LOG_EQUITY_PREMIUM = -0.217
_RET_COEF_LOG_RISK_FREE = 0.893
_RET_COEF_AGE = 0.476
_RET_COEF_AGE_SQ = -0.295
_RET_INTERCEPT = -0.166


def _working_life_discount_rate(
    age: float,
    risk_aversion: float,
    log_equity_premium: float,
    log_risk_free: float,
) -> float:
    """CLL equation from Table 1 (col. 3): one-year-ahead discount rate
    applied by the age-`age` self to age-(age+1) income, when that income
    is still working-life labor income."""
    return (
        _WORK_COEF_RISK_AVERSION * (risk_aversion / 10)
        + _WORK_COEF_LOG_EQUITY_PREMIUM * log_equity_premium
        + _WORK_COEF_LOG_RISK_FREE * log_risk_free
        + _WORK_COEF_SIGMA_PERMANENT_SQ * SIGMA_PERMANENT_INCOME_SHOCK**2
        + _WORK_COEF_SIGMA_TRANSITORY_SQ * SIGMA_TRANSITORY_INCOME_SHOCK**2
        + _WORK_COEF_REPLACEMENT_RATE * RETIREMENT_REPLACEMENT_RATE
        + _WORK_COEF_AGE * (age / 100)
        + _WORK_COEF_AGE_SQ * (age / 100) ** 2
        + _WORK_INTERCEPT
    )


def _retirement_discount_rate(
    age: float,
    risk_aversion: float,
    log_equity_premium: float,
    log_risk_free: float,
) -> float:
    """CLL equation from Table 2 (col. 3): one-year-ahead discount rate
    applied by the age-`age` self to age-(age+1) income, when that income
    is the (risk-free) retirement benefit."""
    return (
        _RET_COEF_RISK_AVERSION * (risk_aversion / 10)
        + _RET_COEF_LOG_EQUITY_PREMIUM * log_equity_premium
        + _RET_COEF_LOG_RISK_FREE * log_risk_free
        + _RET_COEF_AGE * (age / 100)
        + _RET_COEF_AGE_SQ * (age / 100) ** 2
        + _RET_INTERCEPT
    )


def human_capital(
    age: int,
    annual_income: float,
    risk_aversion: float,
    log_equity_premium: float,
    log_risk_free: float,
) -> float:
    """Present value of expected future income (CLL equations 1, 17-18),
    under a flat-real-income simplification (see module docstring).

    `annual_income` is taken to already BE the expected future income at
    each working-life age (CLL's own worked example does the same: a flat
    $100,000 expected income, with no separate convexity adjustment layered
    on top - see tests/test_research.py). If `age` is at or before the
    last working age, `annual_income` is the current/expected working-life
    salary, replaced at retirement by RETIREMENT_REPLACEMENT_RATE of that
    value. If `age` is already past the last working age, `annual_income`
    is the current retirement benefit itself, held flat.
    """
    if age >= MAX_AGE:
        return 0.0

    already_retired = age > LAST_WORKING_AGE
    human_capital_value = 0.0
    cumulative_discount = 1.0

    for future_age in range(age + 1, MAX_AGE + 1):
        prior_age = future_age - 1
        still_working_income = future_age <= LAST_WORKING_AGE

        if still_working_income:
            rate = _working_life_discount_rate(
                prior_age, risk_aversion, log_equity_premium, log_risk_free
            )
        else:
            rate = _retirement_discount_rate(
                prior_age, risk_aversion, log_equity_premium, log_risk_free
            )
        cumulative_discount *= 1 + rate

        if already_retired or still_working_income:
            expected_income = annual_income
        else:
            expected_income = RETIREMENT_REPLACEMENT_RATE * annual_income

        human_capital_value += expected_income / cumulative_discount

    return human_capital_value


def _log_return_params(arithmetic_return: float, arithmetic_volatility: float) -> tuple[float, float]:
    """Convert (arithmetic mean, arithmetic std dev) of a simple return to
    (mean, variance) of its log return, assuming a lognormal gross return."""
    log_variance = math.log(1 + (arithmetic_volatility / (1 + arithmetic_return)) ** 2)
    log_mean = math.log(1 + arithmetic_return) - 0.5 * log_variance
    return log_mean, log_variance


def _sleeve_arithmetic_stats(
    market_data: MarketData, sleeve_weights: dict[str, float]
) -> tuple[float, float]:
    weights = pd.Series(0.0, index=ASSET_KEYS)
    for key, weight in sleeve_weights.items():
        weights[key] = weight
    ret = portfolio_return(weights, market_data.expected_returns)
    vol = portfolio_volatility(weights, market_data.cov_matrix)
    return ret, vol


def merton_equity_share(
    risk_aversion: float, log_equity_premium: float, sigma_log_equity: float
) -> float:
    """CLL equation 8: the Merton (1969) optimal equity share with no
    labor income, subject to no-leverage / no-short-sale constraints."""
    if sigma_log_equity <= 0:
        return 0.0
    raw = (log_equity_premium + 0.5 * sigma_log_equity**2) / (
        risk_aversion * sigma_log_equity**2
    )
    return min(max(raw, 0.0), 1.0)


def duarte_age_anchor(age: float) -> float:
    """DFGP (2021)'s own reported age-conditional equity share (see module
    docstring): a piecewise-linear read of their described curve, clamped
    to their studied working-life start age of 25 at the low end."""
    clamped_age = min(max(age, DUARTE_ANCHOR_AGES[0]), DUARTE_ANCHOR_AGES[-1])
    return float(np.interp(clamped_age, DUARTE_ANCHOR_AGES, DUARTE_ANCHOR_SHARES))


def human_capital_adjustment(human_capital_value: float, wealth: float) -> float:
    """Bounded nudge around the Duarte anchor from a client's own
    human-capital-to-wealth ratio (see module docstring for the
    calibration of the cap and the choice of a saturating tanh shape)."""
    ratio = human_capital_value / max(wealth, 1e-9)
    return HUMAN_CAPITAL_ADJUSTMENT_CAP * math.tanh(ratio)


def research_equity_pct(
    age: int,
    annual_income: float,
    initial_investment: float,
    risk_tolerance: RiskTolerance,
    market_data: MarketData,
) -> float:
    """The equity share for one client: Duarte et al.'s (2021) own
    reported age-conditional share, adjusted for this client's human
    capital (via CLL's discount-rate approximation) and risk tolerance."""
    equity_return, equity_vol = _sleeve_arithmetic_stats(market_data, EQUITY_SLEEVE_SPLIT)
    fixed_income_return, _ = _sleeve_arithmetic_stats(market_data, FIXED_INCOME_SLEEVE_SPLIT)

    log_equity_mean, log_equity_var = _log_return_params(equity_return, equity_vol)
    log_risk_free = math.log(1 + fixed_income_return)
    log_equity_premium = log_equity_mean - log_risk_free

    wealth = max(initial_investment, 1e-9)
    H = human_capital(
        age, annual_income, REFERENCE_RISK_AVERSION, log_equity_premium, log_risk_free
    )

    anchor = duarte_age_anchor(age)
    adjustment = human_capital_adjustment(H, wealth)
    tilt = RISK_TOLERANCE_TILT[RiskTolerance(risk_tolerance)]

    return min(max(anchor + adjustment + tilt, 0.0), 1.0)


def research_informed_allocation(
    age: int,
    annual_income: float,
    initial_investment: float,
    risk_tolerance: RiskTolerance,
    market_data: MarketData,
) -> pd.Series:
    """Full asset-class weight vector for the research-informed method."""
    if age < 18 or age > 100:
        raise ValueError("age must be between 18 and 100")
    if annual_income < 0:
        raise ValueError("annual_income must be non-negative")
    if initial_investment < 0:
        raise ValueError("initial_investment must be non-negative")

    equity_pct = research_equity_pct(
        age, annual_income, initial_investment, risk_tolerance, market_data
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
