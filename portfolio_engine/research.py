"""Research-informed allocation: the human-capital-adjusted equity share
from Choi, Liu & Liu (2025), "Practical Finance: An Approximate Solution to
Lifecycle Portfolio Choice" ("CLL"), grounded in the lifecycle model of
Cocco, Gomes & Maenhout (2005) ("CGM") and the closely related simple rule
of Duarte, Fonseca, Goodman & Parker (2021) ("DFGP"). Choi (2022) surveys
popular financial advice (e.g. "100 minus age") and shows it diverges from
what this literature actually recommends, which is the motivation for
building this method instead of another hand-tuned age curve.

The core idea (Merton, 1969; Bodie, Merton & Samuelson, 1992; CGM; DFGP;
CLL): a client's risk capacity comes from total wealth - financial wealth
W plus human capital H, the present value of expected future income. The
equity share of financial wealth should be

    alpha = clip(0, 1, alpha_star * (1 + H / W))

where alpha_star is the Merton (1969) optimal equity share for an investor
with NO labor income (equation 8 in CLL):

    alpha_star = clip(0, 1, (log_equity_premium + 0.5 * sigma_S^2)
                              / (risk_aversion * sigma_S^2))

CLL's contribution is a practical, regression-based approximation for the
discount rate used to compute H under CGM's realistic lifecycle model
(risky, non-tradable labor income; no borrowing/short-selling), fit across
5,103 numerically-solved parameter combinations. Their approximation
matches the true CGM solution with R^2 = 0.99 and an average welfare loss
of only 0.06% of consumption, versus 2.00% for "100 minus age" and 3.75%
for a constant 60% equity share (CLL Table/Section 4) - i.e. it is not a
hand-tuned curve but a validated approximation to a real optimal-control
solution.

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
  current Social Security full retirement age).
- The "risk-free asset" and "the stock market" in CLL's binary risky/safe
  framing are proxied here by this engine's existing fixed-income sleeve
  (US bonds + TIPS) and equity sleeve (US/international/EM stocks +
  REITs) respectively, blended using the same weights the baseline
  lifecycle heuristic already uses (EQUITY_SLEEVE_SPLIT /
  FIXED_INCOME_SLEEVE_SPLIT) - the papers only model two assets, so
  splitting each sleeve into the app's six asset classes is this app's
  own extension, not something the papers specify.
- This method's equity share depends on age, income, wealth, and risk
  tolerance (via risk aversion) - per CLL, NOT on the app's generic
  "investment horizon" slider, which represents a goal timeline (e.g.
  saving for a house in 5 years) rather than years-to-retirement. Horizon
  is intentionally not an input here, unlike the baseline lifecycle
  heuristic.
"""

import math

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

# Relative risk aversion by risk tolerance, drawn from CLL's own tested
# range (4 to 10, with 10 "commonly regarded as the upper limit of
# reasonable risk aversion").
RISK_AVERSION_BY_TOLERANCE: dict[RiskTolerance, float] = {
    RiskTolerance.CONSERVATIVE: 9.0,
    RiskTolerance.MODERATE: 7.0,
    RiskTolerance.AGGRESSIVE: 4.0,
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


def research_equity_pct(
    age: int,
    annual_income: float,
    initial_investment: float,
    risk_tolerance: RiskTolerance,
    market_data: MarketData,
) -> float:
    """The CLL human-capital-adjusted equity share for one client."""
    risk_aversion = RISK_AVERSION_BY_TOLERANCE[RiskTolerance(risk_tolerance)]

    equity_return, equity_vol = _sleeve_arithmetic_stats(market_data, EQUITY_SLEEVE_SPLIT)
    fixed_income_return, _ = _sleeve_arithmetic_stats(market_data, FIXED_INCOME_SLEEVE_SPLIT)

    log_equity_mean, log_equity_var = _log_return_params(equity_return, equity_vol)
    log_risk_free = math.log(1 + fixed_income_return)
    log_equity_premium = log_equity_mean - log_risk_free
    sigma_log_equity = math.sqrt(log_equity_var)

    alpha_star = merton_equity_share(risk_aversion, log_equity_premium, sigma_log_equity)

    wealth = max(initial_investment, 1e-9)
    H = human_capital(age, annual_income, risk_aversion, log_equity_premium, log_risk_free)

    return min(max(alpha_star * (1 + H / wealth), 0.0), 1.0)


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
