"""Maps a client's risk tolerance and investment horizon to a mean-variance
risk-aversion coefficient, so mean-variance optimization can pick a
client-appropriate point on the efficient frontier instead of always
returning the same portfolio for every client.

Calibrated against the baseline asset universe / fallback assumptions: a
moderate investor with a 20-year horizon lands close to where the old
fixed max-Sharpe portfolio used to sit, so switching this on doesn't
whiplash the "typical" case, while conservative/short-horizon and
aggressive/long-horizon clients land in clearly different places on the
frontier.
"""

from portfolio_engine.lifecycle import RiskTolerance

BASE_RISK_AVERSION: dict[RiskTolerance, float] = {
    RiskTolerance.CONSERVATIVE: 14.0,
    RiskTolerance.MODERATE: 8.0,
    RiskTolerance.AGGRESSIVE: 4.0,
}

MIN_RISK_AVERSION = 1.5
MAX_RISK_AVERSION = 25.0

# Linear in horizon: more risk-averse for a short horizon, less for a long
# one, anchored at 1.0 (no adjustment) at a 20-year horizon.
_HORIZON_SLOPE = 0.03
_HORIZON_ANCHOR_YEARS = 20
_HORIZON_MULTIPLIER_MIN = 0.5
_HORIZON_MULTIPLIER_MAX = 1.6


def _horizon_multiplier(horizon_years: float) -> float:
    raw = 1.0 + _HORIZON_SLOPE * (_HORIZON_ANCHOR_YEARS - horizon_years)
    return min(max(raw, _HORIZON_MULTIPLIER_MIN), _HORIZON_MULTIPLIER_MAX)


def risk_aversion_for_profile(
    risk_tolerance: RiskTolerance, horizon_years: float
) -> float:
    base = BASE_RISK_AVERSION[RiskTolerance(risk_tolerance)]
    raw = base * _horizon_multiplier(horizon_years)
    return min(max(raw, MIN_RISK_AVERSION), MAX_RISK_AVERSION)
