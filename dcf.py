"""Core DCF valuation engine — kept separate from the Gradio UI (app.py).

Forecast structure (10 years total):
  Years 1-5:  explicit per-year revenue growth and operating margin inputs.
  Years 6-10: revenue growth fades linearly from the long-term rate down to
              the terminal growth rate, landing exactly on the terminal rate
              at Year 10 (so there is no discontinuity when the model hands
              off to the terminal-value formula). Operating margin holds
              flat at the Year 5 level.
  Year 10+:   Gordon Growth terminal value at the terminal growth rate.

Free cash flow (full formula, unlevered):
  FCF = NOPAT + D&A - CapEx - change in net working capital
  where change in NWC = nwc_pct_of_revenue_increase * (revenue_t - revenue_(t-1))
  (applied to the year-over-year dollar increase in revenue, not revenue itself).
"""

from dataclasses import dataclass

FORECAST_YEARS = 10
EXPLICIT_YEARS = 5  # Years 1-5 use the per-year growth/margin inputs directly

# D&A as a % of revenue is not a Part 1 input control; it defaults to the
# case's own FY2026 actual (Exhibit 5: D&A/revenue ~1.3%) so the FCF formula
# isn't silently missing the depreciation add-back.
DEFAULT_DA_PCT_OF_REVENUE = 0.013


@dataclass
class YearProjection:
    year: int
    growth_rate: float
    revenue: float
    operating_margin: float
    nopat: float
    depreciation_and_amortization: float
    capex: float
    change_in_nwc: float
    free_cash_flow: float
    present_value: float


@dataclass
class DCFResult:
    years: list[YearProjection]
    wacc: float
    pv_of_forecast_years: float
    terminal_value: float
    pv_of_terminal_value: float
    enterprise_value: float
    equity_value: float
    value_per_share: float


def compute_wacc(risk_free_rate: float, equity_risk_premium: float, beta: float) -> float:
    """WACC approximated as cost of equity (CAPM), since NVIDIA's debt load
    is small enough (~4-5% debt/equity per the case) that blending in a
    cost of debt would barely move the result."""
    return risk_free_rate + beta * equity_risk_premium


def _growth_schedule(
    growth_y1: float, growth_y2: float, growth_y3: float, growth_y4: float, growth_y5: float,
    long_term_growth: float, terminal_growth: float,
) -> dict[int, float]:
    schedule = {1: growth_y1, 2: growth_y2, 3: growth_y3, 4: growth_y4, 5: growth_y5}
    fade_years = FORECAST_YEARS - EXPLICIT_YEARS  # Years 6-10
    for i in range(fade_years):
        year = EXPLICIT_YEARS + 1 + i
        frac = i / (fade_years - 1) if fade_years > 1 else 1.0
        schedule[year] = long_term_growth + (terminal_growth - long_term_growth) * frac
    return schedule


def _margin_schedule(
    margin_y1: float, margin_y2: float, margin_y3: float, margin_y4: float, margin_y5: float,
) -> dict[int, float]:
    schedule = {1: margin_y1, 2: margin_y2, 3: margin_y3, 4: margin_y4, 5: margin_y5}
    for year in range(EXPLICIT_YEARS + 1, FORECAST_YEARS + 1):
        schedule[year] = margin_y5  # held flat beyond Year 5
    return schedule


def run_dcf(
    base_revenue: float,
    revenue_growth_y1: float, revenue_growth_y2: float, revenue_growth_y3: float,
    revenue_growth_y4: float, revenue_growth_y5: float,
    revenue_growth_long_term: float, terminal_growth_rate: float,
    operating_margin_y1: float, operating_margin_y2: float, operating_margin_y3: float,
    operating_margin_y4: float, operating_margin_y5: float,
    tax_rate: float,
    capex_pct_of_revenue: float,
    nwc_pct_of_revenue_increase: float,
    risk_free_rate: float, equity_risk_premium: float, beta: float,
    cash_and_securities: float, nonmarketable_securities: float,
    total_debt: float, shares_outstanding: float,
    da_pct_of_revenue: float = DEFAULT_DA_PCT_OF_REVENUE,
) -> DCFResult:
    wacc = compute_wacc(risk_free_rate, equity_risk_premium, beta)
    if terminal_growth_rate >= wacc:
        raise ValueError(
            f"Terminal growth rate ({terminal_growth_rate:.4f}) must be below WACC ({wacc:.4f})."
        )

    growth = _growth_schedule(
        revenue_growth_y1, revenue_growth_y2, revenue_growth_y3, revenue_growth_y4, revenue_growth_y5,
        revenue_growth_long_term, terminal_growth_rate,
    )
    margin = _margin_schedule(
        operating_margin_y1, operating_margin_y2, operating_margin_y3, operating_margin_y4, operating_margin_y5,
    )

    revenue = {0: base_revenue}
    years = []
    pv_of_forecast_years = 0.0

    for year in range(1, FORECAST_YEARS + 1):
        revenue[year] = revenue[year - 1] * (1 + growth[year])
        nopat = revenue[year] * margin[year] * (1 - tax_rate)
        da = revenue[year] * da_pct_of_revenue
        capex = revenue[year] * capex_pct_of_revenue
        change_in_nwc = nwc_pct_of_revenue_increase * (revenue[year] - revenue[year - 1])
        free_cash_flow = nopat + da - capex - change_in_nwc
        present_value = free_cash_flow / (1 + wacc) ** year

        pv_of_forecast_years += present_value
        years.append(YearProjection(
            year=year,
            growth_rate=growth[year],
            revenue=revenue[year],
            operating_margin=margin[year],
            nopat=nopat,
            depreciation_and_amortization=da,
            capex=capex,
            change_in_nwc=change_in_nwc,
            free_cash_flow=free_cash_flow,
            present_value=present_value,
        ))

    last_fcf = years[-1].free_cash_flow
    terminal_value = last_fcf * (1 + terminal_growth_rate) / (wacc - terminal_growth_rate)
    pv_of_terminal_value = terminal_value / (1 + wacc) ** FORECAST_YEARS

    enterprise_value = pv_of_forecast_years + pv_of_terminal_value
    equity_value = enterprise_value + cash_and_securities + nonmarketable_securities - total_debt
    value_per_share = equity_value / shares_outstanding

    return DCFResult(
        years=years,
        wacc=wacc,
        pv_of_forecast_years=pv_of_forecast_years,
        terminal_value=terminal_value,
        pv_of_terminal_value=pv_of_terminal_value,
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        value_per_share=value_per_share,
    )
