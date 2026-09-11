"""Core DCF valuation math, kept separate from the Gradio UI."""

from dataclasses import dataclass


@dataclass
class YearProjection:
    year: int
    growth_rate: float
    revenue: float
    operating_income: float
    nopat: float
    free_cash_flow: float
    present_value: float


@dataclass
class DCFResult:
    years: list[YearProjection]
    pv_of_forecast_years: float
    terminal_value: float
    pv_of_terminal_value: float
    enterprise_value: float
    equity_value: float
    value_per_share: float


def _growth_schedule(growth_y1: float, growth_y5: float, growth_terminal: float, horizon: int) -> list[float]:
    """Year-by-year growth rate: tapers from growth_y1 (year 1) to growth_y5
    (year 5), then keeps tapering from growth_y5 toward growth_terminal by
    the end of the forecast horizon."""
    rates = []
    taper_years = min(5, horizon)
    for year in range(1, horizon + 1):
        if year <= taper_years:
            frac = (year - 1) / (taper_years - 1) if taper_years > 1 else 1.0
            rates.append(growth_y1 + (growth_y5 - growth_y1) * frac)
        else:
            frac = (year - taper_years) / (horizon - taper_years)
            rates.append(growth_y5 + (growth_terminal - growth_y5) * frac)
    return rates


def run_dcf(
    revenue: float,
    growth_y1: float,
    growth_y5: float,
    growth_terminal: float,
    gross_margin: float,
    opex_pct_of_revenue: float,
    tax_rate: float,
    wacc: float,
    horizon: int,
    cash: float,
    debt: float,
    shares_outstanding: float,
) -> DCFResult:
    growth_rates = _growth_schedule(growth_y1, growth_y5, growth_terminal, horizon)

    years = []
    pv_of_forecast_years = 0.0
    projected_revenue = revenue
    last_fcf = 0.0

    for year, growth_rate in enumerate(growth_rates, start=1):
        projected_revenue *= 1 + growth_rate
        operating_income = projected_revenue * (gross_margin - opex_pct_of_revenue)
        nopat = operating_income * (1 - tax_rate)
        free_cash_flow = nopat * 0.90
        present_value = free_cash_flow / (1 + wacc) ** year

        pv_of_forecast_years += present_value
        last_fcf = free_cash_flow
        years.append(YearProjection(
            year=year,
            growth_rate=growth_rate,
            revenue=projected_revenue,
            operating_income=operating_income,
            nopat=nopat,
            free_cash_flow=free_cash_flow,
            present_value=present_value,
        ))

    terminal_value = last_fcf * (1 + growth_terminal) / (wacc - growth_terminal)
    pv_of_terminal_value = terminal_value / (1 + wacc) ** horizon

    enterprise_value = pv_of_forecast_years + pv_of_terminal_value
    equity_value = enterprise_value + cash - debt
    value_per_share = equity_value / shares_outstanding

    return DCFResult(
        years=years,
        pv_of_forecast_years=pv_of_forecast_years,
        terminal_value=terminal_value,
        pv_of_terminal_value=pv_of_terminal_value,
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        value_per_share=value_per_share,
    )
