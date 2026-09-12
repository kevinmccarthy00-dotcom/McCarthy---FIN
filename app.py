import os

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from dcf import run_dcf

# =============================================================================
# PART 1 — INPUT CONTROLS  /  PART 2 — DCF ENGINE  /  PART 3 — CHARTS
#
# Source for every "Case" figure below: "NVIDIA Corporation: Valuing the
# Engine of the AI Revolution" (Asaf Manela, Olin Business School, case
# written July 30, 2026), plus its accompanying exhibits workbook.
#
# The DCF math itself lives in dcf.py, kept fully separate from this UI file
# and unchanged by Part 3. Part 4 (further scenario analysis) is not built yet.
#
# Chart color language, used consistently across all three charts:
#   blue   = adds value / favorable (revenue, FCF, positive bridge items, undervalued)
#   orange = subtracts value / unfavorable (debt, overvalued)
#   gray   = totals / subtotals
# =============================================================================

CHART_BLUE = "#3B6FA0"
CHART_ORANGE = "#D98E42"
CHART_GRAY = "#4d4d4d"

# --- Starting financials (all case-sourced, Q1 FY2027 / July 29, 2026) -----
CASE_BASE_REVENUE = 215.938              # $B, FY2026 actual revenue (Exhibit 1)
CASE_CASH_AND_SECURITIES = 80.572        # $B, cash + marketable debt + marketable equity securities, Q1 FY27 (Exhibit 4)
CASE_NONMARKETABLE_SECURITIES = 43.364   # $B, private-company stakes, Q1 FY27 (Exhibit 4)
CASE_TOTAL_DEBT = 8.470                  # $B, Q1 FY27 (Exhibit 4)
CASE_SHARES_OUTSTANDING = 24.22          # billions, July 29, 2026 (Market Data)
CASE_SHARE_PRICE = 190.01                # $, close July 29, 2026 (Market Data)

# --- WACC components ---------------------------------------------------
CASE_RISK_FREE_RATE = 4.70               # %, 10-yr UST yield, July 30, 2026 (Exhibit 6A)
CASE_EQUITY_RISK_PREMIUM = 4.23          # %, Damodaran Jan-2026 edition (Exhibit 6A)
CASE_BETA_DEFAULT = 1.55                 # midpoint of case's bottom-up semiconductor-comp range 1.35-1.75 (Exhibit 6A) — a judgment call, see note below

# --- Tax, capex, working capital ----------------------------------------
CASE_TAX_RATE_DEFAULT = 17.0             # %, midpoint of FY2027 guidance 16-18% (Exhibit 2B)
CASE_CAPEX_PCT_DEFAULT = 2.8             # %, FY2026 actual capex/revenue (Exhibit 5)
CASE_NWC_PCT_DEFAULT = 19.3              # %, FY2026 "working capital absorption" (Exhibit 5)

# --- Revenue growth path --------------------------------------------------
CASE_YEAR1_GROWTH = 82.0                 # %, consensus FY2027 revenue $393.6B / FY2026 $215.9B - 1 (Exhibit 6A)
CASE_TERMINAL_GROWTH = 3.0               # %, the case's own "mature NVIDIA" illustration (Section 6.1)

# --- Operating margin path ------------------------------------------------
CASE_YEAR1_OPERATING_MARGIN = 65.6       # %, Q1 FY2027 actual, matching Q2 FY27 guidance-implied rate (Exhibit 2A/2B)


def _validation_banner(terminal_growth_rate, wacc):
    if terminal_growth_rate < wacc:
        return f"""
        <div style="padding:12px 16px;border-radius:8px;background:#1b4332;color:#eafbf0;border:1px solid #2f7a4d;">
        <b>Valid:</b> terminal growth rate ({terminal_growth_rate:.1f}%) is below WACC ({wacc:.2f}%). Terminal value math holds.
        </div>
        """
    return f"""
    <div style="padding:12px 16px;border-radius:8px;background:#5c1a1a;color:#ffe5e5;border:1px solid #a33;">
    <b>⚠ INVALID ASSUMPTION:</b> terminal growth rate ({terminal_growth_rate:.1f}%) must be
    <u>below</u> WACC ({wacc:.2f}%), or the terminal value formula divides by a negative or
    zero number and produces a meaningless (or negative) result. Lower the terminal growth
    rate or raise WACC (higher beta / risk-free rate / equity risk premium).
    </div>
    """


def _placeholder_figure(message):
    fig = go.Figure()
    fig.add_annotation(
        text=message, showarrow=False, font=dict(size=15, color="#888"),
        xref="paper", yref="paper", x=0.5, y=0.5,
    )
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=320, plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def build_revenue_fcf_chart(years):
    year_numbers = [y.year for y in years]
    revenue_values = [y.revenue for y in years]
    fcf_values = [y.free_cash_flow for y in years]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=year_numbers, y=revenue_values, mode="lines+markers", name="Revenue",
        line=dict(color=CHART_BLUE, width=3), marker=dict(size=7),
        hovertemplate="Year %{x}<br>Revenue: $%{y:,.0f}B<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=year_numbers, y=fcf_values, mode="lines+markers", name="Free Cash Flow",
        line=dict(color=CHART_ORANGE, width=3), marker=dict(size=7),
        hovertemplate="Year %{x}<br>FCF: $%{y:,.0f}B<extra></extra>",
    ))
    fig.update_layout(
        title="Revenue & Free Cash Flow — 10-Year Forecast",
        xaxis_title="Forecast Year", yaxis_title="$ Billions",
        xaxis=dict(dtick=1, gridcolor="rgba(120,120,120,0.15)"),
        yaxis=dict(gridcolor="rgba(120,120,120,0.15)"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=60, b=50),
        height=420, plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def build_waterfall_chart(result, cash_and_securities, nonmarketable_securities, total_debt):
    labels = [
        "PV of Forecast FCF<br>(Years 1-10)", "PV of Terminal Value", "Enterprise Value",
        "+ Cash & Marketable<br>Securities", "+ Non-Marketable<br>Securities", "− Total Debt",
        "Equity Value",
    ]
    measures = ["relative", "relative", "total", "relative", "relative", "relative", "total"]
    values = [
        result.pv_of_forecast_years, result.pv_of_terminal_value, result.enterprise_value,
        cash_and_securities, nonmarketable_securities, -total_debt, result.equity_value,
    ]

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measures,
        x=labels,
        y=values,
        text=[f"${v:,.0f}B" for v in values],
        textposition="outside",
        connector=dict(line=dict(color="rgba(120,120,120,0.4)", width=1)),
        increasing=dict(marker=dict(color=CHART_BLUE)),
        decreasing=dict(marker=dict(color=CHART_ORANGE)),
        totals=dict(marker=dict(color=CHART_GRAY)),
        hovertemplate="%{x}<br>$%{y:,.1f}B<extra></extra>",
    ))
    fig.update_layout(
        title="Valuation Waterfall — Enterprise Value to Equity Value",
        yaxis_title="$ Billions",
        yaxis=dict(gridcolor="rgba(120,120,120,0.15)"),
        showlegend=False,
        margin=dict(l=60, r=30, t=60, b=90),
        height=460, plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


# Sensitivity heatmap grid (percent). WACC min (8%) safely exceeds terminal
# growth max (5%) across the whole grid, so every cell is a valid DCF input.
SENSITIVITY_WACC_GRID = [8, 9, 10, 11, 12, 13, 14]
SENSITIVITY_TERMINAL_GRID = [1, 2, 3, 4, 5]


def build_sensitivity_heatmap(
    base_revenue,
    revenue_growth_y1, revenue_growth_y2, revenue_growth_y3, revenue_growth_y4, revenue_growth_y5,
    revenue_growth_long_term,
    operating_margin_y1, operating_margin_y2, operating_margin_y3, operating_margin_y4, operating_margin_y5,
    tax_rate, capex_pct_revenue, nwc_pct_revenue,
    equity_risk_premium, beta,
    cash_and_securities, nonmarketable_securities, total_debt, shares_outstanding,
    current_share_price,
):
    erp_decimal = equity_risk_premium / 100

    z = []
    for terminal_pct in SENSITIVITY_TERMINAL_GRID:
        row = []
        for wacc_pct in SENSITIVITY_WACC_GRID:
            wacc_decimal = wacc_pct / 100
            # Hold beta/ERP fixed at their current slider values and back out
            # the risk-free rate that produces this grid cell's target WACC,
            # so the sweep goes through the same public run_dcf() the rest of
            # the app uses (dcf.py's CAPM formula is untouched).
            risk_free_for_cell = wacc_decimal - beta * erp_decimal
            result = run_dcf(
                base_revenue=base_revenue,
                revenue_growth_y1=revenue_growth_y1 / 100, revenue_growth_y2=revenue_growth_y2 / 100,
                revenue_growth_y3=revenue_growth_y3 / 100, revenue_growth_y4=revenue_growth_y4 / 100,
                revenue_growth_y5=revenue_growth_y5 / 100,
                revenue_growth_long_term=revenue_growth_long_term / 100,
                terminal_growth_rate=terminal_pct / 100,
                operating_margin_y1=operating_margin_y1 / 100, operating_margin_y2=operating_margin_y2 / 100,
                operating_margin_y3=operating_margin_y3 / 100, operating_margin_y4=operating_margin_y4 / 100,
                operating_margin_y5=operating_margin_y5 / 100,
                tax_rate=tax_rate / 100,
                capex_pct_of_revenue=capex_pct_revenue / 100,
                nwc_pct_of_revenue_increase=nwc_pct_revenue / 100,
                risk_free_rate=risk_free_for_cell, equity_risk_premium=erp_decimal, beta=beta,
                cash_and_securities=cash_and_securities, nonmarketable_securities=nonmarketable_securities,
                total_debt=total_debt, shares_outstanding=shares_outstanding,
            )
            row.append(result.value_per_share)
        z.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[f"{w}%" for w in SENSITIVITY_WACC_GRID],
        y=[f"{g}%" for g in SENSITIVITY_TERMINAL_GRID],
        colorscale=[[0.0, "#B35806"], [0.5, "#F5F1EA"], [1.0, "#2166AC"]],
        zmid=current_share_price,
        text=[[f"${v:,.0f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont={"size": 12},
        hovertemplate="WACC %{x}<br>Terminal growth %{y}<br>Intrinsic value $%{z:,.2f}<extra></extra>",
        colorbar=dict(title="$/share"),
    ))
    fig.update_layout(
        title=f"Intrinsic Value per Share — WACC vs. Terminal Growth "
              f"(blue = above ${current_share_price:,.0f} current price, orange = below)",
        xaxis_title="WACC", yaxis_title="Terminal Growth Rate",
        height=440, margin=dict(l=60, r=40, t=70, b=50),
    )
    return fig


# =============================================================================
# PART 4 — BEAR / BASE / BULL SCENARIOS, PROBABILITY-WEIGHTED VALUE,
#          BUY/HOLD/SELL CALL, BREAK-EVEN
#
# Each scenario varies the assumptions that matter most (Year 1 & Year 5
# growth, Year 1 & Year 5 margin, terminal growth, WACC). Years 2-4 are no
# longer tied to the main model's shared sliders — each scenario linearly
# interpolates its own Years 2-4 between its own Year 1 and Year 5 values, so
# every scenario's path is internally consistent. Base revenue, long-term
# growth, tax, capex%, NWC%, and the balance sheet remain shared with the
# main model above. All three still run through the unmodified run_dcf()
# from dcf.py.
# =============================================================================

BUY_SELL_THRESHOLD_PCT = 10.0  # same rule of thumb used earlier: >10% mispricing triggers a call

# Base scenario defaults mirror the main model's defaults exactly, so its
# value reconciles to the same $172.27 baseline established in Parts 2-3.
_BASE_WACC_DEFAULT = round(CASE_RISK_FREE_RATE + CASE_BETA_DEFAULT * CASE_EQUITY_RISK_PREMIUM, 2)


def _interpolate_years_2_to_4(year1_value, year5_value):
    """Linearly interpolate Years 2-4 between a scenario's own Year 1 and
    Year 5 values (same 4-step interpolation used for the Year 6-10 fade)."""
    return [year1_value + (year5_value - year1_value) * fraction for fraction in (0.25, 0.50, 0.75)]


def _scenario_value_per_share(
    growth_y1, growth_y5, margin_y1, margin_y5, terminal_growth, wacc,
    base_revenue, revenue_growth_long_term,
    tax_rate, capex_pct_revenue, nwc_pct_revenue,
    equity_risk_premium, beta,
    cash_and_securities, nonmarketable_securities, total_debt, shares_outstanding,
):
    """Value per share for one scenario. Raises ValueError if terminal_growth >= wacc."""
    erp_decimal = equity_risk_premium / 100
    # Same trick as the Part 3 heatmap: hold beta/ERP fixed and back out the
    # risk-free rate that produces this scenario's target WACC, so this still
    # goes through dcf.py's unmodified CAPM formula.
    risk_free_for_scenario = (wacc / 100) - beta * erp_decimal

    growth_y2, growth_y3, growth_y4 = _interpolate_years_2_to_4(growth_y1, growth_y5)
    margin_y2, margin_y3, margin_y4 = _interpolate_years_2_to_4(margin_y1, margin_y5)

    result = run_dcf(
        base_revenue=base_revenue,
        revenue_growth_y1=growth_y1 / 100, revenue_growth_y2=growth_y2 / 100,
        revenue_growth_y3=growth_y3 / 100, revenue_growth_y4=growth_y4 / 100,
        revenue_growth_y5=growth_y5 / 100,
        revenue_growth_long_term=revenue_growth_long_term / 100,
        terminal_growth_rate=terminal_growth / 100,
        operating_margin_y1=margin_y1 / 100, operating_margin_y2=margin_y2 / 100,
        operating_margin_y3=margin_y3 / 100, operating_margin_y4=margin_y4 / 100,
        operating_margin_y5=margin_y5 / 100,
        tax_rate=tax_rate / 100,
        capex_pct_of_revenue=capex_pct_revenue / 100,
        nwc_pct_of_revenue_increase=nwc_pct_revenue / 100,
        risk_free_rate=risk_free_for_scenario, equity_risk_premium=erp_decimal, beta=beta,
        cash_and_securities=cash_and_securities, nonmarketable_securities=nonmarketable_securities,
        total_debt=total_debt, shares_outstanding=shares_outstanding,
    )
    return result.value_per_share


def _bisect_increasing(f, lo, hi, target, tol=1e-4, max_iter=60):
    """Find x in [lo, hi] such that f(x) == target, assuming f is monotonically
    increasing over that range. Returns None if lo/hi don't bracket target."""
    f_lo, f_hi = f(lo), f(hi)
    if (f_lo - target) * (f_hi - target) > 0:
        return None
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        f_mid = f(mid)
        if abs(f_mid - target) < tol:
            return mid
        if (f_lo - target) * (f_mid - target) <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


def _compute_breakeven(base_growth_y1, base_growth_y5, base_margin_y1, base_margin_y5,
                        base_terminal, base_wacc, shared_kwargs, current_share_price):
    """Holding all other Base-case assumptions fixed, solve for the single
    lever value that would make intrinsic value equal today's share price."""

    def value_with(growth_y1=base_growth_y1, growth_y5=base_growth_y5,
                   margin_y1=base_margin_y1, margin_y5=base_margin_y5,
                   terminal=base_terminal, wacc=base_wacc):
        return _scenario_value_per_share(growth_y1, growth_y5, margin_y1, margin_y5, terminal, wacc, **shared_kwargs)

    required_growth_y1 = _bisect_increasing(
        lambda g: value_with(growth_y1=g), -90.0, 500.0, current_share_price,
    )
    # Value falls as WACC rises, so negate both sides to reuse the
    # increasing-function bisector.
    required_wacc = _bisect_increasing(
        lambda w: -value_with(wacc=w), base_terminal + 0.01, 60.0, -current_share_price,
    )
    required_terminal = _bisect_increasing(
        lambda t: value_with(terminal=t), -5.0, base_wacc - 0.01, current_share_price,
    )
    return required_growth_y1, required_wacc, required_terminal


def _fmt_pct(value):
    return f"{value:.2f}%" if value is not None else "n/a (outside a reasonable range)"


def run_scenarios(
    bear_growth_y1, bear_growth_y5, bear_margin_y1, bear_margin_y5, bear_terminal, bear_wacc, bear_prob,
    base_growth_y1, base_growth_y5, base_margin_y1, base_margin_y5, base_terminal, base_wacc, base_prob,
    bull_growth_y1, bull_growth_y5, bull_margin_y1, bull_margin_y5, bull_terminal, bull_wacc, bull_prob,
    base_revenue, revenue_growth_y2, revenue_growth_y3, revenue_growth_y4, revenue_growth_long_term,
    operating_margin_y2, operating_margin_y3, operating_margin_y4,
    tax_rate, capex_pct_revenue, nwc_pct_revenue,
    equity_risk_premium, beta,
    cash_and_securities, nonmarketable_securities, total_debt, shares_outstanding,
    current_share_price,
):
    # Note: revenue_growth_y2/y3/y4 and operating_margin_y2/y3/y4 are still
    # accepted as inputs (so the scenario section still recomputes if those
    # main-model sliders move) but are no longer used here — each scenario
    # now interpolates its own Years 2-4 from its own Year 1/Year 5 values.
    shared_kwargs = dict(
        base_revenue=base_revenue, revenue_growth_long_term=revenue_growth_long_term,
        tax_rate=tax_rate, capex_pct_revenue=capex_pct_revenue, nwc_pct_revenue=nwc_pct_revenue,
        equity_risk_premium=equity_risk_premium, beta=beta,
        cash_and_securities=cash_and_securities, nonmarketable_securities=nonmarketable_securities,
        total_debt=total_debt, shares_outstanding=shares_outstanding,
    )

    scenarios = [
        ("Bear", bear_growth_y1, bear_growth_y5, bear_margin_y1, bear_margin_y5, bear_terminal, bear_wacc, bear_prob),
        ("Base", base_growth_y1, base_growth_y5, base_margin_y1, base_margin_y5, base_terminal, base_wacc, base_prob),
        ("Bull", bull_growth_y1, bull_growth_y5, bull_margin_y1, bull_margin_y5, bull_terminal, bull_wacc, bull_prob),
    ]

    prob_sum = bear_prob + base_prob + bull_prob
    probs_valid = abs(prob_sum - 100.0) < 0.01

    values = {}
    rows = []
    any_invalid = False
    for name, gy1, gy5, my1, my5, term, wacc, prob in scenarios:
        if term >= wacc:
            values[name] = None
            any_invalid = True
            rows.append({
                "Scenario": name, "Value per Share": "INVALID (terminal ≥ WACC)",
                "Probability": f"{prob:.0f}%", "Weighted Contribution": "—",
            })
            continue
        vps = _scenario_value_per_share(gy1, gy5, my1, my5, term, wacc, **shared_kwargs)
        values[name] = vps
        rows.append({
            "Scenario": name, "Value per Share": f"${vps:,.2f}",
            "Probability": f"{prob:.0f}%", "Weighted Contribution": f"${vps * prob / 100:,.2f}",
        })

    scenario_table = pd.DataFrame(rows)

    issues = []
    if not probs_valid:
        issues.append(f"Probabilities sum to {prob_sum:.0f}%, not 100%. Adjust the sliders so Bear + Base + Bull = 100%.")
    if any_invalid:
        issues.append("One or more scenarios has terminal growth ≥ WACC — fix before the weighted value is meaningful.")

    if issues:
        banner = f"""
        <div style="padding:12px 16px;border-radius:8px;background:#5c1a1a;color:#ffe5e5;border:1px solid #a33;">
        <b>⚠ INVALID:</b> {' '.join(issues)}
        </div>
        """
        return banner, scenario_table, "*Fix the issue above to see the probability-weighted value and call.*", ""

    banner = f"""
    <div style="padding:12px 16px;border-radius:8px;background:#1b4332;color:#eafbf0;border:1px solid #2f7a4d;">
    <b>Valid:</b> probabilities sum to 100% and every scenario's terminal growth is below its WACC.
    </div>
    """

    weighted_value = (values["Bear"] * bear_prob + values["Base"] * base_prob + values["Bull"] * bull_prob) / 100
    upside = (weighted_value / current_share_price - 1) * 100
    verdict = "undervalued" if upside >= 0 else "overvalued"
    if upside > BUY_SELL_THRESHOLD_PCT:
        call = "BUY"
    elif upside < -BUY_SELL_THRESHOLD_PCT:
        call = "SELL"
    else:
        call = "HOLD"

    call_summary = f"""
### Probability-Weighted Intrinsic Value: **${weighted_value:,.2f}**

Current Price: ${current_share_price:,.2f} → **{upside:+.1f}% ({verdict})**

## Call: **{call}**

*Rule of thumb: BUY if >{BUY_SELL_THRESHOLD_PCT:.0f}% undervalued, SELL if >{BUY_SELL_THRESHOLD_PCT:.0f}% overvalued, HOLD otherwise.*
"""

    required_growth_y1, required_wacc, required_terminal = _compute_breakeven(
        base_growth_y1, base_growth_y5, base_margin_y1, base_margin_y5, base_terminal, base_wacc,
        shared_kwargs, current_share_price,
    )

    def _fmt(x):
        return f"{x:.1f}%" if x is not None else "n/a (outside a reasonable range)"

    breakeven_md = f"""
### Break-Even: What Would Have to Be True

Holding every other **Base-case** assumption fixed, here's what a single lever would need to reach to justify today's ${current_share_price:,.2f} price:

| Lever | Base Case | Required to Justify ${current_share_price:,.2f} |
|---|---|---|
| Year 1 Revenue Growth | {base_growth_y1:.1f}% | {_fmt(required_growth_y1)} |
| WACC | {base_wacc:.2f}% | {_fmt(required_wacc)} |
| Terminal Growth Rate | {base_terminal:.1f}% | {_fmt(required_terminal)} |
"""

    return banner, scenario_table, call_summary, breakeven_md


def run_valuation(
    base_revenue, cash_and_securities, nonmarketable_securities, total_debt,
    shares_outstanding, current_share_price,
    revenue_growth_y1, revenue_growth_y2, revenue_growth_y3, revenue_growth_y4,
    revenue_growth_y5, revenue_growth_long_term, terminal_growth_rate,
    operating_margin_y1, operating_margin_y2, operating_margin_y3,
    operating_margin_y4, operating_margin_y5,
    tax_rate, risk_free_rate, equity_risk_premium, beta,
    capex_pct_revenue, nwc_pct_revenue,
):
    wacc = risk_free_rate + beta * equity_risk_premium
    banner = _validation_banner(terminal_growth_rate, wacc)

    if terminal_growth_rate >= wacc:
        empty_forecast = pd.DataFrame(columns=[
            "Year", "Growth %", "Revenue ($B)", "Op. Margin %", "NOPAT ($B)",
            "D&A ($B)", "CapEx ($B)", "ΔNWC ($B)", "FCF ($B)", "PV of FCF ($B)",
        ])
        placeholder = _placeholder_figure("Fix the invalid assumption above to see this chart.")
        return (
            banner, empty_forecast, "*Fix the invalid assumption above to see the valuation.*",
            placeholder, placeholder,
        )

    result = run_dcf(
        base_revenue=base_revenue,
        revenue_growth_y1=revenue_growth_y1 / 100, revenue_growth_y2=revenue_growth_y2 / 100,
        revenue_growth_y3=revenue_growth_y3 / 100, revenue_growth_y4=revenue_growth_y4 / 100,
        revenue_growth_y5=revenue_growth_y5 / 100,
        revenue_growth_long_term=revenue_growth_long_term / 100,
        terminal_growth_rate=terminal_growth_rate / 100,
        operating_margin_y1=operating_margin_y1 / 100, operating_margin_y2=operating_margin_y2 / 100,
        operating_margin_y3=operating_margin_y3 / 100, operating_margin_y4=operating_margin_y4 / 100,
        operating_margin_y5=operating_margin_y5 / 100,
        tax_rate=tax_rate / 100,
        capex_pct_of_revenue=capex_pct_revenue / 100,
        nwc_pct_of_revenue_increase=nwc_pct_revenue / 100,
        risk_free_rate=risk_free_rate / 100, equity_risk_premium=equity_risk_premium / 100, beta=beta,
        cash_and_securities=cash_and_securities, nonmarketable_securities=nonmarketable_securities,
        total_debt=total_debt, shares_outstanding=shares_outstanding,
    )

    forecast_table = pd.DataFrame([
        {
            "Year": y.year,
            "Growth %": round(y.growth_rate * 100, 2),
            "Revenue ($B)": round(y.revenue, 1),
            "Op. Margin %": round(y.operating_margin * 100, 1),
            "NOPAT ($B)": round(y.nopat, 1),
            "D&A ($B)": round(y.depreciation_and_amortization, 1),
            "CapEx ($B)": round(y.capex, 1),
            "ΔNWC ($B)": round(y.change_in_nwc, 1),
            "FCF ($B)": round(y.free_cash_flow, 1),
            "PV of FCF ($B)": round(y.present_value, 1),
        }
        for y in result.years
    ])

    valuation_summary = f"""
| Valuation Bridge | Amount |
|---|---|
| WACC (risk-free + beta × ERP) | {result.wacc * 100:.2f}% |
| Sum of PV — Years 1-10 FCF | ${result.pv_of_forecast_years:,.1f}B |
| Terminal Value (undiscounted, Year 10) | ${result.terminal_value:,.1f}B |
| PV of Terminal Value | ${result.pv_of_terminal_value:,.1f}B |
| **= Enterprise Value** | **${result.enterprise_value:,.1f}B** |
| + Cash & Marketable Securities | ${cash_and_securities:,.1f}B |
| + Non-Marketable Securities | ${nonmarketable_securities:,.1f}B |
| − Total Debt | ${total_debt:,.1f}B |
| **= Equity Value** | **${result.equity_value:,.1f}B** |
| ÷ Shares Outstanding | {shares_outstanding:,.2f}B |
| **= Intrinsic Value per Share** | **${result.value_per_share:,.2f}** |
| Current Share Price (for reference) | ${current_share_price:,.2f} |
"""

    revenue_fcf_fig = build_revenue_fcf_chart(result.years)
    waterfall_fig = build_waterfall_chart(result, cash_and_securities, nonmarketable_securities, total_debt)

    return banner, forecast_table, valuation_summary, revenue_fcf_fig, waterfall_fig


with gr.Blocks(title="NVIDIA DCF Valuation") as demo:
    gr.Markdown("# NVIDIA DCF Valuation")
    gr.Markdown(
        "All defaults are grounded in *NVIDIA Corporation: Valuing the Engine of the AI "
        "Revolution* (Manela, July 2026 case). Every input below states whether its default "
        "is a case-sourced figure or an MBA judgment call. The DCF engine (10-year forecast, "
        "terminal value, enterprise-to-equity bridge) recomputes live as you move any slider. "
        "Sensitivity charts and scenario analysis are not built yet."
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Starting Financials (from the case)")
            base_revenue = gr.Number(
                label="Base Revenue — FY2026 actual ($B)", value=CASE_BASE_REVENUE,
                info="Case Exhibit 1. Year 1 growth is applied on top of this.",
            )
            cash_and_securities = gr.Number(
                label="Cash & Marketable Securities ($B)", value=CASE_CASH_AND_SECURITIES,
                info="Case Exhibit 4, Q1 FY27: cash + marketable debt securities + marketable equity securities.",
            )
            nonmarketable_securities = gr.Number(
                label="Non-Marketable Securities ($B)", value=CASE_NONMARKETABLE_SECURITIES,
                info="Case Exhibit 4, Q1 FY27. Mostly private AI-company stakes (e.g. OpenAI). "
                     "The case itself asks whether these belong at full carrying value — judgment call.",
            )
            total_debt = gr.Number(
                label="Total Debt ($B)", value=CASE_TOTAL_DEBT,
                info="Case Exhibit 4, Q1 FY27.",
            )
            shares_outstanding = gr.Number(
                label="Shares Outstanding (B)", value=CASE_SHARES_OUTSTANDING,
                info="Case Market Data, July 29, 2026 close.",
            )
            current_share_price = gr.Number(
                label="Current Share Price ($)", value=CASE_SHARE_PRICE,
                info="Case Market Data, July 29, 2026 close.",
            )

        with gr.Column():
            gr.Markdown("### Revenue Growth Rate by Year")
            revenue_growth_y1 = gr.Slider(
                -50, 150, value=CASE_YEAR1_GROWTH, label="Year 1 Revenue Growth (%)",
                info="Case-grounded: consensus FY2027 revenue of $393.6B implies +82% over FY2026's $215.9B.",
            )
            revenue_growth_y2 = gr.Slider(
                -50, 150, value=45.0, label="Year 2 Revenue Growth (%)",
                info="Assumption: continued heavy hyperscaler capex (+77% guided for 2026) but decelerating "
                     "off the Year-1 base, echoing the case's own historical pattern of sharp step-downs "
                     "(126% -> 114% -> 65%).",
            )
            revenue_growth_y3 = gr.Slider(
                -50, 150, value=25.0, label="Year 3 Revenue Growth (%)",
                info="Assumption: further deceleration as the memory/interconnect bottleneck discussed in the "
                     "case (Section 3.4) caps how fast the buildout can scale.",
            )
            revenue_growth_y4 = gr.Slider(
                -50, 150, value=15.0, label="Year 4 Revenue Growth (%)",
                info="Assumption: growth approaching more normalized, semiconductor-cycle-like rates.",
            )
            revenue_growth_y5 = gr.Slider(
                -50, 150, value=8.0, label="Year 5 Revenue Growth (%)",
                info="Assumption: glide path nearly complete, one step above the long-term rate.",
            )
            revenue_growth_long_term = gr.Slider(
                -10, 30, value=6.0, label="Long-Term Revenue Growth (Year 6+, %)",
                info="Assumption: still modestly above GDP for a few more years before the perpetuity phase, "
                     "reflecting continued (but no longer extreme) AI infrastructure demand.",
            )
            terminal_growth_rate = gr.Slider(
                -2, 8, value=CASE_TERMINAL_GROWTH, label="Terminal Growth Rate — perpetuity (%)",
                info="Case-grounded: Section 6.1's own 'mature NVIDIA' example uses 3% growing in perpetuity. "
                     "Must stay below WACC (checked below).",
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Operating Margin by Year")
            operating_margin_y1 = gr.Slider(
                -20, 90, value=CASE_YEAR1_OPERATING_MARGIN, label="Year 1 Operating Margin (%)",
                info="Case-grounded: Q1 FY2027 actual operating margin was 65.6%, consistent with Q2 FY27 guidance.",
            )
            operating_margin_y2 = gr.Slider(
                -20, 90, value=63.0, label="Year 2 Operating Margin (%)",
                info="Assumption: slight compression begins as the case's competitive-erosion narrative "
                     "(NVIDIA's data-center accelerator share falling from ~92% to 75-85%) starts to bite.",
            )
            operating_margin_y3 = gr.Slider(
                -20, 90, value=60.0, label="Year 3 Operating Margin (%)",
                info="Assumption: continued gradual compression.",
            )
            operating_margin_y4 = gr.Slider(
                -20, 90, value=57.0, label="Year 4 Operating Margin (%)",
                info="Assumption: reflects the case's 'mix shift' scenario (Answer C, Section 6.3) — more volume, "
                     "lower margin, as inference workloads and custom silicon pressure pricing.",
            )
            operating_margin_y5 = gr.Slider(
                -20, 90, value=55.0, label="Year 5 Operating Margin (%)",
                info="Assumption: approaching a steadier, still-strong but less extraordinary margin. Held flat "
                     "beyond Year 5 for this stage of the model.",
            )

        with gr.Column():
            gr.Markdown("### Tax, Capex & Working Capital")
            tax_rate = gr.Slider(
                0, 50, value=CASE_TAX_RATE_DEFAULT, label="Tax Rate (%)",
                info="Case-grounded: company guidance for full-year FY2027 effective tax rate is 16-18%; "
                     "midpoint used.",
            )
            capex_pct_revenue = gr.Slider(
                0, 20, value=CASE_CAPEX_PCT_DEFAULT, label="Capital Expenditure (% of revenue)",
                info="Case-grounded: FY2026 actual capex/revenue was 2.8% (Exhibit 5).",
            )
            nwc_pct_revenue = gr.Slider(
                -30, 50, value=CASE_NWC_PCT_DEFAULT, label="Working Capital Change (% of revenue increase)",
                info="Case-grounded: FY2026's 'working capital absorption' was 19.3% of the year's revenue "
                     "increase (Exhibit 5). Note Q1 FY27 alone was actually negative (working capital released "
                     "cash) — flagged in the case as a one-off swing, so the fuller-year figure is used here.",
            )

            gr.Markdown("### WACC Components")
            risk_free_rate = gr.Slider(
                0, 10, value=CASE_RISK_FREE_RATE, label="Risk-Free Rate (%)",
                info="Case-grounded: 10-year U.S. Treasury yield, July 30, 2026.",
            )
            equity_risk_premium = gr.Slider(
                0, 10, value=CASE_EQUITY_RISK_PREMIUM, label="Equity Risk Premium (%)",
                info="Case-grounded: Damodaran's January 2026 estimate, as cited in the case.",
            )
            beta = gr.Slider(
                0.3, 3.5, value=CASE_BETA_DEFAULT, label="Beta",
                info="Judgment call: the case cites two very different betas — a 5-year regression beta of "
                     "2.21, and a bottom-up semiconductor-comparables range of 1.35-1.75. This defaults to the "
                     "midpoint of the bottom-up range (1.55); the case itself poses this exact choice as an "
                     "open discussion question.",
            )

    validation_banner = gr.HTML()

    gr.Markdown("### 10-Year Forecast")
    forecast_output = gr.Dataframe(label=None, wrap=True)

    gr.Markdown("### Valuation Summary — Enterprise Value, Equity Value, Intrinsic Value per Share")
    valuation_output = gr.Markdown()

    gr.Markdown("### Charts")
    with gr.Row():
        revenue_fcf_output = gr.Plot()
        waterfall_output = gr.Plot()
    sensitivity_output = gr.Plot()

    gr.Markdown("## Scenario Analysis: Bear / Base / Bull")
    gr.Markdown(
        "Each scenario has its own Year 1 & Year 5 growth, Year 1 & Year 5 margin, terminal growth, "
        "WACC, and probability. Years 2-4 growth/margin, tax rate, capex %, working capital %, and the "
        "balance sheet are shared with the Valuation tab above. Probabilities must sum to 100%."
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("#### Bear Case")
            bear_growth_y1 = gr.Slider(-50, 150, value=40.0, label="Year 1 Growth (%)",
                info="Deflationary scenario (case Answer A): demand air-pocket as efficiency gains outrun new demand.")
            bear_growth_y5 = gr.Slider(-50, 150, value=0.0, label="Year 5 Growth (%)",
                info="Growth flattens out entirely as hyperscalers digest over-provisioned capacity.")
            bear_margin_y1 = gr.Slider(-20, 90, value=60.0, label="Year 1 Margin (%)",
                info="Some immediate compression as competitive/pricing pressure starts.")
            bear_margin_y5 = gr.Slider(-20, 90, value=40.0, label="Year 5 Margin (%)",
                info="Severe compression: case's competitive-erosion + 'mix shift' narrative playing out in full.")
            bear_terminal = gr.Slider(-5, 8, value=1.5, label="Terminal Growth (%)",
                info="Barely above zero — a mature, cyclical semiconductor company, no lasting AI premium.")
            bear_wacc = gr.Slider(1, 30, value=13.0, label="WACC (%)",
                info="Higher discount rate reflecting elevated perceived risk in this outcome.")
            bear_prob = gr.Slider(0, 100, value=25.0, step=1, label="Probability (%)")

        with gr.Column():
            gr.Markdown("#### Base Case")
            base_growth_y1 = gr.Slider(-50, 150, value=CASE_YEAR1_GROWTH, label="Year 1 Growth (%)",
                info="Same as the Valuation tab default: consensus FY2027 revenue implies +82%.")
            base_growth_y5 = gr.Slider(-50, 150, value=8.0, label="Year 5 Growth (%)",
                info="Same as the Valuation tab default.")
            base_margin_y1 = gr.Slider(-20, 90, value=CASE_YEAR1_OPERATING_MARGIN, label="Year 1 Margin (%)",
                info="Same as the Valuation tab default: Q1 FY2027 actual.")
            base_margin_y5 = gr.Slider(-20, 90, value=55.0, label="Year 5 Margin (%)",
                info="Same as the Valuation tab default.")
            base_terminal = gr.Slider(-5, 8, value=CASE_TERMINAL_GROWTH, label="Terminal Growth (%)",
                info="Same as the Valuation tab default: the case's own 3% mature-NVIDIA illustration.")
            base_wacc = gr.Slider(1, 30, value=_BASE_WACC_DEFAULT, label="WACC (%)",
                info="Matches the Valuation tab's computed WACC (risk-free + beta × ERP) by default.")
            base_prob = gr.Slider(0, 100, value=50.0, step=1, label="Probability (%)")

        with gr.Column():
            gr.Markdown("#### Bull Case")
            bull_growth_y1 = gr.Slider(-50, 150, value=90.0, label="Year 1 Growth (%)",
                info="Jevons scenario (case Answer B): cheap, open AI unlocks even more demand than the base case.")
            bull_growth_y5 = gr.Slider(-50, 150, value=20.0, label="Year 5 Growth (%)",
                info="Buildout still robust in Year 5 — the moat holds and capex keeps pace.")
            bull_margin_y1 = gr.Slider(-20, 90, value=70.0, label="Year 1 Margin (%)",
                info="No compression — rack-scale moat (interconnect + CUDA) holds pricing power.")
            bull_margin_y5 = gr.Slider(-20, 90, value=62.0, label="Year 5 Margin (%)",
                info="Still very strong; only minimal erosion versus Year 1.")
            bull_terminal = gr.Slider(-5, 8, value=4.5, label="Terminal Growth (%)",
                info="Durable structural advantage persists well above typical GDP-level terminal growth.")
            bull_wacc = gr.Slider(1, 30, value=10.0, label="WACC (%)",
                info="Lower discount rate reflecting higher confidence in the durability of cash flows.")
            bull_prob = gr.Slider(0, 100, value=25.0, step=1, label="Probability (%)")

    scenario_banner = gr.HTML()
    gr.Markdown("#### Scenario Values & Probability Weighting")
    scenario_table_output = gr.Dataframe(label=None, wrap=True)
    scenario_call_output = gr.Markdown()
    breakeven_output = gr.Markdown()

    all_inputs = [
        base_revenue, cash_and_securities, nonmarketable_securities, total_debt,
        shares_outstanding, current_share_price,
        revenue_growth_y1, revenue_growth_y2, revenue_growth_y3, revenue_growth_y4,
        revenue_growth_y5, revenue_growth_long_term, terminal_growth_rate,
        operating_margin_y1, operating_margin_y2, operating_margin_y3,
        operating_margin_y4, operating_margin_y5,
        tax_rate, risk_free_rate, equity_risk_premium, beta,
        capex_pct_revenue, nwc_pct_revenue,
    ]
    all_outputs = [validation_banner, forecast_output, valuation_output, revenue_fcf_output, waterfall_output]

    # Sensitivity heatmap sweeps WACC and terminal growth itself, so those two
    # sliders are excluded from its input list — everything else feeds it.
    sensitivity_inputs = [
        base_revenue,
        revenue_growth_y1, revenue_growth_y2, revenue_growth_y3, revenue_growth_y4, revenue_growth_y5,
        revenue_growth_long_term,
        operating_margin_y1, operating_margin_y2, operating_margin_y3, operating_margin_y4, operating_margin_y5,
        tax_rate, capex_pct_revenue, nwc_pct_revenue,
        equity_risk_premium, beta,
        cash_and_securities, nonmarketable_securities, total_debt, shares_outstanding,
        current_share_price,
    ]

    scenario_inputs = [
        bear_growth_y1, bear_growth_y5, bear_margin_y1, bear_margin_y5, bear_terminal, bear_wacc, bear_prob,
        base_growth_y1, base_growth_y5, base_margin_y1, base_margin_y5, base_terminal, base_wacc, base_prob,
        bull_growth_y1, bull_growth_y5, bull_margin_y1, bull_margin_y5, bull_terminal, bull_wacc, bull_prob,
        base_revenue, revenue_growth_y2, revenue_growth_y3, revenue_growth_y4, revenue_growth_long_term,
        operating_margin_y2, operating_margin_y3, operating_margin_y4,
        tax_rate, capex_pct_revenue, nwc_pct_revenue,
        equity_risk_premium, beta,
        cash_and_securities, nonmarketable_securities, total_debt, shares_outstanding,
        current_share_price,
    ]
    scenario_outputs = [scenario_banner, scenario_table_output, scenario_call_output, breakeven_output]

    for component in all_inputs:
        component.change(fn=run_valuation, inputs=all_inputs, outputs=all_outputs)

    for component in sensitivity_inputs:
        component.change(fn=build_sensitivity_heatmap, inputs=sensitivity_inputs, outputs=sensitivity_output)

    for component in scenario_inputs:
        component.change(fn=run_scenarios, inputs=scenario_inputs, outputs=scenario_outputs)

    demo.load(fn=run_valuation, inputs=all_inputs, outputs=all_outputs)
    demo.load(fn=run_scenarios, inputs=scenario_inputs, outputs=scenario_outputs)
    demo.load(fn=build_sensitivity_heatmap, inputs=sensitivity_inputs, outputs=sensitivity_output)

if __name__ == "__main__":
    demo.launch(share=os.environ.get("GRADIO_SHARE") == "1")
