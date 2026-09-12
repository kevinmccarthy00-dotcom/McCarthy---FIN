import os

import gradio as gr
import pandas as pd

from dcf import run_dcf

# =============================================================================
# PART 1 — INPUT CONTROLS  /  PART 2 — DCF ENGINE
#
# Source for every "Case" figure below: "NVIDIA Corporation: Valuing the
# Engine of the AI Revolution" (Asaf Manela, Olin Business School, case
# written July 30, 2026), plus its accompanying exhibits workbook.
#
# The DCF math itself lives in dcf.py, kept fully separate from this UI file.
# Part 3 (sensitivity charts / scenario analysis) is not built yet.
# =============================================================================

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
        return banner, empty_forecast, "*Fix the invalid assumption above to see the valuation.*"

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

    return banner, forecast_table, valuation_summary


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
    all_outputs = [validation_banner, forecast_output, valuation_output]

    for component in all_inputs:
        component.change(fn=run_valuation, inputs=all_inputs, outputs=all_outputs)

    demo.load(fn=run_valuation, inputs=all_inputs, outputs=all_outputs)

if __name__ == "__main__":
    demo.launch(share=os.environ.get("GRADIO_SHARE") == "1")
