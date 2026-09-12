import os

import gradio as gr
import pandas as pd

# =============================================================================
# PART 1 — INPUT CONTROLS
#
# Source for every "Case" figure below: "NVIDIA Corporation: Valuing the
# Engine of the AI Revolution" (Asaf Manela, Olin Business School, case
# written July 30, 2026), plus its accompanying exhibits workbook.
#
# This file intentionally stops at input collection + a summary table.
# The DCF calculation engine, charts, and BUY/HOLD/SELL logic from earlier
# prototyping have been removed and will be rebuilt in later parts once the
# richer input set here (year-by-year margins, WACC built from components,
# capex %, working capital change) is ready to be wired in.
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


def build_assumptions_table(
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
    is_valid = terminal_growth_rate < wacc

    if is_valid:
        banner = f"""
        <div style="padding:12px 16px;border-radius:8px;background:#1b4332;color:#eafbf0;border:1px solid #2f7a4d;">
        <b>Valid:</b> terminal growth rate ({terminal_growth_rate:.1f}%) is below WACC ({wacc:.2f}%). Terminal value math holds.
        </div>
        """
    else:
        banner = f"""
        <div style="padding:12px 16px;border-radius:8px;background:#5c1a1a;color:#ffe5e5;border:1px solid #a33;">
        <b>⚠ INVALID ASSUMPTION:</b> terminal growth rate ({terminal_growth_rate:.1f}%) must be
        <u>below</u> WACC ({wacc:.2f}%), or the terminal value formula divides by a negative or
        zero number and produces a meaningless (or negative) result. Lower the terminal growth
        rate or raise WACC (higher beta / risk-free rate / equity risk premium).
        </div>
        """

    rows = [
        # --- Starting financials ---
        ("Base Revenue (FY2026 actual)", f"${base_revenue:.1f}B", "Case (Exhibit 1)"),
        ("Cash & Marketable Securities", f"${cash_and_securities:.1f}B", "Case (Exhibit 4, Q1 FY27)"),
        ("Non-Marketable Securities", f"${nonmarketable_securities:.1f}B", "Case (Exhibit 4, Q1 FY27) — judgment call on treatment, see note"),
        ("Total Debt", f"${total_debt:.1f}B", "Case (Exhibit 4, Q1 FY27)"),
        ("Shares Outstanding", f"{shares_outstanding:.2f}B", "Case (Market Data, Jul 29 2026)"),
        ("Current Share Price", f"${current_share_price:.2f}", "Case (Market Data, Jul 29 2026)"),
        # --- Revenue growth ---
        ("Revenue Growth — Year 1", f"{revenue_growth_y1:.1f}%", "Case (consensus FY2027 revenue implies +82%)"),
        ("Revenue Growth — Year 2", f"{revenue_growth_y2:.1f}%", "Assumption (MBA judgment)"),
        ("Revenue Growth — Year 3", f"{revenue_growth_y3:.1f}%", "Assumption (MBA judgment)"),
        ("Revenue Growth — Year 4", f"{revenue_growth_y4:.1f}%", "Assumption (MBA judgment)"),
        ("Revenue Growth — Year 5", f"{revenue_growth_y5:.1f}%", "Assumption (MBA judgment)"),
        ("Revenue Growth — Long-Term (Year 6+)", f"{revenue_growth_long_term:.1f}%", "Assumption (MBA judgment)"),
        ("Terminal Growth Rate (perpetuity)", f"{terminal_growth_rate:.1f}%", "Case (Section 6.1 illustrative mature-state assumption)"),
        # --- Operating margin ---
        ("Operating Margin — Year 1", f"{operating_margin_y1:.1f}%", "Case (Q1 FY2027 actual / Q2 FY27 guidance-implied)"),
        ("Operating Margin — Year 2", f"{operating_margin_y2:.1f}%", "Assumption (MBA judgment)"),
        ("Operating Margin — Year 3", f"{operating_margin_y3:.1f}%", "Assumption (MBA judgment)"),
        ("Operating Margin — Year 4", f"{operating_margin_y4:.1f}%", "Assumption (MBA judgment)"),
        ("Operating Margin — Year 5", f"{operating_margin_y5:.1f}%", "Assumption (MBA judgment); held flat beyond Year 5 for now"),
        # --- Tax / capex / working capital ---
        ("Tax Rate", f"{tax_rate:.1f}%", "Case (FY2027 guidance range 16-18%, midpoint used)"),
        ("Capital Expenditure (% of revenue)", f"{capex_pct_revenue:.1f}%", "Case (Exhibit 5, FY2026 actual)"),
        ("Working Capital Change (% of revenue increase)", f"{nwc_pct_revenue:.1f}%", "Case (Exhibit 5, FY2026 actual)"),
        # --- WACC components ---
        ("Risk-Free Rate", f"{risk_free_rate:.2f}%", "Case (10-yr UST yield, Exhibit 6A)"),
        ("Equity Risk Premium", f"{equity_risk_premium:.2f}%", "Case (Damodaran Jan-2026, Exhibit 6A)"),
        ("Beta", f"{beta:.2f}", "Assumption (midpoint of case's bottom-up range 1.35-1.75; case also cites a 2.21 regression beta — an open judgment call, see note)"),
        ("WACC (computed = risk-free + beta × ERP)", f"{wacc:.2f}%", "Computed from the three inputs above"),
    ]

    table = pd.DataFrame(rows, columns=["Assumption", "Value", "Basis"])
    return banner, table


with gr.Blocks(title="NVIDIA DCF Valuation") as demo:
    gr.Markdown("# NVIDIA DCF Valuation — Part 1: Input Controls")
    gr.Markdown(
        "All defaults are grounded in *NVIDIA Corporation: Valuing the Engine of the AI "
        "Revolution* (Manela, July 2026 case). Every input below states whether its default "
        "is a case-sourced figure or an MBA judgment call. This screen only collects and "
        "displays assumptions — the DCF calculation itself is built in a later part."
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
    gr.Markdown("### Assumptions Summary")
    assumptions_table = gr.Dataframe(label=None, wrap=True)

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
    all_outputs = [validation_banner, assumptions_table]

    for component in all_inputs:
        component.change(fn=build_assumptions_table, inputs=all_inputs, outputs=all_outputs)

    demo.load(fn=build_assumptions_table, inputs=all_inputs, outputs=all_outputs)

if __name__ == "__main__":
    demo.launch(share=os.environ.get("GRADIO_SHARE") == "1")
