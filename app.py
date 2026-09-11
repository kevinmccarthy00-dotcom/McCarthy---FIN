import os

import gradio as gr
import pandas as pd

from dcf import run_dcf

# Starting figures for NVIDIA (all dollar amounts in $ billions)
NVDA_REVENUE = 130.5
NVDA_CASH = 43.2
NVDA_DEBT = 8.5
NVDA_SHARES = 24.5  # billions of shares
NVDA_PRICE = 180.0


def calculate(
    revenue, growth_y1, growth_y5, growth_terminal, gross_margin,
    opex_pct, tax_rate, wacc, horizon, cash, debt, shares, current_price,
):
    result = run_dcf(
        revenue=revenue,
        growth_y1=growth_y1 / 100,
        growth_y5=growth_y5 / 100,
        growth_terminal=growth_terminal / 100,
        gross_margin=gross_margin / 100,
        opex_pct_of_revenue=opex_pct / 100,
        tax_rate=tax_rate / 100,
        wacc=wacc / 100,
        horizon=int(horizon),
        cash=cash,
        debt=debt,
        shares_outstanding=shares,
    )

    table = pd.DataFrame([
        {
            "Year": y.year,
            "Growth %": round(y.growth_rate * 100, 1),
            "Revenue ($B)": round(y.revenue, 1),
            "Operating Income ($B)": round(y.operating_income, 1),
            "NOPAT ($B)": round(y.nopat, 1),
            "Free Cash Flow ($B)": round(y.free_cash_flow, 1),
            "Present Value ($B)": round(y.present_value, 1),
        }
        for y in result.years
    ])

    upside = (result.value_per_share / current_price - 1) * 100
    verdict = "undervalued" if upside > 0 else "overvalued"

    if upside > 10:
        signal = "BUY"
    elif upside < -10:
        signal = "SELL"
    else:
        signal = "HOLD"

    scorecard = f"""
## Results

| Metric | Value |
|---|---|
| **Intrinsic Value / Share** | **${result.value_per_share:,.2f}** |
| **Enterprise Value** | **${result.enterprise_value:,.1f}B** |
| **Current Price** | **${current_price:,.2f}** |
| **Upside / Downside** | **{'+' if upside >= 0 else ''}{upside:,.1f}% ({verdict})** |
| **Signal** | **{signal}** |

*Signal rule of thumb: BUY if >10% undervalued, SELL if >10% overvalued, HOLD otherwise.*

| Value Bridge | Amount |
|---|---|
| PV of forecast-period cash flows | ${result.pv_of_forecast_years:,.1f}B |
| PV of terminal value | ${result.pv_of_terminal_value:,.1f}B |
| = Enterprise value | ${result.enterprise_value:,.1f}B |
| + Cash | ${cash:,.1f}B |
| − Debt | ${debt:,.1f}B |
| = Equity value | ${result.equity_value:,.1f}B |
| ÷ Shares outstanding | {shares:,.1f}B |
"""

    return scorecard, table


with gr.Blocks(title="DCF Valuation") as demo:
    gr.Markdown("# NVIDIA DCF Valuation")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Starting Financials")
            revenue = gr.Number(label="Current Revenue ($B)", value=NVDA_REVENUE)
            cash = gr.Number(label="Cash ($B)", value=NVDA_CASH)
            debt = gr.Number(label="Debt ($B)", value=NVDA_DEBT)
            shares = gr.Number(label="Shares Outstanding (B)", value=NVDA_SHARES)
            current_price = gr.Number(label="Current Share Price ($)", value=NVDA_PRICE)

        with gr.Column():
            gr.Markdown("### Growth & Margin Assumptions")
            growth_y1 = gr.Slider(0, 60, value=30, label="Year 1 Revenue Growth (%)")
            growth_y5 = gr.Slider(0, 40, value=15, label="Year 5 Revenue Growth (%)")
            growth_terminal = gr.Slider(0, 8, value=4, label="Terminal Growth Rate (%)")
            gross_margin = gr.Slider(0, 100, value=75, label="Gross Margin (%)")
            opex_pct = gr.Slider(0, 50, value=20, label="Operating Expenses (% of Revenue)")
            tax_rate = gr.Slider(0, 40, value=15, label="Tax Rate (%)")
            wacc = gr.Slider(1, 20, value=11, label="WACC (%)")
            horizon = gr.Slider(3, 15, value=10, step=1, label="Forecast Horizon (Years)")

    summary_output = gr.Markdown()
    table_output = gr.Dataframe(label="Year-by-Year Projection")

    all_inputs = [
        revenue, growth_y1, growth_y5, growth_terminal, gross_margin,
        opex_pct, tax_rate, wacc, horizon, cash, debt, shares, current_price,
    ]
    all_outputs = [summary_output, table_output]

    for component in all_inputs:
        component.change(fn=calculate, inputs=all_inputs, outputs=all_outputs)

    demo.load(fn=calculate, inputs=all_inputs, outputs=all_outputs)

if __name__ == "__main__":
    demo.launch(share=os.environ.get("GRADIO_SHARE") == "1")
