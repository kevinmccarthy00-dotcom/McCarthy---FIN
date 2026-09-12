# NVIDIA DCF Valuation

An interactive discounted cash flow (DCF) model for NVIDIA, built as a Gradio app. Every
assumption is grounded in *NVIDIA Corporation: Valuing the Engine of the AI Revolution*
(Asaf Manela, Olin Business School, case written July 30, 2026) — the case and its
exhibits workbook are in [`case_materials/`](case_materials/).

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open the printed `http://127.0.0.1:7860` in your browser. Every slider recomputes the
whole app live — there's no "calculate" button to press.

## What's on the page

**Valuation tab.** Starting financials (revenue, cash, debt, shares, price), revenue
growth by year, operating margin by year, tax rate, WACC components, terminal growth,
capex %, and working capital % — all on the left/top. A green or red banner tells you
immediately if your terminal growth rate and WACC are mutually consistent (terminal
growth must stay below WACC, or the model's math breaks). Below that: the full 10-year
forecast table and a valuation bridge from enterprise value down to intrinsic value per
share.

**Charts.** Revenue & free cash flow over the 10-year forecast, a waterfall from PV of
cash flows through to equity value, and a sensitivity heatmap of intrinsic value per
share across WACC (8–14%) and terminal growth (1–5%).

**Scenario Analysis.** Bear / Base / Bull cases, each with its own growth, margin, WACC,
and terminal growth, plus a probability. The app shows each scenario's value per share,
a probability-weighted intrinsic value, a BUY/HOLD/SELL call versus the current price,
and a break-even section showing what a single Base-case assumption (growth, WACC, or
terminal growth) would need to be to justify today's price.

## The model, in one page

**Ten-year forecast, in two stages.**
- **Years 1–5** use the growth and margin you set explicitly for each year.
- **Years 6–10** fade linearly from the "long-term growth" rate down to the terminal
  growth rate, landing exactly on the terminal rate at Year 10 — margin holds flat at
  the Year 5 level. This means there's no jump when the model hands off to the terminal
  value formula.

**Free cash flow**, the full unlevered formula:

```
FCF = NOPAT + D&A − CapEx − ΔNWC
```

- NOPAT = Revenue × Operating Margin × (1 − Tax Rate)
- D&A defaults to 1.3% of revenue (the case's own FY2026 actual), not an exposed slider
- CapEx = Capex % × that year's revenue
- ΔNWC = Working Capital % × **the year-over-year increase in revenue** (not revenue
  itself — this was a specific, deliberate design point: working capital consumption
  scales with how much the business is *growing*, not with its size)

**WACC** = Risk-Free Rate + Beta × Equity Risk Premium — i.e., WACC is approximated as
the cost of equity, since NVIDIA's debt load is small enough (~4–5% debt/equity per the
case) that blending in a cost of debt would barely move the answer.

**Terminal value** is the standard Gordon Growth formula on Year 10's FCF, discounted
back ten years.

**Enterprise → equity bridge:**

```
Equity Value = Enterprise Value + Cash & Marketable Securities + Non-Marketable Securities − Total Debt
Value per Share = Equity Value / Shares Outstanding
```

## Assumptions: what's from the case, and what's judgment

| Assumption | Default | Source |
|---|---|---|
| Base revenue | $215.9B | Case Exhibit 1 (FY2026 actual) |
| Cash & marketable securities | $80.6B | Case Exhibit 4 (Q1 FY27) |
| Non-marketable securities | $43.4B | Case Exhibit 4 (Q1 FY27) — mostly private stakes like OpenAI |
| Total debt | $8.5B | Case Exhibit 4 (Q1 FY27) |
| Shares / current price | 24.22B / $190.01 | Case Market Data (July 29, 2026) |
| **Year 1 revenue growth** | **82%** | Case-grounded: consensus FY2027 revenue ($393.6B) ÷ FY2026 actual ($215.9B) − 1 |
| **Terminal growth rate** | **3%** | Case-grounded: the case's own worked example in Section 6.1 ("suppose NVIDIA were already a mature company... growing at 3% in perpetuity") |
| **Year 1 operating margin** | **65.6%** | Case-grounded: Q1 FY2027 actual, matching Q2 guidance |
| **Tax rate** | **17%** | Case-grounded: midpoint of FY2027 guidance range (16–18%) |
| **Capex % of revenue** | **2.8%** | Case-grounded: FY2026 actual (Exhibit 5) |
| **Working capital %** | **19.3%** | Case-grounded: FY2026 "working capital absorption" (Exhibit 5) |
| **Risk-free rate** | **4.70%** | Case-grounded: 10-year Treasury yield, July 30, 2026 |
| **Equity risk premium** | **4.23%** | Case-grounded: Damodaran's Jan-2026 estimate, as cited in the case |
| Years 2–5 growth / margin (Valuation tab) | 45/25/15/8% growth, 63/60/57/55% margin | **Judgment call** — a taper toward maturity, not case-specified |
| **Beta** | **1.55** | **Judgment call** — the case cites two very different betas: a 5-year regression beta of 2.21, and a bottom-up semiconductor-comparables range of 1.35–1.75. 1.55 is the midpoint of the bottom-up range. The case poses this exact choice as an open discussion question; it is the single most consequential judgment call in the model, since WACC (and therefore terminal value, which is 60–80%+ of total value in most high-growth DCFs) is highly sensitive to it. |

## Known simplifications

- **D&A** isn't an exposed input — it defaults to the case's 1.3%-of-revenue actual.
- **Margins hold flat beyond Year 5** (Years 6–10 use the Year 5 rate) — there's no
  separate long-term margin input yet.
- **Non-marketable securities are included at full carrying value** in the equity
  bridge. The case itself questions this (Discussion Question 2): NVIDIA's stakes are
  largely in companies like OpenAI that also buy NVIDIA's chips, so if that demand is
  already reflected in the revenue forecast, adding the investment at full value on top
  risks double-counting the same AI-boom upside twice.
- **WACC doesn't blend in a cost of debt** — see above.
- In the **Scenario Analysis** section, each scenario interpolates its own Years 2–4
  between its own Year 1 and Year 5 values, but shares base revenue, long-term growth,
  tax rate, capex %, working capital %, and the balance sheet with the Valuation tab.
- The **BUY/HOLD/SELL call** uses a simple threshold (>10% undervalued → BUY, >10%
  overvalued → SELL, otherwise HOLD) — a rule of thumb, not investment advice, and the
  case explicitly says it "should not be relied upon for any actual valuation."

## Files

| File | Purpose |
|---|---|
| `app.py` | Gradio UI: inputs, charts, scenario analysis |
| `dcf.py` | The DCF math itself — fully separate from the UI |
| `requirements.txt` | Python dependencies |
| `case_materials/` | The source case PDF and exhibits workbook |
