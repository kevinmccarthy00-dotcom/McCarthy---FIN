# Audit — Part 3: Hallucination Audit & Trust Assessment

This audit reviews claims made in the Stage 5 recommendation letters produced
by the AI Financial Advisor Agent for three clients, classifying each claim
and assessing overall trust in the agent's output.

## Part 3.1 — Hallucination Audit

**Classifications:**
- **Verified** — directly supported by the client's supplied profile data, or by a reputable external financial source.
- **Plausible but unverified** — a reasonable inference or judgment call, but not something the available data can confirm.
- **Fabricated** — stated as fact but not supported by any information the agent actually had.

| # | Client | Claim from recommendation letter | Classification | Evaluation / Source |
|---|--------|-----------------------------------|-----------------|----------------------|
| 1 | A | The client is 28, earns $95,000, has $45,000 in savings, and owes $35,000 in student loans at 5.5% | Verified | Matches the supplied client profile / Stage 1 intake summary directly. |
| 2 | A | The client has "steady" or "stable" income | Fabricated | The profile states an income figure but provides no evidence about job security, tenure, or income stability; Stage 1 itself flagged job stability as missing information. |
| 3 | A | Investment-grade bonds generally provide portfolio ballast / lower volatility than equities | Verified | Confirmed by the U.S. SEC's investor education site: bonds are described as generally less volatile than stocks and useful for offsetting equity volatility. Source: [Investor.gov — Asset Allocation and Diversification](https://www.investor.gov/introduction-investing/getting-started/asset-allocation) |
| 4 | A | Paying down the 5.5% student loan is a "compelling, risk-free alternative" to investing | Plausible but unverified | Reasonable framing in general personal-finance terms, but whether it's actually the better choice for this client depends on liquidity needs, tax treatment, loan repayment terms, and employer-match/other alternatives not supplied in the profile. |
| 5 | B | The client is 58, earns $180,000, has $1.2 million saved, is approximately 80% equity, and has a spouse pension covering about 40% of retirement expenses | Verified | Matches the supplied client profile / Stage 1 intake summary directly. |
| 6 | B | Sequence-of-returns risk is important near retirement when withdrawals begin | Verified | Confirmed by a reputable external source explaining that poor returns early in retirement, combined with ongoing withdrawals, can materially shorten how long a portfolio lasts. Source: [Charles Schwab — What Is Sequence-of-Returns Risk?](https://www.schwab.com/learn/story/timing-matters-understanding-sequence-returns-risk) |
| 7 | B | The client's 80% equity allocation is a strong behavioral signal of aggressive risk tolerance | Plausible but unverified | A common heuristic, but an existing allocation this concentrated in equities could equally reflect inertia, legacy holdings, or lack of rebalancing rather than a deliberate, confirmed risk preference. |
| 8 | B | The client has a "lack of debt" | Fabricated | No debt information was supplied in the intake at all; absence of disclosed debt is not evidence of no debt, and the agent should have flagged this as unknown rather than asserting it. |
| 9 | C | The client is 52, earns $65,000, has $800,000 in savings including a $500,000 recent life-insurance payout, and has two children with four years of college remaining | Verified | Matches the supplied client profile / Stage 1 intake summary directly. |
| 10 | C | The client will "likely rely on the portfolio for cash flow" | Plausible but unverified | A reasonable assumption given the income/asset mix described, but actual current spending, other income sources, and withdrawal needs were not provided in the intake. |
| 11 | C | Roughly 15–20% should be maintained in a highly liquid bucket for college costs | Plausible but unverified | A sensible planning heuristic, but actual tuition costs, financial aid, 529 balances, or other college-funding sources were not provided, so the specific percentage cannot be confirmed against real numbers. |
| 12 | C | The unknown $300,000 brokerage account should be reviewed before finalizing an overall allocation because it could contain concentrated or overlapping exposures | Verified | The composition being unknown is a fact of the intake itself, and the underlying principle is confirmed by a reputable source on diversification risk. Source: [Investor.gov — Diversify Your Investments](https://www.investor.gov/introduction-investing/investing-basics/save-and-invest/diversify-your-investments) |

**Totals across all 12 claims:**

| Classification | Count |
|----------------|-------|
| Verified | 6 |
| Plausible but unverified | 4 |
| Fabricated | 2 |

## Part 3.2 — Trust Assessment

### Client A
- **Would I act on it without a human advisor?** No.
- **Biggest issue:** The recommendation allocates savings before resolving emergency-fund needs and the tradeoff between investing and paying down the 5.5% student loan.
- **Trust score:** 3/5.

### Client B
- **Would I act on it without a human advisor?** No.
- **Biggest issue:** No retirement cash-flow analysis. We do not know the dollar amount of expenses not covered by the pension, Social Security timing, or required portfolio withdrawals.
- **Trust score:** 4/5.

### Client C
- **Would I act on it without a human advisor?** No.
- **Biggest issue:** The agent recommends precise allocations before knowing actual college costs, household cash needs, or the composition of the inherited brokerage account.
- **Trust score:** 3/5.

## Conclusion

Across all three clients, the agent was strongest at organizing structured data, retrieving and correctly citing real market information, and producing consistent, well-formatted reports stage over stage. Its biggest weakness was a pattern of false precision: turning genuinely missing information (job stability, debt, cash-flow needs, account composition, college costs) into confident-sounding assumptions or descriptive claims rather than clearly flagged unknowns. This means the agent's output is a useful, well-organized starting draft for a human advisor to review, but not something a client should act on directly without that human review closing the gaps this audit identified.
