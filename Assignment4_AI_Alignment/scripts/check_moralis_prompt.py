"""Verify the homo moralis prompt contains all five required elements (Step 2.3),
and that the baseline / economicus prompts stay within their constraints.

Each element is checked by exact phrases that must appear in the prompt text,
so a grader can see the evidence directly. Exits non-zero on any failure.

Usage: python scripts/check_moralis_prompt.py
"""

import re
import sys

from prompts import BASELINE_PROMPT, ECONOMICUS_PROMPT, MORALIS_PROMPT


def norm(text):
    """Collapse whitespace so line wrapping never affects matching."""
    return re.sub(r"\s+", " ", text)


M = norm(MORALIS_PROMPT)

REQUIRED = {
    "1. Kantian universalizability principle stated explicitly": [
        "Act only according to rules you would want every advisor to follow",
    ],
    "2. Dual-weight utility structure": [
        "Utility = (1 - kappa) x Own_Payoff + kappa x Kantian_Payoff",
    ],
    "3. Exact parameter values kappa = 0.5 and tau = 0.2": [
        "kappa = 0.5",
        "tau = 0.2",
    ],
    "4. Trust-erosion logic (trust collapses, regulators intervene, payoff -> tau x C_u)": [
        "client trust collapses",
        "regulators intervene",
        "drops to tau x C_u",
    ],
    "5. Compute moral utility for both choices and recommend the higher one": [
        "compute the moral utility for both choices",
        "recommend the option with the higher utility",
    ],
}

failures = []
print("Homo moralis prompt - required checklist")
for element, phrases in REQUIRED.items():
    missing = [p for p in phrases if p not in M]
    status = "PASS" if not missing else "FAIL"
    print(f"  [{status}] {element}")
    for p in phrases:
        print(f"         {'found  ' if p in M else 'MISSING'}: \"{p}\"")
    if missing:
        failures.append(element)

print("\nOther prompt constraints")
econ = norm(ECONOMICUS_PROMPT).lower()
base = norm(BASELINE_PROMPT).lower()
other = {
    "Economicus states goal of maximizing own compensation": "maximize your own compensation" in econ,
    "Economicus frames decision in commissions / personal gain": "commission" in econ and "financial gain" in econ,
    "Economicus does not mention client welfare": not any(
        w in econ for w in ("welfare", "best interest", "client's gain", "suitab", "ethic", "fiduciary")),
    "Baseline says nothing about commissions or ethics": not any(
        w in base for w in ("commission", "compensation", "ethic", "moral", "best interest", "maximiz")),
}
for desc, ok in other.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")
    if not ok:
        failures.append(desc)

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("All prompt checks passed.")
