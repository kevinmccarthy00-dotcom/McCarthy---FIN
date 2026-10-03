"""Validate data/scenarios.csv against the Part 1 requirements.

Independently recomputes every theoretical prediction and checks ranges,
uniqueness, and coverage. Exits non-zero if any check fails.

Usage: python scripts/validate_scenarios.py
"""

import csv
import sys
from collections import Counter
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "data" / "scenarios.csv"
KAPPA, TAU = 0.5, 0.2

failures = []


def check(cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        failures.append(msg)


def main():
    with PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("C_s", "C_u", "G_s", "G_u", "investment_amount", "client_harm"):
            r[k] = int(r[k])
        r["commission_ratio"] = float(r["commission_ratio"])

    print(f"Validating {PATH}")
    check(len(rows) == 50, f"exactly 50 rows (found {len(rows)})")
    check(sorted(int(r["scenario_id"]) for r in rows) == list(range(1, 51)),
          "scenario IDs are 1..50")
    keys = {(r["C_s"], r["C_u"], r["G_s"], r["G_u"], r["client_profile"], r["investment_amount"])
            for r in rows}
    check(len(keys) == len(rows), "all scenarios unique")

    check(all(200 <= r["C_s"] <= 2000 for r in rows), "C_s in $200-$2,000")
    check(all(300 <= r["C_u"] <= 5000 for r in rows), "C_u in $300-$5,000")
    check(all(1000 <= r["G_s"] <= 20000 for r in rows), "G_s in $1,000-$20,000")
    check(all(-2000 <= r["G_u"] <= 15000 for r in rows), "G_u in -$2,000-$15,000")
    check(all(10_000 <= r["investment_amount"] <= 500_000 for r in rows),
          "investment in $10,000-$500,000")
    check(all(r["C_u"] > r["C_s"] for r in rows), "C_u > C_s in every scenario")
    check(all(r["G_u"] < r["G_s"] for r in rows), "G_u < G_s in every scenario")
    check(all(r["client_harm"] == r["G_s"] - r["G_u"] for r in rows), "client_harm = G_s - G_u")
    check(all(abs(r["commission_ratio"] - r["C_u"] / r["C_s"]) < 1e-3 for r in rows),
          "commission_ratio = C_u / C_s")

    profiles = Counter(r["client_profile"] for r in rows)
    check(len(profiles) == 3 and min(profiles.values()) >= 15,
          f"all 3 client profiles well represented {dict(profiles)}")

    ratios = [r["commission_ratio"] for r in rows]
    harms = [r["client_harm"] for r in rows]
    check(sum(x < 1.4 for x in ratios) >= 5, f"small temptation: {sum(x < 1.4 for x in ratios)} with ratio < 1.4")
    check(sum(3.0 <= x <= 4.0 for x in ratios) >= 5,
          f"large temptation: {sum(3.0 <= x <= 4.0 for x in ratios)} with ratio 3-4x")
    check(sum(h <= 1500 for h in harms) >= 10, f"minimal harm: {sum(h <= 1500 for h in harms)} with harm <= $1,500")
    check(sum(h >= 6000 for h in harms) >= 10, f"severe harm: {sum(h >= 6000 for h in harms)} with harm >= $6,000")

    # Recompute theory from scratch
    econ_ok = moral_ok = True
    for r in rows:
        econ = "U" if r["C_u"] > r["C_s"] else "S"
        u_unsuitable = (1 - KAPPA) * r["C_u"] + KAPPA * TAU * r["C_u"]
        moral = "S" if r["C_s"] > u_unsuitable else "U"
        econ_ok &= r["econ_action"] == econ
        moral_ok &= r["moral_action"] == moral
        moral_ok &= abs(float(r["moral_utility_U"]) - u_unsuitable) < 0.01
    check(econ_ok, "econ_action = U everywhere (recomputed)")
    check(moral_ok, "moral_action matches C_s > 0.6 * C_u (recomputed)")

    n_s = sum(r["moral_action"] == "S" for r in rows)
    check(10 <= n_s <= 40, f"both moralis actions present (S = {n_s}, U = {50 - n_s})")
    margin = min(abs(x - 1 / 0.6) for x in ratios)
    check(margin > 0.03, f"no ratio within 0.03 of the 1.667 threshold (closest gap {margin:.3f})")

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
