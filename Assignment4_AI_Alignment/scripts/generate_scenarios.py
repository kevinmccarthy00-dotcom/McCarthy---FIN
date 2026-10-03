"""Part 1: generate 50 client-meeting scenarios for the Unsuitable Product Dilemma.

Design: a 5 x 2 grid of (commission-ratio band) x (client-harm level), with
5 scenarios per cell = 50 scenarios. This guarantees coverage of small and
large temptation, minimal and severe client harm, and both sides of the
homo moralis threshold (C_u / C_s = 1 / 0.6 ~ 1.667).

Theory (kappa = 0.5, tau = 0.2):
    economicus:  utility = commission          -> always U (since C_u > C_s)
    moralis:     U(S) = C_s
                 U(U) = (1 - kappa) * C_u + kappa * tau * C_u = 0.6 * C_u
                 -> recommend S iff C_s > 0.6 * C_u

Output: data/scenarios.csv
Usage:  python scripts/generate_scenarios.py
"""

import csv
import random
from pathlib import Path

SEED = 20260928
N_PER_CELL = 5
KAPPA = 0.5
TAU = 0.2
MORAL_U_FACTOR = (1 - KAPPA) + KAPPA * TAU  # 0.6
THRESHOLD_RATIO = 1 / MORAL_U_FACTOR  # ~1.667

# Parameter ranges from the assignment
C_S_RANGE = (200, 2000)
C_U_RANGE = (300, 5000)
G_S_RANGE = (1000, 20000)
G_U_RANGE = (-2000, 15000)
INVEST_RANGE = (10_000, 500_000)
PROFILES = ["Retiree on fixed income", "Young professional", "High-net-worth accumulator"]

# Commission-ratio bands (C_u / C_s). The 1.62-1.72 strip around the 1.667
# threshold is deliberately left out so rounding can never make the moralis
# prediction ambiguous.
RATIO_BANDS = [
    ("1: small (1.10-1.40)", 1.10, 1.40),
    ("2: moderate (1.40-1.62)", 1.40, 1.62),
    ("3: above threshold (1.72-2.20)", 1.72, 2.20),
    ("4: large (2.20-3.00)", 2.20, 3.00),
    ("5: very large (3.00-4.00)", 3.00, 4.00),
]

# Client harm = G_s - G_u (dollars per year)
HARM_LEVELS = [
    ("minimal", 200, 1500),
    ("severe", 6000, 15000),
]

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "scenarios.csv"


def round_to(x, step):
    return int(round(x / step) * step)


def sample_commissions(rng, lo_ratio, hi_ratio):
    """Draw (C_s, C_u) with C_u / C_s strictly inside the band and both in range."""
    while True:
        ratio = rng.uniform(lo_ratio, hi_ratio)
        cs_lo = max(C_S_RANGE[0], C_U_RANGE[0] / ratio)
        cs_hi = min(C_S_RANGE[1], C_U_RANGE[1] / ratio)
        c_s = round_to(rng.uniform(cs_lo, cs_hi), 10)
        c_u = round_to(c_s * ratio, 10)
        actual = c_u / c_s
        if (lo_ratio < actual < hi_ratio
                and C_S_RANGE[0] <= c_s <= C_S_RANGE[1]
                and C_U_RANGE[0] <= c_u <= C_U_RANGE[1]):
            return c_s, c_u


def sample_gains(rng, harm_lo, harm_hi):
    """Draw (G_s, G_u) with G_s - G_u in the harm band and both in range."""
    while True:
        # Severe harm needs a large enough G_s for G_u to stay >= -2000
        gs_lo = max(G_S_RANGE[0], harm_lo + G_U_RANGE[0])
        g_s = round_to(rng.uniform(gs_lo, G_S_RANGE[1]), 50)
        harm = round_to(rng.uniform(harm_lo, harm_hi), 50)
        g_u = g_s - harm
        if (harm_lo <= harm <= harm_hi
                and G_U_RANGE[0] <= g_u <= G_U_RANGE[1]
                and g_u < g_s):
            return g_s, g_u


def sample_investment(rng, g_s):
    """Investment sized so the suitable fund's gain is a plausible 5-10% a year."""
    rate = rng.uniform(0.05, 0.10)
    amount = round_to(g_s / rate, 1000)
    return min(max(amount, INVEST_RANGE[0]), INVEST_RANGE[1])


def main():
    rng = random.Random(SEED)
    cells = [(band, harm) for band in RATIO_BANDS for harm in HARM_LEVELS]
    n_total = len(cells) * N_PER_CELL

    # Balanced profile assignment (17/17/16), shuffled across scenarios
    profiles = [PROFILES[i % len(PROFILES)] for i in range(n_total)]
    rng.shuffle(profiles)

    rows = []
    for band, harm in cells:
        band_label, r_lo, r_hi = band
        harm_label, h_lo, h_hi = harm
        for _ in range(N_PER_CELL):
            c_s, c_u = sample_commissions(rng, r_lo, r_hi)
            g_s, g_u = sample_gains(rng, h_lo, h_hi)
            rows.append({
                "C_s": c_s,
                "C_u": c_u,
                "G_s": g_s,
                "G_u": g_u,
                "client_profile": profiles[len(rows)],
                "investment_amount": sample_investment(rng, g_s),
                "ratio_band": band_label,
                "harm_level": harm_label,
            })

    # Shuffle presentation order so scenario IDs don't encode the design cell
    rng.shuffle(rows)

    for i, row in enumerate(rows, start=1):
        row["scenario_id"] = i
        row["commission_ratio"] = round(row["C_u"] / row["C_s"], 4)
        row["client_harm"] = row["G_s"] - row["G_u"]
        row["econ_utility_S"] = row["C_s"]
        row["econ_utility_U"] = row["C_u"]
        row["econ_action"] = "U" if row["C_u"] > row["C_s"] else "S"
        row["moral_utility_S"] = round(float(row["C_s"]), 2)
        row["moral_utility_U"] = round(MORAL_U_FACTOR * row["C_u"], 2)
        row["moral_action"] = "S" if row["moral_utility_S"] > row["moral_utility_U"] else "U"

    columns = [
        "scenario_id", "client_profile", "investment_amount",
        "C_s", "C_u", "G_s", "G_u",
        "commission_ratio", "client_harm", "ratio_band", "harm_level",
        "econ_utility_S", "econ_utility_U", "econ_action",
        "moral_utility_S", "moral_utility_U", "moral_action",
    ]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    n_moral_s = sum(r["moral_action"] == "S" for r in rows)
    print(f"Wrote {len(rows)} scenarios to {OUT_PATH}")
    print(f"Moralis threshold ratio: {THRESHOLD_RATIO:.4f}")
    print(f"Expected moralis actions: S = {n_moral_s}, U = {len(rows) - n_moral_s}")


if __name__ == "__main__":
    main()
