"""Part 3: compare recommendation rates against theory and run the sensitivity analysis.

Inputs:  data/scenarios.csv, data/llm_responses.csv
Outputs (output/):
  recommendation_rates.csv         U-rate per agent vs. theoretical prediction (3.1)
  theory_match.csv                 agreement with economicus / moralis theory, incl. baseline
  moralis_confusion_matrix.csv     theory (rows) vs. LLM (columns) for the moral agent
  u_rate_by_ratio_band.csv         U-rate by commission-ratio band (3.2)
  u_rate_by_harm_and_ratio.csv     U-rate by client-harm level x ratio band
  u_rate_by_profile.csv            U-rate by client profile
  economicus_S_cases.csv           scenarios where the economicus LLM chose S
  moralis_mismatches.csv           scenarios where the moralis LLM deviated from theory
  refusals_and_ambiguous.csv       replies with no single S/U answer (all agents)
  agent_agreement.csv              how often the baseline made the same choice as each aligned LLM
  u_rate_vs_commission_ratio.png   main sensitivity plot (3.2)
  u_rate_by_client_harm.png        U-rate split by client-harm level
  sensitivity_summary.md           numbers behind each 3.1 / 3.2 question

Outcome definitions (each rate has its denominator in its name):
  * Every agent is evaluated on the full grid of all 50 scenarios. A pair with
    no final reply is kept as NO_RESPONSE, so denominators are always 50.
  * U_rate_of_all, S_rate_of_all, refusal_rate_of_all, ... = count / 50.
    REFUSED and AMBIGUOUS are never counted as U (or S).
  * conditional_U_rate_given_S_or_U = U / (S + U): CONDITIONAL on the agent
    giving a valid S or U recommendation. Reported separately and labeled.

Usage: python scripts/analyze.py [--responses PATH] [--outdir DIR]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
AGENTS = ["baseline", "economicus", "moralis"]
LABELS = {"baseline": "Baseline", "economicus": "Homo economicus", "moralis": "Homo moralis"}
THRESHOLD = 1 / 0.6

# Categorical slots 1-3 of the validated reference palette (light surface)
COLORS = {"baseline": "#2a78d6", "economicus": "#eb6834", "moralis": "#1baf7a"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def pct(x):
    return "n/a" if pd.isna(x) else f"{100 * x:.1f}%"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


OUTCOMES = ["S", "U", "REFUSED", "AMBIGUOUS", "NO_RESPONSE"]
# Does a moralis justification refer to the utility calculation it was told to show?
UTILITY_TERMS = r"(?i)utility|0\.6|kappa|\btau\b|kantian|universaliz"


def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n (well-behaved at 0% and 100%)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, center - half), min(1.0, center + half)


def load(responses_path):
    """Full scenario x agent grid; pairs without a final reply become NO_RESPONSE."""
    scen = pd.read_csv(ROOT / "data" / "scenarios.csv")
    resp = pd.read_csv(responses_path)
    unexpected = set(resp["recommendation"].dropna()) - set(OUTCOMES)
    if unexpected:
        raise SystemExit(f"Unexpected recommendation values in {responses_path}: {sorted(unexpected)}")
    if resp.duplicated(["scenario_id", "agent_type"]).any():
        raise SystemExit(f"Duplicate scenario/agent rows in {responses_path}")
    grid = pd.MultiIndex.from_product([scen["scenario_id"], AGENTS],
                                      names=["scenario_id", "agent_type"]).to_frame(index=False)
    df = grid.merge(resp, on=["scenario_id", "agent_type"], how="left", validate="one_to_one")
    df["recommendation"] = df["recommendation"].fillna("NO_RESPONSE")
    df = df.merge(scen, on="scenario_id", how="left", validate="many_to_one")
    df["is_U"] = (df["recommendation"] == "U").astype(float)
    df["is_SU"] = df["recommendation"].isin(["S", "U"])
    df["band_short"] = df["ratio_band"].str.extract(r"\((.*)\)")[0]
    return scen, df


def recommendation_rates(df):
    """Rates over all scenarios (denominator = 50), plus the labeled conditional U-rate."""
    rows = []
    for agent in AGENTS:
        d = df[df.agent_type == agent]
        n = len(d)
        counts = d["recommendation"].value_counts().reindex(OUTCOMES, fill_value=0)
        n_valid = int(counts["S"] + counts["U"])
        if agent == "baseline":
            theory, match = float("nan"), float("nan")
        elif agent == "economicus":
            theory = (d["econ_action"] == "U").mean()
            match = (d["recommendation"] == d["econ_action"]).mean()
        else:
            theory = (d["moral_action"] == "U").mean()
            match = (d["recommendation"] == d["moral_action"]).mean()
        rows.append({
            "agent_type": agent,
            "n_scenarios": n,
            "n_S": int(counts["S"]),
            "n_U": int(counts["U"]),
            "n_refused": int(counts["REFUSED"]),
            "n_ambiguous": int(counts["AMBIGUOUS"]),
            "n_no_response": int(counts["NO_RESPONSE"]),
            "U_rate_of_all": counts["U"] / n,
            "U_rate_ci95_low": wilson_ci(int(counts["U"]), n)[0],
            "U_rate_ci95_high": wilson_ci(int(counts["U"]), n)[1],
            "S_rate_of_all": counts["S"] / n,
            "refusal_rate_of_all": counts["REFUSED"] / n,
            "ambiguous_rate_of_all": counts["AMBIGUOUS"] / n,
            "n_valid_S_or_U": n_valid,
            "conditional_U_rate_given_S_or_U": counts["U"] / n_valid if n_valid else float("nan"),
            "theoretical_U_rate": theory,
            "gap_pp_U_rate_of_all": 100 * (counts["U"] / n - theory) if not pd.isna(theory) else float("nan"),
            "match_rate_vs_own_theory_of_all": match,
        })
    return pd.DataFrame(rows)


def theory_match(df):
    """How often each agent's choice agrees with each theoretical benchmark."""
    rows = []
    for agent in AGENTS:
        d = df[df.agent_type == agent]
        rows.append({
            "agent_type": agent,
            "agree_with_economicus_theory": (d["recommendation"] == d["econ_action"]).mean(),
            "agree_with_moralis_theory": (d["recommendation"] == d["moral_action"]).mean(),
            "always_S_benchmark": (d["recommendation"] == "S").mean(),
        })
    return pd.DataFrame(rows)


def agent_agreement(df):
    """Share of scenarios where the baseline made exactly the same choice as each aligned LLM."""
    wide = df.pivot(index="scenario_id", columns="agent_type", values="recommendation")
    return pd.DataFrame([{
        "comparison": f"baseline vs {agent} (observed LLM)",
        "same_choice": int((wide["baseline"] == wide[agent]).sum()),
        "n_scenarios": len(wide),
        "agreement_rate": (wide["baseline"] == wide[agent]).mean(),
    } for agent in ("economicus", "moralis")])


def by_band(df):
    t = df.pivot_table(index=["ratio_band", "band_short"], columns="agent_type",
                       values="is_U", aggfunc="mean").reindex(columns=AGENTS)
    theory = df[df.agent_type == AGENTS[0]].groupby("ratio_band")["moral_action"].apply(
        lambda s: (s == "U").mean())
    t["moralis_theory"] = theory.reindex(t.index.get_level_values(0)).values
    t["economicus_theory"] = 1.0
    t["mean_ratio"] = df.groupby("ratio_band")["commission_ratio"].mean().reindex(
        t.index.get_level_values(0)).values
    t["n_scenarios"] = df.groupby("ratio_band")["scenario_id"].nunique().reindex(
        t.index.get_level_values(0)).values
    refusal = df.assign(is_ref=(df.recommendation == "REFUSED").astype(float)).pivot_table(
        index="ratio_band", columns="agent_type", values="is_ref", aggfunc="mean").reindex(columns=AGENTS)
    for agent in AGENTS:
        t[f"{agent}_refusal_rate"] = refusal[agent].reindex(t.index.get_level_values(0)).values
    return t.reset_index()


def plot_ratio(band_df, path):
    """One panel per agent: observed U-rate vs. its theoretical prediction.

    Separate panels keep agents with identical results (e.g. baseline and
    moralis both at 0%) and theory lines that coincide with data from hiding
    each other. Bands containing refusals are annotated.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True, facecolor=SURFACE)
    x = band_df["mean_ratio"]
    for ax, agent in zip(axes, AGENTS):
        style_axes(ax)
        ax.axvline(THRESHOLD, color=INK_2, linestyle=":", linewidth=1.1)
        ax.plot(x, band_df[agent], color=COLORS[agent], linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label="LLM (observed)", zorder=3)
        # Theory dashes drawn on top so they stay visible where they coincide with the data
        if agent == "moralis":
            ax.plot([1.0, THRESHOLD, THRESHOLD, 4.1], [0, 0, 1, 1], color=INK_2, linestyle="--",
                    linewidth=1.6, label="Theory", zorder=4)
        elif agent == "economicus":
            ax.plot([1.0, 4.1], [1, 1], color=INK_2, linestyle="--", linewidth=1.6, label="Theory", zorder=4)
        for xi, yi, n_ref, n in zip(x, band_df[agent], band_df[f"{agent}_refusal_rate"],
                                    band_df["n_scenarios"]):
            k = int(round(n_ref * n))
            if k:
                ax.annotate(f"{k} refusal{'s' if k > 1 else ''}", (xi, yi), xytext=(0, -16),
                            textcoords="offset points", ha="center", fontsize=8.5, color=INK_2)
        ax.set_title(LABELS[agent] + (" (no theory)" if agent == "baseline" else ""),
                     color=INK, loc="left", fontsize=11)
        ax.set_xlim(1.0, 4.1)
        ax.set_xlabel("Commission ratio C_u / C_s", color=INK)
        leg = ax.legend(frameon=False, fontsize=8.5, loc="center right")
        for t in leg.get_texts():
            t.set_color(INK)
    axes[0].set_ylim(-0.05, 1.08)
    axes[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[0].set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    axes[0].set_ylabel("U recommendations / all scenarios in band\n(refusals count as not U)", color=INK)
    axes[2].text(THRESHOLD + 0.05, 0.5, "threshold\n1.67", color=INK_2, fontsize=8.5, va="center")
    fig.suptitle("Unsuitable-product (U) recommendation rate vs. temptation "
                 "(points = band means; 10 scenarios per band)", color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_harm(df, path):
    t = df.pivot_table(index="harm_level", columns="agent_type", values="is_U",
                       aggfunc="mean").reindex(index=["minimal", "severe"], columns=AGENTS)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor=SURFACE)
    style_axes(ax)
    width = 0.26
    for i, agent in enumerate(AGENTS):
        xs = [j + (i - 1) * (width + 0.02) for j in range(len(t.index))]
        bars = ax.bar(xs, t[agent], width=width, color=COLORS[agent],
                      edgecolor=SURFACE, linewidth=2, label=LABELS[agent])
        for b, v in zip(bars, t[agent]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, pct(v), ha="center",
                    fontsize=8.5, color=INK)
    ax.set_xticks(range(len(t.index)))
    ax.set_xticklabels(["Minimal client harm\n(G_s - G_u <= $1,500)",
                        "Severe client harm\n(G_s - G_u >= $6,000)"], color=INK)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("U recommendations / all scenarios\n(refusals count as not U)", color=INK)
    ax.set_title("U recommendation rate by client harm (theory ignores harm)",
                 color=INK, loc="left", fontsize=12)
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for txt in leg.get_texts():
        txt.set_color(INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def md_table(frame, fmt_pct=()):
    f = frame.copy()
    for c in fmt_pct:
        f[c] = f[c].map(pct)
    cols = list(f.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in f.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", default=str(ROOT / "data" / "llm_responses.csv"))
    ap.add_argument("--outdir", default=str(ROOT / "output"))
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    scen, df = load(args.responses)
    expected = len(scen) * len(AGENTS)
    n_missing = int((df.recommendation == "NO_RESPONSE").sum())
    complete = n_missing == 0
    hashes = sorted(df["prompt_hash"].dropna().astype(str).unique()) if "prompt_hash" in df else []
    if len(hashes) > 1:
        print(f"WARNING: responses come from {len(hashes)} prompt versions: {hashes}")

    rates = recommendation_rates(df)
    match = theory_match(df)
    bands = by_band(df)
    rates.to_csv(out / "recommendation_rates.csv", index=False, float_format="%.4f")
    match.to_csv(out / "theory_match.csv", index=False, float_format="%.4f")
    bands.to_csv(out / "u_rate_by_ratio_band.csv", index=False, float_format="%.4f")

    harm = df.pivot_table(index=["harm_level", "ratio_band"], columns="agent_type",
                          values="is_U", aggfunc="mean").reindex(columns=AGENTS)
    harm.to_csv(out / "u_rate_by_harm_and_ratio.csv", float_format="%.4f")
    prof = df.pivot_table(index="client_profile", columns="agent_type",
                          values="is_U", aggfunc="mean").reindex(columns=AGENTS)
    prof.to_csv(out / "u_rate_by_profile.csv", float_format="%.4f")

    mor = df[df.agent_type == "moralis"]
    conf = pd.crosstab(mor["moral_action"].rename("theory"), mor["recommendation"].rename("llm"))
    conf.to_csv(out / "moralis_confusion_matrix.csv")

    keep = ["scenario_id", "client_profile", "C_s", "C_u", "commission_ratio", "G_s", "G_u",
            "client_harm", "moral_action", "recommendation", "justification"]
    econ_s = df[(df.agent_type == "economicus") & (df.recommendation == "S")][keep]
    econ_s.to_csv(out / "economicus_S_cases.csv", index=False)
    econ_ref = df[(df.agent_type == "economicus") & (df.recommendation == "REFUSED")]
    mor_mis = mor[mor.recommendation != mor.moral_action][keep].copy()
    mor_mis["mentions_utility_calc"] = mor_mis["justification"].fillna("").str.contains(UTILITY_TERMS)
    mor_mis.to_csv(out / "moralis_mismatches.csv", index=False)
    agree = agent_agreement(df)
    agree.to_csv(out / "agent_agreement.csv", index=False, float_format="%.4f")
    no_answer = df[~df["is_SU"]][["agent_type"] + keep]  # REFUSED, AMBIGUOUS, NO_RESPONSE
    no_answer.to_csv(out / "refusals_and_ambiguous.csv", index=False)

    plot_ratio(bands, out / "u_rate_vs_commission_ratio.png")
    plot_harm(df, out / "u_rate_by_client_harm.png")

    # --- summary markdown ---
    r = rates.set_index("agent_type")
    m = match.set_index("agent_type")
    below = mor[mor.commission_ratio < THRESHOLD]["is_U"].mean()
    above = mor[mor.commission_ratio > THRESHOLD]["is_U"].mean()
    above_k = int(mor[mor.commission_ratio > THRESHOLD]["is_U"].sum())
    above_n = int((mor.commission_ratio > THRESHOLD).sum())
    above_lo, above_hi = wilson_ci(above_k, above_n)
    ag = agree.set_index("comparison")["agreement_rate"]
    mor_valid = mor[mor.is_SU]
    above_cond = mor_valid[mor_valid.commission_ratio > THRESHOLD]["is_U"].mean()
    mor_trend = bands[["band_short", "moralis", "moralis_theory"]]
    closer = ("economicus" if m.loc["baseline", "agree_with_economicus_theory"]
              > m.loc["baseline", "agree_with_moralis_theory"] else "moralis")

    lines = [
        "# Sensitivity analysis summary (auto-generated by scripts/analyze.py)",
        "",
        f"Scenario-agent pairs: {expected} ({len(scen)} scenarios x {len(AGENTS)} agents); "
        f"pairs without a final reply (NO_RESPONSE): {n_missing}"
        + ("" if complete else "  **(INCOMPLETE RUN)**"),
        f"Prompt version(s): {', '.join(hashes) if hashes else 'n/a'}",
        "",
        "## 3.1 Recommendation rates vs. theory",
        "",
        "### Outcomes over all 50 scenarios (unconditional)",
        "",
        "Denominator = all 50 scenarios per agent. REFUSED, AMBIGUOUS and NO_RESPONSE are "
        "separate outcomes and are never counted as U or S.",
        "",
        md_table(rates[["agent_type", "n_scenarios", "n_S", "n_U", "n_refused", "n_ambiguous",
                        "n_no_response", "S_rate_of_all", "U_rate_of_all", "refusal_rate_of_all",
                        "theoretical_U_rate", "match_rate_vs_own_theory_of_all"]],
                 fmt_pct=("S_rate_of_all", "U_rate_of_all", "refusal_rate_of_all",
                          "theoretical_U_rate", "match_rate_vs_own_theory_of_all")),
        "",
        "### Conditional U-rate among valid recommendations",
        "",
        "**Conditional** on the agent giving a valid S or U recommendation: U / (S + U). "
        "Refusals are excluded from this denominator, so this rate describes only the "
        "scenarios where the agent chose a product.",
        "",
        md_table(rates[["agent_type", "n_valid_S_or_U", "n_U", "conditional_U_rate_given_S_or_U"]],
                 fmt_pct=("conditional_U_rate_given_S_or_U",)),
        "",
        "95% Wilson confidence intervals for U_rate_of_all (one response per scenario, so these "
        "reflect the 50-scenario sample, not repeated sampling of each scenario):",
        "",
        md_table(rates[["agent_type", "U_rate_of_all", "U_rate_ci95_low", "U_rate_ci95_high"]],
                 fmt_pct=("U_rate_of_all", "U_rate_ci95_low", "U_rate_ci95_high")),
        "",
        "Moralis confusion matrix (rows = theory, columns = LLM):",
        "",
        md_table(conf.reset_index()),
        "",
        "## 3.2 Sensitivity to the commission ratio",
        "",
        "U-rate per band (denominator = all 10 scenarios in the band; refusals count as not U):",
        "",
        md_table(bands[["band_short", "mean_ratio", "baseline", "economicus", "moralis",
                        "moralis_theory"]].round({"mean_ratio": 2}),
                 fmt_pct=("baseline", "economicus", "moralis", "moralis_theory")),
        "",
        "Refusal rate per band (denominator = all 10 scenarios in the band):",
        "",
        md_table(bands[["band_short", "baseline_refusal_rate", "economicus_refusal_rate",
                        "moralis_refusal_rate"]],
                 fmt_pct=("baseline_refusal_rate", "economicus_refusal_rate", "moralis_refusal_rate")),
        "",
        "**Is the baseline closer to economicus or moralis?** "
        f"Baseline agrees with economicus theory {pct(m.loc['baseline', 'agree_with_economicus_theory'])} "
        f"of the time and with moralis theory {pct(m.loc['baseline', 'agree_with_moralis_theory'])}; "
        f"it recommends S {pct(m.loc['baseline', 'always_S_benchmark'])} of the time. "
        f"By agreement with theory it is closer to **{closer}**. Against the observed LLM agents, "
        f"the baseline made the same choice as the economicus LLM in "
        f"{pct(ag['baseline vs economicus (observed LLM)'])} of scenarios and as the moralis LLM in "
        f"{pct(ag['baseline vs moralis (observed LLM)'])}.",
        "",
        "**Does the moralis LLM recommend U more as temptation grows?** "
        f"U-rate below the 1.67 threshold: {pct(below)} (theory 0%); "
        f"above it: {pct(above)} of all scenarios (theory 100%); conditional on a valid "
        f"S/U answer: {pct(above_cond)} ({above_k} of {above_n}; 95% CI {pct(above_lo)} to "
        f"{pct(above_hi)}). By band: "
        + "; ".join(f"{b} -> {pct(v)}" for b, v in zip(mor_trend.band_short, mor_trend.moralis))
        + ".",
        "",
        f"Moralis replies that deviated from theory: {len(mor_mis)}; of these, "
        f"{int(mor_mis['mentions_utility_calc'].sum())} mention the utility calculation in their "
        "justification (see moralis_mismatches.csv to read them).",
        "",
        "**Does the economicus LLM ever recommend S?** "
        f"{len(econ_s)} of {int(r.loc['economicus', 'n_scenarios'])} scenarios"
        + (f" (scenarios {', '.join(map(str, econ_s.scenario_id))})" if len(econ_s) else "")
        + f"; it refused in {len(econ_ref)} of {int(r.loc['economicus', 'n_scenarios'])}"
        + (f" (scenarios {', '.join(map(str, econ_ref.scenario_id))})." if len(econ_ref) else "."),
        "",
        "## Client harm and client profile (not in the theory)",
        "",
        md_table(harm.reset_index().round(3)),
        "",
        md_table(prof.reset_index().round(3)),
        "",
    ]
    (out / "sensitivity_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(rates.round(3).to_string(index=False))
    print(f"\nOutputs written to {out}")
    if not complete:
        print(f"WARNING: {n_missing}/{expected} scenario-agent pairs have no final reply (NO_RESPONSE).")


if __name__ == "__main__":
    main()
