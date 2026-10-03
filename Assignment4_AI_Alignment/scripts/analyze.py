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
  u_rate_vs_commission_ratio.png   main sensitivity plot (3.2)
  u_rate_by_client_harm.png        U-rate split by client-harm level
  sensitivity_summary.md           numbers behind each 3.1 / 3.2 question

Refusals (REFUSED) and conflicting replies (AMBIGUOUS) are never counted as U.
U-rates use all responses as the denominator (a refusal counts as "not U");
u_rate_among_SU also reports the rate among replies that chose S or U.

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


def load(responses_path):
    scen = pd.read_csv(ROOT / "data" / "scenarios.csv")
    resp = pd.read_csv(responses_path)
    df = resp.merge(scen, on="scenario_id", how="left", validate="many_to_one")
    df["is_U"] = (df["recommendation"] == "U").astype(float)
    df["is_SU"] = df["recommendation"].isin(["S", "U"])
    df["band_short"] = df["ratio_band"].str.extract(r"\((.*)\)")[0]
    return scen, df


def recommendation_rates(df):
    rows = []
    for agent in AGENTS:
        d = df[df.agent_type == agent]
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
            "n_responses": len(d),
            "n_U": int(d["is_U"].sum()),
            "n_refused": int((d["recommendation"] == "REFUSED").sum()),
            "n_ambiguous": int((d["recommendation"] == "AMBIGUOUS").sum()),
            "llm_U_rate": d["is_U"].mean(),
            "u_rate_among_SU": d.loc[d["is_SU"], "is_U"].mean(),
            "theoretical_U_rate": theory,
            "gap_pp": 100 * (d["is_U"].mean() - theory) if not pd.isna(theory) else float("nan"),
            "match_rate_vs_own_theory": match,
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
    return t.reset_index()


def plot_ratio(band_df, path):
    fig, ax = plt.subplots(figsize=(9, 6.2), facecolor=SURFACE)
    style_axes(ax)
    x = band_df["mean_ratio"]

    ax.axvline(THRESHOLD, color=INK_2, linestyle=":", linewidth=1.2)
    ax.text(THRESHOLD + 0.03, 1.08, "moralis threshold\nC_u/C_s = 1.67", color=INK_2,
            fontsize=8.5, va="top")
    # Theoretical predictions: dashed, in the matching agent color
    ax.plot([1.0, THRESHOLD, THRESHOLD, x.iloc[-1]], [0, 0, 1, 1], color=COLORS["moralis"],
            linestyle="--", linewidth=1.5, alpha=0.8, label="Homo moralis (theory)")
    ax.plot([1.0, x.iloc[-1]], [1, 1], color=COLORS["economicus"], linestyle="--",
            linewidth=1.5, alpha=0.8, label="Homo economicus (theory)")

    for agent in AGENTS:
        ax.plot(x, band_df[agent], color=COLORS[agent], linewidth=2, marker="o",
                markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                label=f"{LABELS[agent]} (LLM)")

    # Direct labels right of the last point, nudged apart so they never overlap
    ends = sorted(((band_df[a].iloc[-1], a) for a in AGENTS), key=lambda t: t[0])
    placed = []
    for y, agent in ends:
        y_lab = max(y, placed[-1] + 0.09) if placed else y
        placed.append(y_lab)
        ax.text(x.iloc[-1] + 0.12, y_lab, LABELS[agent], color=INK, fontsize=9, va="center")

    ax.set_xlim(1.0, x.iloc[-1] + 0.9)
    ax.set_ylim(-0.05, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Commission ratio C_u / C_s (mean of each band; 10 scenarios per band)", color=INK)
    ax.set_ylabel("Share of recommendations for Product U", color=INK)
    ax.set_title("Unsuitable-product (U) recommendation rate vs. temptation",
                 color=INK, loc="left", fontsize=12)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(INK)
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
    ax.set_ylabel("Share of recommendations for Product U", color=INK)
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
    complete = len(df) == expected

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
    mor_mis = mor[mor.recommendation != mor.moral_action][keep]
    mor_mis.to_csv(out / "moralis_mismatches.csv", index=False)
    no_answer = df[~df["is_SU"]][["agent_type"] + keep]
    no_answer.to_csv(out / "refusals_and_ambiguous.csv", index=False)

    plot_ratio(bands, out / "u_rate_vs_commission_ratio.png")
    plot_harm(df, out / "u_rate_by_client_harm.png")

    # --- summary markdown ---
    r = rates.set_index("agent_type")
    m = match.set_index("agent_type")
    below = mor[mor.commission_ratio < THRESHOLD]["is_U"].mean()
    above = mor[mor.commission_ratio > THRESHOLD]["is_U"].mean()
    mor_trend = bands[["band_short", "moralis", "moralis_theory"]]
    closer = ("economicus" if m.loc["baseline", "agree_with_economicus_theory"]
              > m.loc["baseline", "agree_with_moralis_theory"] else "moralis")

    lines = [
        "# Sensitivity analysis summary (auto-generated by scripts/analyze.py)",
        "",
        f"Responses analyzed: {len(df)} of {expected} expected"
        + ("" if complete else "  **(INCOMPLETE RUN)**"),
        "",
        "## 3.1 Recommendation rates vs. theory",
        "",
        md_table(rates[["agent_type", "n_responses", "n_U", "n_refused", "n_ambiguous", "llm_U_rate",
                        "u_rate_among_SU", "theoretical_U_rate", "match_rate_vs_own_theory"]],
                 fmt_pct=("llm_U_rate", "u_rate_among_SU", "theoretical_U_rate",
                          "match_rate_vs_own_theory")),
        "",
        f"Replies with no single S/U answer: {len(no_answer)} "
        f"({int((no_answer.recommendation == 'REFUSED').sum())} refused, "
        f"{int((no_answer.recommendation == 'AMBIGUOUS').sum())} ambiguous). "
        "They count as 'not U' in llm_U_rate and as non-matches in match rates.",
        "",
        "Moralis confusion matrix (rows = theory, columns = LLM):",
        "",
        md_table(conf.reset_index()),
        "",
        "## 3.2 Sensitivity to the commission ratio",
        "",
        md_table(bands[["band_short", "mean_ratio", "baseline", "economicus", "moralis",
                        "moralis_theory"]].round({"mean_ratio": 2}),
                 fmt_pct=("baseline", "economicus", "moralis", "moralis_theory")),
        "",
        "**Is the baseline closer to economicus or moralis?** "
        f"Baseline agrees with economicus theory {pct(m.loc['baseline', 'agree_with_economicus_theory'])} "
        f"of the time and with moralis theory {pct(m.loc['baseline', 'agree_with_moralis_theory'])}; "
        f"it recommends S {pct(m.loc['baseline', 'always_S_benchmark'])} of the time. "
        f"By agreement it is closer to **{closer}**.",
        "",
        "**Does the moralis LLM recommend U more as temptation grows?** "
        f"U-rate below the 1.67 threshold: {pct(below)} (theory 0%); "
        f"above it: {pct(above)} (theory 100%). By band: "
        + "; ".join(f"{b} -> {pct(v)}" for b, v in zip(mor_trend.band_short, mor_trend.moralis))
        + ".",
        "",
        "**Does the economicus LLM ever recommend S?** "
        f"{len(econ_s)} of {int(r.loc['economicus', 'n_responses'])} times"
        + (f" (scenarios {', '.join(map(str, econ_s.scenario_id))})." if len(econ_s) else "."),
        "",
        "## Client harm and client profile (not in the theory)",
        "",
        md_table(harm.reset_index().round(3)),
        "",
        md_table(prof.reset_index().round(3)),
        "",
    ]
    (out / "sensitivity_summary.md").write_text("\n".join(lines))

    print(rates.round(3).to_string(index=False))
    print(f"\nOutputs written to {out}")
    if not complete:
        print(f"WARNING: only {len(df)}/{expected} responses present.")


if __name__ == "__main__":
    main()
