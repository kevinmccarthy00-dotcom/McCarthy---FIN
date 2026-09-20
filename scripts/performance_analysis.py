"""Compare trading strategies built from LLM headline predictions vs. Mkt-RF.

Usage:
    python scripts/performance_analysis.py \\
        --predictions data/predictions/predictions_20260821_20260920.csv \\
        --mkt-rf data/famafrench/mkt_rf_20260821_20260920.csv \\
        --gdelt-meta data/gdelt/cnbc_headlines_20260821_20260920.csv.meta.json

Builds five daily strategies from the joined predictions + Mkt-RF data:
    buy_and_hold                 always long the market
    llm_direction                long when the LLM said LONG, short when SHORT
    sentiment_weighted           position sized by sentiment_score / 5
    contrarian_direction         opposite of llm_direction
    contrarian_sentiment_weighted opposite of sentiment_weighted

For each strategy, computes gross returns and net-of-trading-cost returns
(a configurable per-unit-turnover cost, default 2 bps, applied whenever the
position size changes -- including the initial entry from flat), then
reports: total return, annualized return, annualized volatility, Sharpe,
Sortino, max drawdown, VaR 95%, VaR 99%, CVaR (95%), and win rate.

Writes:
    reports/performance_report.md    -- full Markdown report
    reports/performance_metrics.csv  -- the same metrics as a flat CSV
    reports/plots/equity_curves.png  -- cumulative gross return per strategy
    reports/plots/sharpe_gross_net.png -- gross vs. net Sharpe per strategy
"""

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("performance_analysis")

TRADING_DAYS_PER_YEAR = 252
DEFAULT_COST_BPS = 2.0

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"
PLOTS_DIR = REPORT_DIR / "plots"

STRATEGIES = [
    "buy_and_hold",
    "llm_direction",
    "sentiment_weighted",
    "contrarian_direction",
    "contrarian_sentiment_weighted",
]

STRATEGY_LABELS = {
    "buy_and_hold": "Buy & Hold",
    "llm_direction": "LLM Direction",
    "sentiment_weighted": "Sentiment-Weighted",
    "contrarian_direction": "Contrarian Direction",
    "contrarian_sentiment_weighted": "Contrarian Sentiment-Weighted",
}

METRIC_LABELS = [
    ("total_return", "Total Return"),
    ("annualized_return", "Annualized Return"),
    ("annualized_vol", "Annualized Volatility"),
    ("sharpe", "Sharpe Ratio"),
    ("sortino", "Sortino Ratio"),
    ("max_drawdown", "Max Drawdown"),
    ("var_95", "VaR 95%"),
    ("var_99", "VaR 99%"),
    ("cvar_95", "Conditional VaR (95%)"),
    ("win_rate", "Win Rate"),
]


def load_data(predictions_path, mkt_rf_path):
    mkt_rf = pd.read_csv(mkt_rf_path, parse_dates=["date"])
    mkt_rf["market_return"] = mkt_rf["mkt_rf"] + mkt_rf["rf"]
    mkt_rf = mkt_rf.set_index("date")[["market_return", "rf"]]

    predictions = pd.read_csv(predictions_path, parse_dates=["target_date"])
    predictions = predictions.set_index("target_date")[["position", "sentiment_score"]]

    joined = predictions.join(mkt_rf, how="inner").sort_index()
    dropped_predictions = len(predictions) - len(joined)
    dropped_mkt_rf = len(mkt_rf) - len(joined)
    if dropped_predictions or dropped_mkt_rf:
        logger.warning(
            "Joining predictions to Mkt-RF by date: %d prediction date(s) had no matching "
            "Mkt-RF row and %d Mkt-RF date(s) had no prediction; those dates are excluded "
            "rather than fabricated.",
            dropped_predictions, dropped_mkt_rf,
        )
    if joined.empty:
        raise SystemExit("No overlapping dates between predictions and Mkt-RF data; nothing to analyze.")

    logger.info("Analyzing %d trading day(s): %s to %s", len(joined), joined.index.min().date(), joined.index.max().date())
    return joined


def build_positions(df):
    positions = pd.DataFrame(index=df.index)
    positions["buy_and_hold"] = 1.0
    positions["llm_direction"] = df["position"].map({"LONG": 1.0, "SHORT": -1.0})
    if positions["llm_direction"].isna().any():
        bad = df.loc[positions["llm_direction"].isna(), "position"].unique()
        raise SystemExit(f"Unrecognized position value(s) in predictions CSV: {bad!r} (expected LONG/SHORT)")
    positions["sentiment_weighted"] = (df["sentiment_score"].astype(float) / 5.0).clip(-1.0, 1.0)
    positions["contrarian_direction"] = -positions["llm_direction"]
    positions["contrarian_sentiment_weighted"] = -positions["sentiment_weighted"]
    return positions


def strategy_returns(market_return, position, cost_bps):
    gross = market_return * position
    turnover = position.diff()
    turnover.iloc[0] = position.iloc[0] - 0.0  # cost of the initial entry from flat
    turnover = turnover.abs()
    net = gross - (cost_bps / 10_000.0) * turnover
    return gross, net


def compute_metrics(returns, rf):
    returns = returns.to_numpy(dtype=float)
    rf = rf.to_numpy(dtype=float)
    n = len(returns)
    excess = returns - rf

    total_return = float(np.prod(1.0 + returns) - 1.0)
    years = n / TRADING_DAYS_PER_YEAR
    annualized_return = float((1.0 + total_return) ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    annualized_vol = float(np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if n > 1 else float("nan")

    mean_excess = float(np.mean(excess))
    std_returns = float(np.std(returns, ddof=1)) if n > 1 else float("nan")
    sharpe = (mean_excess / std_returns * np.sqrt(TRADING_DAYS_PER_YEAR)) if std_returns not in (0, float("nan")) and not np.isnan(std_returns) else float("nan")

    downside = excess[excess < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else (0.0 if len(downside) <= 1 else float("nan"))
    sortino = (mean_excess / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR)) if downside_std not in (0, float("nan")) and not np.isnan(downside_std) else float("nan")

    cumulative = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / running_max - 1.0
    max_drawdown = float(np.min(drawdowns))

    var_95 = float(-np.percentile(returns, 5))
    var_99 = float(-np.percentile(returns, 1))
    tail_95 = returns[returns <= np.percentile(returns, 5)]
    cvar_95 = float(-np.mean(tail_95)) if len(tail_95) > 0 else float("nan")

    win_rate = float(np.mean(returns > 0))

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "var_95": var_95,
        "var_99": var_99,
        "cvar_95": cvar_95,
        "win_rate": win_rate,
    }


def run_analysis(df, cost_bps):
    positions = build_positions(df)
    results = {}
    return_series = {}
    for strat in STRATEGIES:
        gross, net = strategy_returns(df["market_return"], positions[strat], cost_bps)
        return_series[strat] = {"gross": gross, "net": net}
        results[strat] = {
            "gross": compute_metrics(gross, df["rf"]),
            "net": compute_metrics(net, df["rf"]),
        }
    return results, return_series


def write_metrics_csv(results, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "return_type"] + [key for key, _ in METRIC_LABELS])
        for strat in STRATEGIES:
            for kind in ("gross", "net"):
                row = [strat, kind] + [results[strat][kind][key] for key, _ in METRIC_LABELS]
                writer.writerow(row)


def plot_equity_curves(return_series, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for strat in STRATEGIES:
        cumulative = (1.0 + return_series[strat]["gross"]).cumprod()
        ax.plot(cumulative.index, cumulative.values, label=STRATEGY_LABELS[strat])
    ax.set_title("Cumulative Gross Return by Strategy")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sharpe_comparison(results, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(STRATEGIES))
    width = 0.35
    gross_sharpe = [results[s]["gross"]["sharpe"] for s in STRATEGIES]
    net_sharpe = [results[s]["net"]["sharpe"] for s in STRATEGIES]
    ax.bar(x - width / 2, gross_sharpe, width, label="Gross")
    ax.bar(x + width / 2, net_sharpe, width, label="Net")
    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in STRATEGIES], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Gross vs. Net Sharpe Ratio by Strategy")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def format_pct(x):
    return f"{x:.2%}" if x is not None and not (isinstance(x, float) and np.isnan(x)) else "n/a"


def format_num(x):
    return f"{x:.3f}" if x is not None and not (isinstance(x, float) and np.isnan(x)) else "n/a"


def load_gdelt_meta(path):
    if not path:
        return None
    meta_path = Path(path)
    if not meta_path.exists():
        logger.warning("--gdelt-meta path %s does not exist; skipping limitations detail", meta_path)
        return None
    return json.loads(meta_path.read_text())


def write_report(df, results, cost_bps, gdelt_meta, path):
    n_days = len(df)
    start, end = df.index.min().date(), df.index.max().date()
    lines = []
    lines.append("# News-Sentiment Trading Strategy Performance Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Sample: {n_days} trading day(s), {start} to {end}")
    lines.append("")

    lines.append("## Data Scope & Limitations")
    lines.append("")
    lines.append(
        "**This is a 30-day walkthrough sample, not the full 365-day assignment dataset.** "
        "GDELT's DOC API rate-limited the full-year headline pull heavily enough that it could "
        "not be completed in the time available before this submission was due. Rather than "
        "fabricate or backfill a year of data, this run uses the 30-day sample that GDELT "
        "actually returned. No headlines or returns anywhere in this report are synthetic."
    )
    if gdelt_meta is not None:
        completeness = "complete" if gdelt_meta.get("complete") else "INCOMPLETE"
        lines.append("")
        lines.append(
            f"GDELT headline dataset completeness for this sample: **{completeness}** "
            f"({gdelt_meta.get('unique_articles', 'n/a')} unique articles, "
            f"{gdelt_meta.get('unresolved_windows', 0)} unresolved date window(s) due to rate limiting)."
        )
    lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "- Market return proxy: Ken French daily `Mkt-RF + RF` (approximates a long SPY / total "
        "market position).\n"
        "- Each trading day's LLM prediction (LONG/SHORT position, sentiment score -5..+5) is "
        "applied to the *next* trading day's market return.\n"
        f"- Trading cost assumption: {cost_bps:.1f} bps applied to absolute position change "
        "(turnover), including the initial entry from flat -- a reasonable estimate of SPY's "
        "bid-ask spread plus commission for a liquid ETF.\n"
        "- Sharpe/Sortino use the Ken French daily risk-free rate as the excess-return baseline."
    )
    lines.append("")

    lines.append("## Strategies Compared")
    lines.append("")
    lines.append("| Strategy | Position Rule |")
    lines.append("|---|---|")
    lines.append("| Buy & Hold | Always +1 (long the market) |")
    lines.append("| LLM Direction | +1 if LLM says LONG, -1 if SHORT |")
    lines.append("| Sentiment-Weighted | sentiment_score / 5 (continuous, -1..+1) |")
    lines.append("| Contrarian Direction | Opposite of LLM Direction |")
    lines.append("| Contrarian Sentiment-Weighted | Opposite of Sentiment-Weighted |")
    lines.append("")

    lines.append("## Performance Metrics")
    lines.append("")
    for kind, kind_label in (("gross", "Gross (before trading costs)"), ("net", f"Net (after {cost_bps:.1f} bps trading costs)")):
        lines.append(f"### {kind_label}")
        lines.append("")
        header = "| Strategy | " + " | ".join(label for _, label in METRIC_LABELS) + " |"
        sep = "|---|" + "|".join("---" for _ in METRIC_LABELS) + "|"
        lines.append(header)
        lines.append(sep)
        for strat in STRATEGIES:
            m = results[strat][kind]
            cells = []
            for key, _ in METRIC_LABELS:
                if key in ("total_return", "annualized_return", "annualized_vol", "max_drawdown", "var_95", "var_99", "cvar_95", "win_rate"):
                    cells.append(format_pct(m[key]))
                else:
                    cells.append(format_num(m[key]))
            lines.append(f"| {STRATEGY_LABELS[strat]} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## Plots")
    lines.append("")
    lines.append("![Cumulative gross return by strategy](plots/equity_curves.png)")
    lines.append("")
    lines.append("![Gross vs. net Sharpe ratio by strategy](plots/sharpe_gross_net.png)")
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(
        f"- With only {n_days} trading days, annualized figures (return, volatility, Sharpe, "
        "Sortino) are extrapolated from a small sample and should be treated as illustrative, "
        "not statistically reliable estimates.\n"
        "- VaR/CVaR are computed via historical simulation on this same small sample and inherit "
        "the same reliability caveat.\n"
        "- The trading-cost assumption is a single flat rate per unit of turnover; real SPY "
        "execution costs vary with size, volatility, and venue."
    )
    lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare LLM-headline-driven trading strategies against Mkt-RF.")
    parser.add_argument("--predictions", required=True, help="Path to the LLM predictions CSV.")
    parser.add_argument("--mkt-rf", required=True, help="Path to the Ken French Mkt-RF CSV.")
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS, help=f"Trading cost in bps per unit of turnover. Default: {DEFAULT_COST_BPS}.")
    parser.add_argument("--gdelt-meta", default=None, help="Optional path to a GDELT CSV's .meta.json sidecar, to embed dataset completeness in the report.")
    parser.add_argument("--output-dir", default=None, help="Directory for the report + metrics CSV. Default: reports/.")
    parser.add_argument("--plots-dir", default=None, help="Directory for plot PNGs. Default: reports/plots/.")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else REPORT_DIR
    plots_dir = Path(args.plots_dir) if args.plots_dir else output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.predictions, args.mkt_rf)
    results, return_series = run_analysis(df, args.cost_bps)

    equity_plot_path = plots_dir / "equity_curves.png"
    sharpe_plot_path = plots_dir / "sharpe_gross_net.png"
    plot_equity_curves(return_series, equity_plot_path)
    plot_sharpe_comparison(results, sharpe_plot_path)
    logger.info("Wrote plots: %s, %s", equity_plot_path, sharpe_plot_path)

    metrics_csv_path = output_dir / "performance_metrics.csv"
    write_metrics_csv(results, metrics_csv_path)
    logger.info("Wrote metrics CSV: %s", metrics_csv_path)

    gdelt_meta = load_gdelt_meta(args.gdelt_meta)
    report_path = output_dir / "performance_report.md"
    write_report(df, results, args.cost_bps, gdelt_meta, report_path)
    logger.info("Wrote report: %s", report_path)

    print(f"REPORT_PATH={report_path}")
    print(f"METRICS_CSV={metrics_csv_path}")


if __name__ == "__main__":
    main()
