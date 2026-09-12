import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gradio as gr
from matplotlib.ticker import FuncFormatter

from portfolio_engine import (
    ASSET_CLASSES,
    RiskTolerance,
    build_lifecycle_portfolio,
    build_mvo_portfolio,
    compute_efficient_frontier,
    get_market_data,
    wealth_scenarios,
)

ASSET_NAME_BY_KEY = {a.key: a.name for a in ASSET_CLASSES}
ASSET_TICKER_BY_KEY = {a.key: a.ticker for a in ASSET_CLASSES}

METHOD_LIFECYCLE = "Lifecycle / Heuristic"
METHOD_MVO = "Mean-Variance Optimization"
RISK_TOLERANCE_OPTIONS = ["Conservative", "Moderate", "Aggressive"]
GOAL_OPTIONS = ["Retirement", "Home Purchase", "Education", "General Wealth"]

AGE_MIN, AGE_MAX = 18, 80
HORIZON_MIN, HORIZON_MAX = 1, 30
INITIAL_MIN, INITIAL_MAX = 1_000, 10_000_000
CONTRIBUTION_MIN, CONTRIBUTION_MAX = 0, 50_000

# Fetched once at startup rather than on every click - avoids hammering
# yfinance (or retrying a dead connection) on each button press. Restart
# the app to pick up fresh market data.
MARKET_DATA = get_market_data()


def _validate_inputs(age, horizon_years, initial_investment, monthly_contribution):
    if age is None or not (AGE_MIN <= age <= AGE_MAX):
        raise gr.Error(f"Age must be between {AGE_MIN} and {AGE_MAX}.")
    if horizon_years is None or not (HORIZON_MIN <= horizon_years <= HORIZON_MAX):
        raise gr.Error(
            f"Investment horizon must be between {HORIZON_MIN} and {HORIZON_MAX} years."
        )
    if initial_investment is None or not (INITIAL_MIN <= initial_investment <= INITIAL_MAX):
        raise gr.Error(
            f"Initial investment must be between ${INITIAL_MIN:,} and ${INITIAL_MAX:,}."
        )
    if monthly_contribution is None or not (
        CONTRIBUTION_MIN <= monthly_contribution <= CONTRIBUTION_MAX
    ):
        raise gr.Error(
            f"Monthly contribution must be between ${CONTRIBUTION_MIN:,} and ${CONTRIBUTION_MAX:,}."
        )


def _metrics_markdown(result, method_label):
    return "\n\n".join(
        [
            f"**Method:** {method_label}",
            f"**Expected Annual Return:** {result.expected_return:.2%}",
            f"**Expected Annual Volatility:** {result.expected_volatility:.2%}",
            f"**Sharpe Ratio:** {result.sharpe_ratio:.2f}",
        ]
    )


def _data_source_markdown(market_data):
    if market_data.source == "yfinance":
        return "📡 Market data: **live 10-year history via yfinance**."
    return (
        "⚠️ Market data: **fallback capital-market assumptions** "
        f"(yfinance was unreachable or returned unusable data: _{market_data.note}_)."
    )


def _allocation_dataframe(result):
    df = pd.DataFrame(
        {
            "Asset Class": [ASSET_NAME_BY_KEY[k] for k in result.weights.index],
            "Ticker": [ASSET_TICKER_BY_KEY[k] for k in result.weights.index],
            "Weight (%)": (result.weights.values * 100).round(1),
        }
    )
    return df.sort_values("Weight (%)", ascending=False).reset_index(drop=True)


def _comparison_dataframe(lifecycle_result, mvo_result):
    return pd.DataFrame(
        [
            {
                "Method": "Lifecycle / Heuristic",
                "Expected Return": f"{lifecycle_result.expected_return:.2%}",
                "Volatility": f"{lifecycle_result.expected_volatility:.2%}",
                "Sharpe Ratio": f"{lifecycle_result.sharpe_ratio:.2f}",
            },
            {
                "Method": "Mean-Variance Optimization",
                "Expected Return": f"{mvo_result.expected_return:.2%}",
                "Volatility": f"{mvo_result.expected_volatility:.2%}",
                "Sharpe Ratio": f"{mvo_result.sharpe_ratio:.2f}",
            },
        ]
    )


def _plot_allocation(result, title):
    fig, ax = plt.subplots(figsize=(5, 5))
    keys = list(result.weights.index)
    values = result.weights.values
    mask = values > 0.001
    labels = [ASSET_NAME_BY_KEY[k] for k, m in zip(keys, mask) if m]
    ax.pie(values[mask], labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _plot_risk_return(market_data, lifecycle_result, mvo_result):
    fig, ax = plt.subplots(figsize=(6, 5))
    vols = np.sqrt(np.diag(market_data.cov_matrix.values))
    rets = market_data.expected_returns.values

    ax.scatter(vols, rets, color="gray", label="Individual Asset Classes", zorder=2)
    for key, v, r in zip(market_data.expected_returns.index, vols, rets):
        ax.annotate(
            ASSET_TICKER_BY_KEY[key],
            (v, r),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.scatter(
        [lifecycle_result.expected_volatility],
        [lifecycle_result.expected_return],
        color="tab:blue",
        marker="*",
        s=250,
        label="Lifecycle Portfolio",
        zorder=3,
    )
    ax.scatter(
        [mvo_result.expected_volatility],
        [mvo_result.expected_return],
        color="tab:red",
        marker="*",
        s=250,
        label="Mean-Variance Portfolio",
        zorder=3,
    )

    ax.set_xlabel("Volatility (annualized std. dev.)")
    ax.set_ylabel("Expected Annual Return")
    ax.set_title("Risk vs. Return")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _plot_wealth(scenarios, goal_label):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    years = scenarios["expected"].index

    ax.fill_between(
        years,
        scenarios["pessimistic"].values,
        scenarios["optimistic"].values,
        color="gray",
        alpha=0.15,
        label="Optimistic / Pessimistic Range",
    )
    ax.plot(years, scenarios["expected"].values, color="tab:blue", label="Expected")
    ax.plot(
        years,
        scenarios["optimistic"].values,
        color="tab:green",
        linestyle="--",
        label="Optimistic (+1 SD)",
    )
    ax.plot(
        years,
        scenarios["pessimistic"].values,
        color="tab:red",
        linestyle="--",
        label="Pessimistic (-1 SD)",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Portfolio Value")
    ax.set_title(f"Projected Wealth ({goal_label})")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _plot_comparison(lifecycle_result, mvo_result):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    keys = list(lifecycle_result.weights.index)
    labels = [ASSET_NAME_BY_KEY[k] for k in keys]
    x = np.arange(len(keys))
    width = 0.35

    ax.bar(x - width / 2, lifecycle_result.weights.values * 100, width, label="Lifecycle")
    ax.bar(
        x + width / 2,
        mvo_result.weights.reindex(keys).values * 100,
        width,
        label="Mean-Variance",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Weight (%)")
    ax.set_title("Allocation Comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _plot_frontier(market_data, frontier_df, mvo_result, lifecycle_result):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        frontier_df["volatility"],
        frontier_df["expected_return"],
        color="black",
        label="Efficient Frontier",
        zorder=2,
    )

    vols = np.sqrt(np.diag(market_data.cov_matrix.values))
    rets = market_data.expected_returns.values
    ax.scatter(vols, rets, color="gray", label="Individual Asset Classes", zorder=2)
    for key, v, r in zip(market_data.expected_returns.index, vols, rets):
        ax.annotate(
            ASSET_TICKER_BY_KEY[key],
            (v, r),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.scatter(
        [mvo_result.expected_volatility],
        [mvo_result.expected_return],
        color="tab:red",
        marker="*",
        s=250,
        label="Max-Sharpe Portfolio",
        zorder=3,
    )
    ax.scatter(
        [lifecycle_result.expected_volatility],
        [lifecycle_result.expected_return],
        color="tab:blue",
        marker="*",
        s=250,
        label="Lifecycle Portfolio",
        zorder=3,
    )

    ax.set_xlabel("Volatility (annualized std. dev.)")
    ax.set_ylabel("Expected Annual Return")
    ax.set_title("Efficient Frontier (Long-Only)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def generate_portfolio(
    age,
    risk_tolerance_label,
    horizon_years,
    initial_investment,
    monthly_contribution,
    financial_goal,
    method_label,
):
    plt.close("all")  # avoid unbounded figure accumulation across clicks

    age = int(age)
    horizon_years = int(horizon_years)
    _validate_inputs(age, horizon_years, initial_investment, monthly_contribution)

    risk_tolerance = RiskTolerance(risk_tolerance_label.lower())
    market_data = MARKET_DATA

    lifecycle_result = build_lifecycle_portfolio(
        age=age,
        risk_tolerance=risk_tolerance,
        horizon_years=horizon_years,
        market_data=market_data,
    )
    mvo_result = build_mvo_portfolio(market_data=market_data)

    is_mvo = method_label == METHOD_MVO
    primary = mvo_result if is_mvo else lifecycle_result

    scenarios = wealth_scenarios(
        initial_investment,
        monthly_contribution,
        primary.expected_return,
        primary.expected_volatility,
        horizon_years,
    )

    frontier_fig = None
    if is_mvo:
        frontier_df = compute_efficient_frontier(market_data=market_data, num_points=20)
        frontier_fig = _plot_frontier(market_data, frontier_df, mvo_result, lifecycle_result)

    return (
        _metrics_markdown(primary, method_label),
        _data_source_markdown(market_data),
        _allocation_dataframe(primary),
        _plot_allocation(primary, f"{method_label} Allocation"),
        _plot_risk_return(market_data, lifecycle_result, mvo_result),
        _plot_wealth(scenarios, financial_goal),
        _comparison_dataframe(lifecycle_result, mvo_result),
        _plot_comparison(lifecycle_result, mvo_result),
        gr.update(visible=is_mvo),
        frontier_fig,
    )


with gr.Blocks(title="Robo-Advisor Portfolio Engine") as demo:
    gr.Markdown("# Robo-Advisor: Portfolio Engine (Baseline)")
    gr.Markdown(
        "Baseline allocation engine covering six asset classes via representative "
        "ETFs, with two allocation methods: an age-driven lifecycle heuristic and "
        "a long-only mean-variance optimizer. A research-informed method is planned "
        "as a future third approach."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Your Profile")
            age = gr.Slider(AGE_MIN, AGE_MAX, value=35, step=1, label="Age")
            risk_tolerance = gr.Radio(
                RISK_TOLERANCE_OPTIONS, value="Moderate", label="Risk Tolerance"
            )
            horizon_years = gr.Slider(
                HORIZON_MIN, HORIZON_MAX, value=20, step=1, label="Investment Horizon (years)"
            )
            initial_investment = gr.Number(
                value=10_000,
                minimum=INITIAL_MIN,
                maximum=INITIAL_MAX,
                step=1_000,
                label="Initial Investment ($)",
            )
            monthly_contribution = gr.Number(
                value=200,
                minimum=CONTRIBUTION_MIN,
                maximum=CONTRIBUTION_MAX,
                step=50,
                label="Monthly Contribution ($)",
            )
            financial_goal = gr.Dropdown(
                GOAL_OPTIONS, value="Retirement", label="Financial Goal"
            )
            method = gr.Radio(
                [METHOD_LIFECYCLE, METHOD_MVO],
                value=METHOD_LIFECYCLE,
                label="Allocation Method",
            )
            generate_btn = gr.Button("Generate Portfolio", variant="primary")
            data_source_md = gr.Markdown()

        with gr.Column(scale=2):
            gr.Markdown("### Recommended Portfolio")
            metrics_md = gr.Markdown()
            allocation_table = gr.Dataframe(label="Allocation", interactive=False)
            allocation_plot = gr.Plot(label="Asset Allocation")

            gr.Markdown("### Risk vs. Return")
            risk_return_plot = gr.Plot(label="Risk vs. Return")

            gr.Markdown("### Projected Wealth")
            wealth_plot = gr.Plot(label="Projected Wealth")

            gr.Markdown("### Lifecycle vs. Mean-Variance Comparison")
            comparison_table = gr.Dataframe(label="Comparison", interactive=False)
            comparison_plot = gr.Plot(label="Allocation Comparison")

            with gr.Column(visible=False) as frontier_section:
                gr.Markdown("### Efficient Frontier")
                frontier_plot = gr.Plot(label="Efficient Frontier")

    inputs = [
        age,
        risk_tolerance,
        horizon_years,
        initial_investment,
        monthly_contribution,
        financial_goal,
        method,
    ]
    outputs = [
        metrics_md,
        data_source_md,
        allocation_table,
        allocation_plot,
        risk_return_plot,
        wealth_plot,
        comparison_table,
        comparison_plot,
        frontier_section,
        frontier_plot,
    ]

    generate_btn.click(generate_portfolio, inputs=inputs, outputs=outputs)
    demo.load(generate_portfolio, inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    demo.launch(share=os.environ.get("GRADIO_SHARE") == "1")
