"""Gradio UI for the AI Financial Advisor Agent.

Lets you enter or load a client profile and run the five-stage prompt
chain, showing each stage's output in its own panel so the progression
through the chain is visible.
"""

import os

import gradio as gr

from advisor_agent import run_advisor_chain
from client_profiles import SAMPLE_PROFILE


def load_sample_profile():
    return SAMPLE_PROFILE


def run_chain(client_name, client_profile_text):
    if not client_profile_text or not client_profile_text.strip():
        error = "Please enter a client profile before running the chain."
        return {}, {}, {}, {}, {}, "", error

    name = client_name.strip() if client_name and client_name.strip() else "client"

    try:
        result = run_advisor_chain(client_profile_text, client_name=name)
    except Exception as exc:  # noqa: BLE001 - surface any failure in the UI
        error = f"Advisor chain failed: {exc}"
        return {}, {}, {}, {}, {}, "", error

    stage1 = result["stage_1_intake"]["parsed"]
    stage2 = result["stage_2_risk_profile"]["parsed"]
    raw_market_data = result["raw_market_data"]
    stage3 = result["stage_3_market_interpretation"]["parsed"]
    stage4 = result["stage_4_portfolio"]["parsed"]
    stage5_text = result["stage_5_recommendation_letter"]["text"]
    status = f"Done. Full chain output saved to: {result['saved_output_path']}"

    return stage1, stage2, raw_market_data, stage3, stage4, stage5_text, status


with gr.Blocks(title="AI Financial Advisor Agent") as demo:
    gr.Markdown("# AI Financial Advisor Agent")
    gr.Markdown(
        "Five sequential LLM stages: Intake → Risk Profiling → Market Data "
        "Interpretation → Portfolio Construction → Recommendation Letter. "
        "Each stage's full output is shown below and saved to `output/`."
    )

    with gr.Row():
        client_name_input = gr.Textbox(label="Client name (used for the saved filename)", value="client")

    client_profile_input = gr.Textbox(
        label="Client profile",
        lines=10,
        placeholder="Paste or type a free-form client profile here...",
    )

    with gr.Row():
        load_sample_btn = gr.Button("Load sample profile")
        run_btn = gr.Button("Run advisor chain", variant="primary")

    status_output = gr.Textbox(label="Status", interactive=False)

    with gr.Tab("Stage 1: Client Intake"):
        stage1_output = gr.JSON(label="Structured intake summary")
    with gr.Tab("Stage 2: Risk Profiling"):
        stage2_output = gr.JSON(label="Risk tolerance vs. capacity assessment")
    with gr.Tab("Raw Market Data (yfinance)"):
        raw_market_data_output = gr.JSON(label="Raw retrieved data (no LLM involved)")
    with gr.Tab("Stage 3: Market Data Interpretation"):
        stage3_output = gr.JSON(label="LLM-organized market data (no invented figures)")
    with gr.Tab("Stage 4: Portfolio Construction"):
        stage4_output = gr.JSON(label="Recommended allocation")
    with gr.Tab("Stage 5: Recommendation Letter"):
        stage5_output = gr.Textbox(label="Client letter", lines=25)

    load_sample_btn.click(fn=load_sample_profile, outputs=client_profile_input)
    run_btn.click(
        fn=run_chain,
        inputs=[client_name_input, client_profile_input],
        outputs=[
            stage1_output,
            stage2_output,
            raw_market_data_output,
            stage3_output,
            stage4_output,
            stage5_output,
            status_output,
        ],
    )

if __name__ == "__main__":
    demo.launch(share=os.environ.get("GRADIO_SHARE") == "1")
