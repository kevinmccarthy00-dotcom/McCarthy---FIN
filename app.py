import os

import gradio as gr

with gr.Blocks(title="DCF Valuation") as demo:
    gr.Markdown("# DCF Valuation")

if __name__ == "__main__":
    demo.launch(share=os.environ.get("GRADIO_SHARE") == "1")
