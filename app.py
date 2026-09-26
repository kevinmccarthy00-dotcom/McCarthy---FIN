"""Thin launcher so the app can be run from the repository root.

The actual Gradio UI and all chain logic live under scripts/.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

from gradio_app import demo  # noqa: E402

if __name__ == "__main__":
    demo.launch(share=os.environ.get("GRADIO_SHARE") == "1")
