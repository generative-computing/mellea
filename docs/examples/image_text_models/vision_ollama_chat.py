# pytest: ollama, e2e

"""Example of using Ollama with vision models with linear context.

Demonstrates passing a PIL image to `session.instruct()` with a
`granite-vision` model via Ollama and `ChatContext` to preserve image context
across follow-up instructions.

Prerequisites:
    ollama serve
    ollama pull hf.co/ibm-granite/granite-vision-4.1-4b-GGUF:Q4_K_M

Usage:
    uv run python docs/examples/image_text_models/vision_ollama_chat.py
"""

import pathlib

from PIL import Image

from mellea import start_session
from mellea.stdlib.context import ChatContext

m = start_session(
    model_id="hf.co/ibm-granite/granite-vision-4.1-4b-GGUF:Q4_K_M", ctx=ChatContext()
)
# m = start_session(model_id="llava", ctx=ChatContext())

# load image
image_path = pathlib.Path(__file__).parent.joinpath("pointing_up.jpg")
test_pil = Image.open(image_path)

# ask a question about the image
res = m.instruct("Is the subject in the image smiling?", images=[test_pil])  # type: ignore[arg-type]
print(f"Result:{res!s}")

# This instruction should refer to the first image.
res2 = m.instruct("How many eyes can you identify in the image? Explain.")
print(f"Result:{res2!s}")
