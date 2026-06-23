# pytest: ollama, e2e

"""Example: M serve with image support.

This example shows how to create a serve function that reads images from
message objects and uses them with a vision model via Ollama's OpenAI-compatible API.

Prerequisites:
    - Ollama running locally with a vision model pulled
    - Run: ollama pull granite3.2-vision

Usage:
    m serve docs/examples/m_serve/m_serve_example_multimodal_image.py

Then test with:
    uv run python docs/examples/m_serve/client_multimodal_image.py
"""

from typing import Any, cast

from mellea import start_session
from mellea.core import ModelOutputThunk
from mellea.serve import ChatMessage

MODEL_ID = "granite3.2-vision"
session = start_session(model_id=MODEL_ID)

# MODEL_ID = "llava"
#
# _ollama_host = os.environ.get("OLLAMA_HOST", "localhost:11434")
# if not _ollama_host.startswith(("http://", "https://")):
# _ollama_host = f"http://{_ollama_host}"

# backend = OpenAIBackend(
# model_id=MODEL_ID,
# base_url=f"{_ollama_host}/v1",
# api_key="ollama",
# )
# session = MelleaSession(backend, ctx=ChatContext())


async def serve(
    input: list[ChatMessage],
    requirements: list[str] | None = None,
    model_options: dict[str, Any] | None = None,
) -> ModelOutputThunk:
    """Serve function that supports multimodal image input."""

    _ = requirements, model_options  # Not used in this example

    if not input:
        return ModelOutputThunk(value="No input provided")

    last_message = input[-1]
    text = last_message.get_text_content() or "Describe this image"
    image_blocks = last_message.get_image_blocks()
    result = session.chat(content=text, images=image_blocks)

    print(f"Result content: {result.content[:100] if result.content else 'None'}...")
    return cast(ModelOutputThunk, session.ctx.as_list()[-1])
