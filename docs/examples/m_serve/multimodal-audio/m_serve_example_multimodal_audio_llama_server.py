# pytest: openai, e2e, qualitative

"""Example: M serve with audio input support via llama-server + Gemma.

This example shows how to create a serve function that reads audio from
message objects and uses them with an audio-capable model via a
llama-server OpenAI-compatible endpoint.

Unlike the Ollama version, llama-server with a multimodal Gemma checkpoint
supports native audio-text-to-text: audio and text are sent together in a
single request and the model responds in text.

Caller-supplied `requirements` and `model_options` are forwarded to `ainstruct`.

Prerequisites:
    - llama-server running with an audio-capable Gemma checkpoint, e.g.:

    ```bash
    llama-server \\
        --model gemma-4-12b-it-Q8_0.gguf \\
        --mmproj mmproj-F16.gguf \\
        --n-gpu-layers 99 \\
        --ctx-size 32768 \\
        --flash-attn on \\
        --parallel 1 \\
        --jinja \\
        --host 0.0.0.0 --port 8088
    ```

Environment variables (all optional):
    LLAMA_SERVER_URL      base URL of the llama-server  (default: http://localhost:8088/v1)
    LLAMA_SERVER_API_KEY  API key                       (default: default)
    LLAMA_SERVER_MODEL    model name                    (default: gemma-4-12b-it-Q8_0.gguf)

Usage:
    m serve docs/examples/m_serve/multimodal-audio/m_serve_example_multimodal_audio_llama_server.py

Then test with:
    uv run python docs/examples/m_serve/multimodal-audio/client_multimodal_audio.py
"""

import os
from typing import Any

from mellea import start_session
from mellea.backends import ModelOption
from mellea.core import AudioBlock, AudioUrlBlock, ModelOutputThunk, Requirement
from mellea.serve import ChatMessage
from mellea.stdlib.sampling import RejectionSamplingStrategy

_base_url = os.environ.get("LLAMA_SERVER_URL", "http://localhost:8088/v1")
_api_key = os.environ.get("LLAMA_SERVER_API_KEY", "default")
_model_id = os.environ.get("LLAMA_SERVER_MODEL", "gemma-4-12b-it-Q8_0.gguf")


async def serve(
    input: list[ChatMessage],
    requirements: list[str] | None = None,
    model_options: dict[str, Any] | None = None,
) -> ModelOutputThunk:
    """Serve function that supports native audio-text-to-text via llama-server.

    Caller-supplied `requirements` and `model_options` are forwarded to `ainstruct`.
    """
    if not input:
        return ModelOutputThunk(value="No input provided")

    last_message = input[-1]
    text = last_message.get_text_content() or "Transcribe or describe this audio"
    audio_blocks: list[AudioBlock | AudioUrlBlock] = list(
        last_message.get_audio_blocks()
    )

    if not audio_blocks:
        raise ValueError(
            "No audio provided. Please include an audio clip in your message."
        )

    session = start_session(
        "openai",
        model_id=_model_id,
        base_url=_base_url,
        api_key=_api_key,
        model_options={ModelOption.MAX_NEW_TOKENS: 1000, "modalities": ["text"]},
    )
    result = await session.ainstruct(
        description=text,
        audio=audio_blocks,
        requirements=[Requirement(r) for r in (requirements or []) if r],
        strategy=RejectionSamplingStrategy(loop_budget=2),
        model_options=model_options,
        await_result=True,
    )

    print(f"Result content: {result.value[:100] if result.value else 'None'}...")
    return result
