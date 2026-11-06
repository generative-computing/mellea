# pytest: ollama, e2e

"""Example to run m serve with true streaming support.

Supports both normal and streaming responses from the same `serve()`
function: returns a computed `ComputedModelOutputThunk` for non-streaming
requests, and an uncomputed `ModelOutputThunk` for streaming requests so
the server can emit incremental Server-Sent Events (SSE) chunks.

Prerequisites:
    ollama serve
    ollama pull granite4.2:3b

Usage:
    m serve docs/examples/m_serve/streaming/m_serve_example_streaming.py
"""

from typing import Any

import mellea
from mellea.backends.model_options import ModelOption
from mellea.core import ComputedModelOutputThunk, ModelOutputThunk
from mellea.serve import ChatMessage
from mellea.stdlib.context import SimpleContext


async def serve(
    input: list[ChatMessage],
    requirements: list[str] | None = None,
    model_options: dict[str, Any] | None = None,
) -> ModelOutputThunk | ComputedModelOutputThunk:
    """Support both normal and streaming responses from the same example.

    Returns a computed result for non-streaming requests and an uncomputed thunk
    for streaming requests.
    """
    del requirements
    message = input[-1].get_text_content()
    is_streaming = bool((model_options or {}).get(ModelOption.STREAM, False))

    session = mellea.start_session(ctx=SimpleContext())
    if is_streaming:
        return await session.ainstruct(
            description=message,
            strategy=None,
            model_options=model_options,
            await_result=False,
        )

    return await session.ainstruct(
        description=message,
        strategy=None,
        model_options=model_options,
        await_result=True,
    )
