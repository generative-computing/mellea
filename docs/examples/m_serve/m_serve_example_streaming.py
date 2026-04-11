# pytest: ollama, e2e

"""Example to run m serve with true streaming support."""

import mellea
from cli.serve.models import ChatMessage
from mellea.backends.model_options import ModelOption
from mellea.core import ComputedModelOutputThunk, ModelOutputThunk
from mellea.stdlib.context import SimpleContext

session = mellea.start_session(ctx=SimpleContext())


async def serve(
    input: list[ChatMessage],
    requirements: list[str] | None = None,
    model_options: dict | None = None,
) -> ModelOutputThunk | ComputedModelOutputThunk:
    """Support both normal and streaming responses from the same example.

    Returns a computed result for non-streaming requests and an uncomputed thunk
    for streaming requests.
    """
    del requirements
    message = input[-1].content or ""
    is_streaming = bool((model_options or {}).get(ModelOption.STREAM, False))

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
