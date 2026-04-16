"""Streaming utilities for OpenAI-compatible server responses."""

from collections.abc import AsyncGenerator

from mellea.core.base import ModelOutputThunk
from mellea.core.utils import MelleaLogger
from mellea.helpers.openai_compatible_helpers import build_completion_usage

from .models import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    OpenAIError,
    OpenAIErrorResponse,
    StreamOptions,
)


async def stream_chat_completion_chunks(
    output: ModelOutputThunk,
    completion_id: str,
    model: str,
    created: int,
    stream_options: StreamOptions | None = None,
    system_fingerprint: str | None = None,
) -> AsyncGenerator[str, None]:
    """Generate OpenAI-compatible SSE chat completion chunks from a model output.

    Args:
        output: The model output object to stream.
        completion_id: Unique identifier for this completion.
        model: Model name to include in chunks.
        created: Unix timestamp of when the completion was created.
        stream_options: OpenAI-compatible streaming options. Controls whether
            usage statistics are included in the final chunk via the
            ``include_usage`` field.
        system_fingerprint: Backend configuration fingerprint to include in chunks.
            Defaults to ``None``.

    Yields:
        Server-sent event payload strings representing OpenAI-compatible chat
        completion chunks, including the terminating ``[DONE]`` event.
    """
    try:
        initial_chunk = ChatCompletionChunk(
            id=completion_id,
            model=model,
            created=created,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(role="assistant", content=None),
                    finish_reason=None,
                )
            ],
            object="chat.completion.chunk",
            system_fingerprint=system_fingerprint,
        )
        yield f"data: {initial_chunk.model_dump_json()}\n\n"

        # Handle pre-computed output: emit value as a single content chunk
        if output.is_computed():
            if output.value is not None:
                chunk = ChatCompletionChunk(
                    id=completion_id,
                    model=model,
                    created=created,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(content=output.value),
                            finish_reason=None,
                        )
                    ],
                    object="chat.completion.chunk",
                    system_fingerprint=system_fingerprint,
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
        else:
            # Stream incremental chunks for uncomputed output
            while not output.is_computed():
                delta_content = await output.astream()

                if delta_content:
                    chunk = ChatCompletionChunk(
                        id=completion_id,
                        model=model,
                        created=created,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionChunkDelta(content=delta_content),
                                finish_reason=None,
                            )
                        ],
                        object="chat.completion.chunk",
                        system_fingerprint=system_fingerprint,
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

        # Include usage in final chunk only if explicitly requested via stream_options
        # Per OpenAI spec: usage is only included when stream_options.include_usage=True
        include_usage = stream_options is not None and stream_options.include_usage

        usage = build_completion_usage(output) if include_usage else None

        final_chunk = ChatCompletionChunk(
            id=completion_id,
            model=model,
            created=created,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(content=None),
                    finish_reason="stop",
                )
            ],
            object="chat.completion.chunk",
            system_fingerprint=system_fingerprint,
            usage=usage,
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        MelleaLogger.get_logger().exception("Streaming error")
        error_response = OpenAIErrorResponse(
            error=OpenAIError(message=f"Streaming error: {e!s}", type="server_error")
        )
        yield f"data: {error_response.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
