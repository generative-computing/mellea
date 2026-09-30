# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming utilities for OpenAI-compatible server responses."""

import json
from collections.abc import AsyncGenerator
from typing import Literal

from mellea.core.base import ModelOutputThunk
from mellea.core.utils import MelleaLogger
from mellea.helpers.openai_compatible_helpers import (
    build_completion_usage,
    build_response_usage,
    build_tool_calls,
)

from .models import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionMessageToolCallDelta,
    OpenAIError,
    OpenAIErrorResponse,
    StreamOptions,
)
from .utils import extract_finish_reason


async def stream_chat_completion_chunks(
    output: ModelOutputThunk,
    completion_id: str,
    model: str,
    created: int,
    stream_options: StreamOptions | None = None,
    system_fingerprint: str | None = None,
) -> AsyncGenerator[str, None]:
    """Generate OpenAI-compatible SSE chat completion chunks from a model output.

    This function acts as a pass-through streaming layer, forwarding chunks directly
    from the backend to the client without buffering or validation. Format validation
    for structured outputs happens at the module level (in the serve function) and
    client side, not in this streaming layer.

    Args:
        output: The model output object to stream.
        completion_id: Unique identifier for this completion.
        model: Model name to include in chunks.
        created: Unix timestamp of when the completion was created.
        stream_options: OpenAI-compatible streaming options. Controls whether
            usage statistics are included in the final chunk via the
            `include_usage` field.
        system_fingerprint: Backend configuration fingerprint to include in chunks.
            Defaults to `None`.

    Yields:
        Server-sent event payload strings representing OpenAI-compatible chat
        completion chunks, including the terminating `[DONE]` event.
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

        tool_calls_list = build_tool_calls(output)

        if tool_calls_list:
            # Convert to ChatCompletionMessageToolCallDelta objects with required index
            tool_calls = [
                ChatCompletionMessageToolCallDelta.model_validate({**tc, "index": idx})
                for idx, tc in enumerate(tool_calls_list)
            ]

            # Emit tool calls in a separate chunk before the final chunk
            tool_call_chunk = ChatCompletionChunk(
                id=completion_id,
                model=model,
                created=created,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(tool_calls=tool_calls),
                        finish_reason=None,
                    )
                ],
                object="chat.completion.chunk",
                system_fingerprint=system_fingerprint,
            )
            yield f"data: {tool_call_chunk.model_dump_json()}\n\n"

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
                    finish_reason=extract_finish_reason(output),
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


async def stream_response_chunks(
    output,
    response_id: str,
    model: str,
    created: int,
    conversation: str | None = None,
    include: list[str] | None = None,
) -> AsyncGenerator[str, None]:
    """Generate Responses API SSE events with semantic event names.

    Emits the following event sequence:

    - ``response.created`` — initial in-progress envelope
    - ``response.in_progress`` — signals streaming has started
    - ``response.output_text.delta`` — one per streamed token (or one for pre-computed)
    - ``response.output_text.done`` — full accumulated text
    - ``response.function_call_arguments.done`` — one per tool call (if any)
    - ``response.completed`` — final envelope with status; usage only when
      ``"usage"`` appears in ``include``

    On error, emits ``response.failed`` instead of the completion event.

    Args:
        output: The model output thunk to stream.
        response_id: Unique response identifier (``resp_…``).
        model: Model name to include in event payloads.
        created: Unix timestamp of when the response was created.
        conversation: Optional conversation ID to echo back in events.
        include: List of optional fields to include in the response. Pass
            ``["usage"]`` to have token usage emitted in the
            ``response.completed`` event. Defaults to ``None`` (no usage).
    """
    try:
        yield (
            f"event: response.created\n"
            f"data: {json.dumps({'id': response_id, 'created_at': created, 'model': model, 'status': 'in_progress', 'conversation': conversation})}\n\n"
        )
        yield (
            f"event: response.in_progress\n"
            f"data: {json.dumps({'id': response_id, 'status': 'in_progress'})}\n\n"
        )

        accumulated_text = ""

        if output.is_computed():
            text = output.value or ""
            accumulated_text = text
            yield (
                f"event: response.output_text.delta\n"
                f"data: {json.dumps({'id': response_id, 'output_index': 0, 'delta': text})}\n\n"
            )
        else:
            while not output.is_computed():
                delta = await output.astream()
                if delta:
                    accumulated_text += delta
                    yield (
                        f"event: response.output_text.delta\n"
                        f"data: {json.dumps({'id': response_id, 'output_index': 0, 'delta': delta})}\n\n"
                    )

        yield (
            f"event: response.output_text.done\n"
            f"data: {json.dumps({'id': response_id, 'output_index': 0, 'text': accumulated_text})}\n\n"
        )

        tool_calls = build_tool_calls(output)
        if tool_calls:
            for idx, tool_call in enumerate(tool_calls):
                yield (
                    f"event: response.function_call_arguments.done\n"
                    f"data: {json.dumps({'id': response_id, 'output_index': idx + 1, 'call_id': tool_call['id'], 'name': tool_call['function']['name'], 'arguments': tool_call['function']['arguments']})}\n\n"
                )

        include_usage = include is not None and "usage" in include
        usage_obj = build_response_usage(output) if include_usage else None
        yield (
            f"event: response.completed\n"
            f"data: {json.dumps({'id': response_id, 'status': 'completed', 'usage': usage_obj.model_dump() if usage_obj else None})}\n\n"
        )

    except Exception as e:
        MelleaLogger.get_logger().exception("Streaming error in responses endpoint")
        yield (
            f"event: response.failed\n"
            f"data: {json.dumps({'id': response_id, 'status': 'failed', 'error': {'message': str(e), 'code': 'streaming_error'}})}\n\n"
        )
