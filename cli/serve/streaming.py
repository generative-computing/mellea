# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming utilities for OpenAI-compatible server responses."""

import json
import uuid
from collections.abc import AsyncGenerator, Callable
from typing import Any

from mellea.core.base import ModelOutputThunk
from mellea.core.utils import MelleaLogger
from mellea.helpers.openai_compatible_helpers import (
    build_completion_usage,
    build_response_output_items,
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
    OutputTextContent,
    Response,
    ResponseError,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseUsage,
    StreamOptions,
)
from .utils import extract_finish_reason


def _format_sse_event(event_name: str, payload: dict[str, Any]) -> str:
    """Format a single Responses API SSE event, injecting its `type` field.

    Args:
        event_name: SSE event name; also used as the payload's `type` field.
        payload: Event payload fields (without `type`).

    Returns:
        The `event:`/`data:` line pair as one SSE chunk string.
    """
    return (
        f"event: {event_name}\ndata: {json.dumps({'type': event_name, **payload})}\n\n"
    )


def _message_item_open_events(
    item_id: str, output_index: int, content_index: int, sequence_number: int
) -> str:
    """Build the event pair that opens a message output item.

    Per the Responses API spec, the first event for an item is
    `response.output_item.added`, and streamable content is backed by a
    content part opened with `response.content_part.added`.

    Args:
        item_id: ID of the message item being opened.
        output_index: Output index of the message item.
        content_index: Content index of the text part within the item.
        sequence_number: Sequence number for the `output_item.added` event;
            the `content_part.added` event uses the next number.

    Returns:
        Both events as a single SSE payload string.
    """
    return _format_sse_event(
        "response.output_item.added",
        {
            "sequence_number": sequence_number,
            "output_index": output_index,
            "item": {
                "id": item_id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        },
    ) + _format_sse_event(
        "response.content_part.added",
        {
            "sequence_number": sequence_number + 1,
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
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
    store: bool = True,
    ttl: int = 1800,
    previous_response_id: str | None = None,
    on_complete: Callable[[Response], None] | None = None,
) -> AsyncGenerator[str, None]:
    """Generate Responses API SSE events with semantic event names.

    Emits the following event sequence:

    - `response.created` — initial in-progress envelope
    - `response.in_progress` — signals streaming has started
    - `response.output_item.added` — message item opened (when text is produced)
    - `response.content_part.added` — text content part opened (when text is produced)
    - `response.output_text.delta` — one per streamed token (or one for pre-computed)
    - `response.output_text.done` — full accumulated text (when text is produced)
    - `response.content_part.done` — text content part closed (when text is produced)
    - `response.output_item.done` — message item closed (when text is produced)
    - `response.output_item.added` — function call item opened (one per tool call, if any)
    - `response.function_call_arguments.delta` — one per tool call argument payload (if any)
    - `response.function_call_arguments.done` — one per tool call (if any)
    - `response.output_item.done` — function call item closed (one per tool call, if any)
    - `response.completed` — full `Response` object matching the non-streaming shape

    On error, emits `response.failed` instead of the completion event, carrying
    the output items whose `output_item.done` events were already emitted.

    Args:
        output: The model output thunk to stream.
        response_id: Unique response identifier (`resp_…`).
        model: Model name to include in event payloads.
        created: Unix timestamp of when the response was created.
        conversation: Optional conversation ID to echo back in events.
        include: Reserved for future use. The `response.completed` event
            always carries the full `Response` object including usage.
            Defaults to `None`.
        store: Whether the response is being stored server-side. Controls
            `expires_at` in the completed envelope.
        ttl: Response TTL in seconds. Used to compute `expires_at`.
        previous_response_id: Echoed back in the completed envelope.
        on_complete: Optional callback invoked with the completed `Response`
            object after the `response.completed` event has been sent. Used by
            callers to persist the response for `previous_response_id`
            chaining. Not called if the stream fails or the client disconnects
            before the final event is delivered.
    """
    sequence_number = 0
    accumulated_text = ""
    # Items whose output_item.done event has been emitted; used by both the
    # completed and failed envelopes so the final response always matches
    # what the client saw closed on the stream.
    output_items: list[ResponseOutputItem] = []
    try:
        # Build the in-progress response object from the full Response model so
        # the created/in_progress envelopes parse into a Response model on strict
        # SDK clients and stay consistent with the completed/failed envelopes.
        in_progress_response = Response(
            id=response_id,
            created_at=created,
            expires_at=created + ttl if store else None,
            model=model,
            status="in_progress",
            output=[],
            output_text="",
            usage=ResponseUsage(input_tokens=0, output_tokens=0, total_tokens=0),
            conversation=conversation,
            previous_response_id=previous_response_id,
        )
        in_progress_payload = in_progress_response.model_dump()

        # response.created event - OpenAI SDK format
        yield _format_sse_event(
            "response.created",
            {"sequence_number": sequence_number, "response": in_progress_payload},
        )
        sequence_number += 1

        # response.in_progress event - OpenAI SDK format
        yield _format_sse_event(
            "response.in_progress",
            {"sequence_number": sequence_number, "response": in_progress_payload},
        )
        sequence_number += 1

        output_index = 0
        content_index = 0
        # Generate a message item ID for the output_text events
        item_id = f"msg_{uuid.uuid4().hex[:24]}"

        if output.is_computed():
            text = output.value or ""
            if text:
                accumulated_text = text
                # Open the message item and its text content part before the
                # first delta.
                yield _message_item_open_events(
                    item_id, output_index, content_index, sequence_number
                )
                sequence_number += 2
                # For pre-computed output, emit delta + done
                yield _format_sse_event(
                    "response.output_text.delta",
                    {
                        "sequence_number": sequence_number,
                        "output_index": output_index,
                        "content_index": content_index,
                        "item_id": item_id,
                        "delta": text,
                    },
                )
                sequence_number += 1
                yield _format_sse_event(
                    "response.output_text.done",
                    {
                        "sequence_number": sequence_number,
                        "output_index": output_index,
                        "content_index": content_index,
                        "item_id": item_id,
                        "text": text,
                    },
                )
                sequence_number += 1
        else:
            message_item_opened = False
            while not output.is_computed():
                delta = await output.astream()
                if delta:
                    if not message_item_opened:
                        message_item_opened = True
                        # Open the message item and its text content part
                        # before the first streamed delta.
                        yield _message_item_open_events(
                            item_id, output_index, content_index, sequence_number
                        )
                        sequence_number += 2
                    accumulated_text += delta
                    yield _format_sse_event(
                        "response.output_text.delta",
                        {
                            "sequence_number": sequence_number,
                            "output_index": output_index,
                            "content_index": content_index,
                            "item_id": item_id,
                            "delta": delta,
                        },
                    )
                    sequence_number += 1

            if accumulated_text:
                yield _format_sse_event(
                    "response.output_text.done",
                    {
                        "sequence_number": sequence_number,
                        "output_index": output_index,
                        "content_index": content_index,
                        "item_id": item_id,
                        "text": accumulated_text,
                    },
                )
                sequence_number += 1

        # Build the message item from the accumulated stream text — not
        # `output.value`, which post-processing may rewrite after the last
        # delta — so the item-close events and `response.completed` carry
        # exactly the text the client saw streamed.
        if accumulated_text:
            message_item = ResponseOutputMessage(
                id=item_id,
                type="message",
                role="assistant",
                status="completed",
                content=[OutputTextContent(type="output_text", text=accumulated_text)],
            )
            # Close the text content part, then the message item, before
            # building the function-call items: if that construction fails,
            # the message item is already closed and included in the failed
            # envelope rather than left dangling in_progress on the client.
            yield _format_sse_event(
                "response.content_part.done",
                {
                    "sequence_number": sequence_number,
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "part": {
                        "type": "output_text",
                        "text": accumulated_text,
                        "annotations": [],
                    },
                },
            )
            sequence_number += 1
            yield _format_sse_event(
                "response.output_item.done",
                {
                    "sequence_number": sequence_number,
                    "output_index": output_index,
                    "item": message_item.model_dump(),
                },
            )
            sequence_number += 1
            output_items.append(message_item)

        tool_calls = build_tool_calls(output)
        # Pre-generate function call item IDs so streaming events and the
        # completed response share the same identifiers (per OpenAI spec,
        # item_id in events must match the id of the corresponding output item).
        function_call_ids = (
            [f"fc_{uuid.uuid4().hex[:24]}" for _ in tool_calls] if tool_calls else []
        )

        if tool_calls:
            # Build only the function-call items here (message_text="" skips
            # the value-derived message item, built above from streamed text).
            fc_items = build_response_output_items(
                output, tool_calls, function_call_ids=function_call_ids, message_text=""
            )
            # output_index for tool calls: starts at 0 if no text, 1 if text exists
            start_index = 1 if accumulated_text else 0
            for idx, (tool_call, fc_id, fc_item) in enumerate(
                zip(tool_calls, function_call_ids, fc_items), start=start_index
            ):
                output_index = idx
                name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]
                # Open the function call item before the argument events
                # (spec: the first event for an item is output_item.added,
                # echoing at least the function name).
                yield _format_sse_event(
                    "response.output_item.added",
                    {
                        "sequence_number": sequence_number,
                        "output_index": output_index,
                        "item": {
                            "id": fc_id,
                            "type": "function_call",
                            "status": "in_progress",
                            "name": name,
                            "arguments": "",
                            "call_id": tool_call["id"],
                        },
                    },
                )
                sequence_number += 1
                # Emit a single delta carrying the full arguments string.
                # The backend doesn't stream arguments incrementally, so one
                # delta with the complete payload is spec-conformant.
                yield _format_sse_event(
                    "response.function_call_arguments.delta",
                    {
                        "sequence_number": sequence_number,
                        "item_id": fc_id,
                        "output_index": output_index,
                        "call_id": tool_call["id"],
                        "delta": arguments,
                    },
                )
                sequence_number += 1
                yield _format_sse_event(
                    "response.function_call_arguments.done",
                    {
                        "sequence_number": sequence_number,
                        "item_id": fc_id,
                        "output_index": output_index,
                        "call_id": tool_call["id"],
                        "arguments": arguments,
                    },
                )
                sequence_number += 1
                # Close the function call item with the completed item object.
                yield _format_sse_event(
                    "response.output_item.done",
                    {
                        "sequence_number": sequence_number,
                        "output_index": output_index,
                        "item": fc_item.model_dump(),
                    },
                )
                sequence_number += 1
                output_items.append(fc_item)

        usage = build_response_usage(output) or ResponseUsage(
            input_tokens=0, output_tokens=0, total_tokens=0
        )
        completed_response = Response(
            id=response_id,
            created_at=created,
            expires_at=created + ttl if store else None,
            model=model,
            status="completed",
            output=output_items,
            output_text=accumulated_text,
            usage=usage,
            conversation=conversation,
            previous_response_id=previous_response_id,
        )
        yield _format_sse_event(
            "response.completed",
            {
                "sequence_number": sequence_number,
                "response": completed_response.model_dump(),
            },
        )

        # Invoked after the completed event has been sent to the client. Runs
        # only when the consumer pulls the generator to completion, so a
        # client disconnect (or a failed stream, which takes the except
        # branch) leaves the callback uninvoked. Isolated in its own
        # try/except so a callback failure cannot send a response.failed
        # event after response.completed has already been delivered.
        if on_complete is not None:
            try:
                on_complete(completed_response)
            except Exception:
                MelleaLogger.get_logger().exception("on_complete callback failed")

    except Exception as e:
        MelleaLogger.get_logger().exception("Streaming error in responses endpoint")
        # Build the failed envelope from the full Response model so strict
        # SDK clients can parse it (a minimal dict would be missing required
        # fields like `model`, `created_at`, and `output`). It carries the
        # items the client already saw closed so event-driven reconstruction
        # and the failed response agree.
        failed_response = Response(
            id=response_id,
            created_at=created,
            expires_at=created + ttl if store else None,
            model=model,
            status="failed",
            output=output_items,
            output_text=accumulated_text,
            error=ResponseError(code="server_error", message=str(e)),
            usage=ResponseUsage(input_tokens=0, output_tokens=0, total_tokens=0),
            conversation=conversation,
            previous_response_id=previous_response_id,
        )
        yield _format_sse_event(
            "response.failed",
            {
                "sequence_number": sequence_number,
                "response": failed_response.model_dump(),
            },
        )
