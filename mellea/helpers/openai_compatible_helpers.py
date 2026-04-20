"""A file for helper functions that deal with OpenAI API compatible helpers."""

import json
from typing import Any

from pydantic import BaseModel

from ..backends.tools import validate_tool_arguments
from ..core import MelleaLogger, ModelToolCall
from ..core.base import AbstractMelleaTool
from ..stdlib.components import Document, Message


class CompletionUsage(BaseModel):
    """Token usage statistics for a completion request."""

    completion_tokens: int
    """Number of tokens in the generated completion."""

    prompt_tokens: int
    """Number of tokens in the prompt."""

    total_tokens: int
    """Total number of tokens used in the request (prompt + completion)."""


def extract_model_tool_requests(
    tools: dict[str, AbstractMelleaTool], response: dict[str, Any]
) -> dict[str, ModelToolCall] | None:
    """Extract tool calls from the dict representation of an OpenAI-like chat response object.

    Args:
        tools: Mapping of tool name to ``AbstractMelleaTool`` for lookup.
        response: Dict representation of an OpenAI-compatible chat completion message
            (must contain a ``"message"`` key).

    Returns:
        Mapping of tool name to ``ModelToolCall`` for each requested tool call, or
        ``None`` if no tool calls were found.
    """
    model_tool_calls: dict[str, ModelToolCall] = {}
    calls = response["message"].get("tool_calls", None)
    if calls:
        for tool_call in calls:
            tool_name = tool_call["function"]["name"]  # type: ignore
            tool_args = tool_call["function"]["arguments"]  # type: ignore

            func = tools.get(tool_name)
            if func is None:
                MelleaLogger.get_logger().warning(
                    f"model attempted to call a non-existing function: {tool_name}"
                )
                continue  # skip this function if we can't find it.

            args = {}
            if tool_args is not None:
                # Returns the args as a string. Parse it here.
                args = json.loads(tool_args)

            # Validate and coerce argument types
            validated_args = validate_tool_arguments(func, args, strict=False)
            model_tool_calls[tool_name] = ModelToolCall(tool_name, func, validated_args)

    if len(model_tool_calls) > 0:
        return model_tool_calls
    return None


def chat_completion_delta_merge(
    chunks: list[dict], force_all_tool_calls_separate: bool = False
) -> dict:
    """Merge a list of deltas from ``ChatCompletionChunk``s into a single dict representing the ``ChatCompletion`` choice.

    Args:
        chunks: The list of dicts that represent the message deltas.
        force_all_tool_calls_separate: If ``True``, tool calls in separate message
            deltas will not be merged even if their index values are the same. Use
            when providers do not return the correct index value for tool calls; all
            tool calls must then be fully populated in a single delta.

    Returns:
        A single merged dict representing the assembled ``ChatCompletion`` choice,
        with ``finish_reason``, ``index``, and a ``message`` sub-dict containing
        ``content``, ``role``, and ``tool_calls``.
    """
    merged: dict[str, Any] = dict()

    # `delta`s map to a single choice.
    merged["finish_reason"] = None
    merged["index"] = 0  # We always do the first choice.
    merged["logprobs"] = None
    merged["stop_reason"] = None

    # message fields
    message: dict[str, Any] = dict()
    message["content"] = ""
    message["reasoning_content"] = ""
    message["role"] = None
    m_tool_calls: list[dict] = []
    message["tool_calls"] = m_tool_calls
    merged["message"] = message

    for chunk in chunks:
        # Handle top level fields.
        if chunk.get("finish_reason", None) is not None:
            merged["finish_reason"] = chunk["finish_reason"]
        if chunk.get("stop_reason", None) is not None:
            merged["stop_reason"] = chunk["stop_reason"]

        # Handle fields of the message object.
        if message["role"] is None and chunk["delta"].get("role", None) is not None:
            message["role"] = chunk["delta"]["role"]

        if chunk["delta"].get("content", None) is not None:
            message["content"] += chunk["delta"]["content"]

        thinking = chunk["delta"].get("reasoning_content", None)
        if thinking is not None:
            message["reasoning_content"] += thinking

        tool_calls = chunk["delta"].get("tool_calls", None)
        if tool_calls is not None:
            # Merge the pieces of each tool call from separate chunks into one dict.
            # Example:
            #  chunks: [{'arguments': None, 'name': 'get_weather_precise'}, {'arguments': '{"location": "', 'name': None}, {'arguments': 'Dallas}', 'name': None}]
            #  -> [{'arguments': '{"location": "Dallas"}', 'name': 'get_weather_precise'}]
            for tool_call in tool_calls:
                idx: int = tool_call["index"]
                current_tool = None

                # In a few special cases, we want to force all tool calls to be separate regardless of the index value.
                # If not forced, check that the tool call index in the response isn't already in our list.
                create_new_tool_call = force_all_tool_calls_separate or (
                    idx > len(m_tool_calls) - 1
                )
                if create_new_tool_call:
                    current_tool = {"function": {"name": "", "arguments": None}}
                    m_tool_calls.append(current_tool)
                else:
                    # This tool has already started to be defined.
                    current_tool = m_tool_calls[idx]

                # Get the info from the function chunk.
                fx_info = tool_call["function"]
                if fx_info["name"] is not None:
                    current_tool["function"]["name"] += fx_info["name"]

                if fx_info["arguments"] is not None:
                    # Only populate args if there are any to add.
                    if current_tool["function"]["arguments"] is None:
                        current_tool["function"]["arguments"] = ""
                    current_tool["function"]["arguments"] += fx_info["arguments"]

    return merged


def message_to_openai_message(msg: Message) -> dict:
    """Serialise a Mellea ``Message`` to the format required by OpenAI-compatible API providers.

    Args:
        msg: The ``Message`` object to serialise.

    Returns:
        A dict with ``"role"`` and ``"content"`` fields. When the message carries
        images, ``"content"`` is a list of text and image-URL dicts; otherwise it
        is a plain string.
    """
    if msg.images is not None:
        img_list = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}
            for img in msg.images
        ]

        return {
            "role": msg.role,
            "content": [{"type": "text", "text": msg.content}, *img_list],
        }
    else:
        return {"role": msg.role, "content": msg.content}
        # Target format:
        # {
        #     "role": "user",
        #     "content": [
        #       {
        #         "type": "text",
        #         "text": "What's in this picture?"
        #       },
        #       {
        #         "type": "image_url",
        #         "image_url": {
        #           "url": "data:image/jpeg;base64,<base64_string>"
        #         }
        #       }
        #     ]
        #   }


def messages_to_docs(msgs: list[Message]) -> list[dict[str, str]]:
    """Extract all ``Document`` objects from a list of ``Message`` objects.

    Args:
        msgs: List of ``Message`` objects whose ``_docs`` attributes are inspected.

    Returns:
        A list of dicts, each with a ``"text"`` key and optional ``"title"`` and
        ``"doc_id"`` keys, suitable for passing to an OpenAI-compatible RAG API.
    """
    docs: list[Document] = []
    for message in msgs:
        if message._docs is not None:
            docs.extend(message._docs)

    json_docs: list[dict[str, str]] = []
    for doc in docs:
        json_doc = {"text": doc.text}
        if doc.title is not None:
            json_doc["title"] = doc.title
        if doc.doc_id is not None:
            json_doc["doc_id"] = doc.doc_id
        json_docs.append(json_doc)
    return json_docs


def build_completion_usage(output: Any) -> CompletionUsage | None:
    """Build a normalized usage object from a model output, if available.

    Args:
        output: Model output object that may expose a ``usage`` mapping with
            token counts.

    Returns:
        A ``CompletionUsage`` object when usage metadata is present on the
        output, otherwise ``None``.
    """
    if not hasattr(output, "usage") or output.usage is None:
        return None

    prompt_tokens = output.usage.get("prompt_tokens", 0)
    completion_tokens = output.usage.get("completion_tokens", 0)
    total_tokens = output.usage.get("total_tokens", prompt_tokens + completion_tokens)
    return CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
