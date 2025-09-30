"""A file for helper functions that deal with OpenAI API compatible helpers."""

import json
from collections.abc import Callable
from typing import Any

from mellea.helpers.fancy_logger import FancyLogger
from mellea.stdlib.base import ModelToolCall


def extract_model_tool_requests(
    tools: dict[str, Callable], response: dict[str, Any]
) -> dict[str, ModelToolCall] | None:
    """Extracts tool calls from the dict representation of an OpenAI-like chat response object."""
    model_tool_calls: dict[str, ModelToolCall] = {}
    calls = response["message"].get("tool_calls", None)
    if calls:
        for tool_call in calls:
            tool_name = tool_call["function"]["name"]  # type: ignore
            tool_args = tool_call["function"]["arguments"]  # type: ignore

            func = tools.get(tool_name)
            if func is None:
                FancyLogger.get_logger().warning(
                    f"model attempted to call a non-existing function: {tool_name}"
                )
                continue  # skip this function if we can't find it.

            args = {}
            if tool_args is not None:
                # Returns the args as a string. Parse it here.
                args = json.loads(tool_args)
            model_tool_calls[tool_name] = ModelToolCall(tool_name, func, args)

    if len(model_tool_calls) > 0:
        return model_tool_calls
    return None


def chat_completion_delta_merge(
    chunks: list[dict], force_all_tool_calls_separate: bool = False
) -> dict:
    """Takes a list of deltas from `ChatCompletionChunk`s and merges them into a single dict representing the `ChatCompletion` choice.

    Args:
        chunks: the list of dicts that represent the message deltas
        force_all_tool_calls_separate: if `True`, tool calls in separate message deltas will not be merged (even if their index values are the same); use when providers do not return the correct index value for tool calls. If using this option, all tool calls must be fully populated in a single delta since they won't be merged.
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
