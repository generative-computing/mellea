# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from mellea.helpers.openai_compatible_helpers import has_tool_calls

FinishReason = Literal[
    "stop", "length", "content_filter", "tool_calls", "function_call"
]


def extract_finish_reason(output: Any) -> FinishReason:
    """Extract finish_reason from ModelOutputThunk metadata.

    First checks if tool_calls are present (returns "tool_calls" if so).
    Then checks backend-specific metadata fields in order: Ollama, OpenAI, LiteLLM.
    Backends without finish_reason metadata (e.g., HuggingFace) fall through to
    the default "stop" value.

    Args:
        output: The model output thunk containing response metadata.

    Returns:
        The finish_reason from the backend response, defaulting to "stop" if unavailable.
        Possible values: "stop", "length", "content_filter", "tool_calls", "function_call".
    """
    # If tool calls are present, finish_reason is always "tool_calls"
    if has_tool_calls(output):
        return "tool_calls"

    # Valid finish_reason values per OpenAI spec
    valid_reasons: set[FinishReason] = {
        "stop",
        "length",
        "content_filter",
        "tool_calls",
        "function_call",
    }

    # Try to get finish_reason from the backend-native response on mot.raw.
    # Different backends store this in different places; switch on mot.raw.provider.
    raw = getattr(output, "raw", None)
    if raw is not None:
        provider = raw.provider
        response = raw.response

        if provider == "ollama" and response is not None:
            # ollama.ChatResponse object with done_reason attribute.
            done_reason = getattr(response, "done_reason", None)
            if done_reason in valid_reasons:
                return done_reason

        elif provider in ("openai", "watsonx", "litellm") and isinstance(
            response, dict
        ):
            # Chat path: full response dict, finish_reason nested under choices[0].
            choices = response.get("choices", [])
            if choices and len(choices) > 0:
                finish_reason = choices[0].get("finish_reason")
                if finish_reason in valid_reasons:
                    return finish_reason
            # Raw-completion path: single choice dict, finish_reason at top level.
            finish_reason = response.get("finish_reason")
            if finish_reason in valid_reasons:
                return finish_reason

    # Default to "stop" per OpenAI spec
    return "stop"
