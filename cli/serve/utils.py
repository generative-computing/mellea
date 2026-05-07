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

    # Try to get finish_reason from the response metadata
    # Different backends store this in different places
    if hasattr(output, "_meta") and output._meta:
        # Ollama backend stores response in chat_response with done_reason field
        # (ollama.ChatResponse object with done_reason attribute)
        chat_response = output._meta.get("chat_response")
        if chat_response and hasattr(chat_response, "done_reason"):
            done_reason = chat_response.done_reason
            if done_reason in valid_reasons:
                return done_reason

        # OpenAI backend stores full response dict in oai_chat_response
        # (from chunk.model_dump() which includes choices array)
        oai_response = output._meta.get("oai_chat_response")
        if oai_response and isinstance(oai_response, dict):
            choices = oai_response.get("choices", [])
            if choices and len(choices) > 0:
                finish_reason = choices[0].get("finish_reason")
                if finish_reason in valid_reasons:
                    return finish_reason

        # LiteLLM backend stores response dict in litellm_chat_response
        litellm_response = output._meta.get("litellm_chat_response")
        if litellm_response and isinstance(litellm_response, dict):
            finish_reason = litellm_response.get("finish_reason")
            if finish_reason in valid_reasons:
                return finish_reason

    # Default to "stop" per OpenAI spec
    return "stop"
