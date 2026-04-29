from typing import Any, Literal

FinishReason = Literal[
    "stop", "length", "content_filter", "tool_calls", "function_call"
]


def extract_finish_reason(output: Any) -> FinishReason:
    """Extract finish_reason from ModelOutputThunk metadata.

    Args:
        output: The model output thunk containing response metadata.

    Returns:
        The finish_reason from the backend response, defaulting to "stop" if unavailable.
        Possible values: "stop", "length", "content_filter", "tool_calls", "function_call".
    """
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

    # Default to "stop" per OpenAI spec
    return "stop"
