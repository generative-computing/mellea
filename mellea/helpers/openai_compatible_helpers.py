"""A file for helper functions that deal with OpenAI API compatible helpers."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from pydantic import BaseModel

from ..core.base import AudioBlock, AudioUrlBlock, ImageUrlBlock

if TYPE_CHECKING:
    from ..core import Formatter, ModelToolCall
    from ..core.base import AbstractMelleaTool, ModelOutputThunk
    from ..stdlib.components import Document, Message


class ToolCallFunction(TypedDict):
    """Function details in a tool call."""

    name: str
    arguments: str


class ToolCallDict(TypedDict):
    """OpenAI-compatible tool call dictionary with ID and function."""

    id: str
    type: Literal["function"]
    function: ToolCallFunction


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
        tools: Mapping of tool name to `AbstractMelleaTool` for lookup.
        response: Dict representation of an OpenAI-compatible chat completion message
            (must contain a `"message"` key).

    Returns:
        Mapping of tool name to `ModelToolCall` for each requested tool call, or
        `None` if no tool calls were found.
    """
    from ..backends.tools import validate_tool_arguments
    from ..core import MelleaLogger, ModelToolCall

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
                try:
                    args = json.loads(tool_args)
                except json.JSONDecodeError:
                    MelleaLogger.get_logger().warning(
                        f"model returned malformed JSON arguments for tool {tool_name!r} "
                        f"(possibly truncated during streaming); skipping this tool call: {tool_args!r}"
                    )
                    continue

            # Validate and coerce argument types
            validated_args = validate_tool_arguments(func, args, strict=False)
            model_tool_calls[tool_name] = ModelToolCall(tool_name, func, validated_args)

    if len(model_tool_calls) > 0:
        return model_tool_calls
    return None


def chat_completion_delta_merge(
    chunks: list[dict], force_all_tool_calls_separate: bool = False
) -> dict:
    """Merge a list of deltas from `ChatCompletionChunk`s into a single dict representing the `ChatCompletion` choice.

    Args:
        chunks: The list of dicts that represent the message deltas.
        force_all_tool_calls_separate: If `True`, tool calls in separate message
            deltas will not be merged even if their index values are the same. Use
            when providers do not return the correct index value for tool calls; all
            tool calls must then be fully populated in a single delta.

    Returns:
        A single merged dict representing the assembled `ChatCompletion` choice,
        with `finish_reason`, `index`, and a `message` sub-dict containing
        `content`, `role`, and `tool_calls`.
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


def should_replay_reasoning(
    messages: list[Message], provider: str | None
) -> list[bool]:
    """Decide, per message, whether its reasoning trace should be replayed to the provider.

    Implements the cross-provider consensus rule from issue #1201: an assistant
    message's reasoning is round-tripped only when that turn issued a tool call —
    detected by a `tool`-role message immediately following it — and stripped on
    plain follow-up turns. Non-assistant messages and assistant messages without
    reasoning always return `False`.

    Args:
        messages: The conversation in order, as it will be serialised.
        provider: The backend provider name (e.g. `"openai"`, `"ollama"`).
            Currently unused — every provider follows the consensus rule above.
            It is a reserved hook for a provider-specific deviation (e.g. a model
            that must replay reasoning on plain turns, or must not after a tool
            call); add a keyed branch here once such a case is verified live.

    Returns:
        A list of booleans, one per message in `messages`, indicating whether that
        message's reasoning should be included in the serialised payload.
    """
    flags: list[bool] = []
    for i, msg in enumerate(messages):
        if msg.role != "assistant" or not msg.thinking:
            flags.append(False)
            continue
        # The prior turn "had a tool call" iff the next message is a tool result.
        prior_turn_had_tool_call = (
            i + 1 < len(messages) and messages[i + 1].role == "tool"
        )
        flags.append(prior_turn_had_tool_call)
    return flags


def message_to_openai_message(
    msg: Message, formatter: Formatter | None = None, *, replay_reasoning: bool = False
) -> dict:
    """Serialise a Mellea `Message` to the format required by OpenAI-compatible API providers.

    Args:
        msg: The `Message` object to serialise.
        formatter: Optional formatter used to render the message content (including
            documents) through the template system. When `None`, uses the raw
            `msg.content` string without document rendering.
        replay_reasoning: When `True` and `msg.thinking` is a non-empty string,
            the reasoning trace is emitted under the `"reasoning_content"` key so
            the provider receives the model's prior reasoning. Defaults to `False`
            (reasoning is stripped), preserving the historical behaviour; callers
            decide per-turn via their replay policy (see `should_replay_reasoning`).

    Returns:
        A dict with `"role"` and `"content"` fields. When the message carries
        images or audio, `"content"` is a list of content-part dicts; otherwise
        is a plain string. For tool-only assistant turns, `"content"` is `None`
        and `"tool_calls"` carries the structured call list. When content is
        present alongside tool calls, both keys are included. When
        `replay_reasoning` is `True` and reasoning is present, the dict also
        carries a `"reasoning_content"` field.

    Raises:
        ValueError: If the message contains an `AudioUrlBlock`. The OpenAI Chat
            Completions audio schema does not support audio by URL; fetch the
            audio and pass it as an `AudioBlock` with base64 data instead.
    """
    # NOTE: `self.formatter.to_chat_messages` explicitly skips `Message` objects. However, we need
    # to print `Message`s to correctly serialize any documents with the message. Do the printing here.
    content = formatter.print(msg) if formatter else msg.content
    if msg.images is not None or msg.audio is not None:
        parts: list[dict] = [{"type": "text", "text": content}]

        if msg.images is not None:
            for img in msg.images:
                if isinstance(img, ImageUrlBlock):
                    url = str(img.value)
                else:
                    # ImageBlock: base64-encoded PNG
                    raw = str(img.value)
                    url = (
                        raw
                        if raw.startswith("data:")
                        else f"data:image/png;base64,{raw}"
                    )
                parts.append({"type": "image_url", "image_url": {"url": url}})

        if msg.audio is not None:
            for audio in msg.audio:
                if isinstance(audio, AudioBlock):
                    raw = str(audio.value)
                    # Strip data URI prefix — OpenAI expects raw base64 in the `data` field.
                    if "base64," in raw:
                        raw = raw.split("base64,", 1)[1]
                    parts.append(
                        {
                            "type": "input_audio",
                            "input_audio": {"data": raw, "format": audio.format},
                        }
                    )
                elif isinstance(audio, AudioUrlBlock):
                    # OpenAI Chat Completions does not support audio by URL;
                    # AudioUrlBlock cannot be serialised to this schema.
                    raise ValueError(
                        f"AudioUrlBlock cannot be serialised to the OpenAI Chat Completions "
                        f"audio schema (URL: {audio.value!r}). "
                        "Fetch the audio and use AudioBlock with base64 data instead."
                    )

        result: dict[str, Any] = {"role": msg.role, "content": parts}
    else:
        result = {"role": msg.role, "content": content}

    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        result["tool_calls"] = tool_calls
        if msg.images is None and not content:
            result["content"] = None

    if replay_reasoning and msg.thinking:
        result["reasoning_content"] = msg.thinking
    return result


def messages_to_docs(msgs: list[Message]) -> list[dict[str, str]]:
    """Extract all `Document` objects from a list of `Message` objects.

    Args:
        msgs: List of `Message` objects whose `_docs` attributes are inspected.

    Returns:
        A list of dicts, each with a `"text"` key and optional `"title"` and
        `"doc_id"` keys, suitable for passing to an OpenAI-compatible RAG API.
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


def build_completion_usage(output: ModelOutputThunk) -> CompletionUsage | None:
    """Build a normalized usage object from a model output, if available.

    Args:
        output: Model output object whose `generation.usage` mapping contains
            token counts.

    Returns:
        A `CompletionUsage` object when usage metadata is present on the
        output, otherwise `None`.
    """
    if output.generation.usage is None:
        return None

    prompt_tokens = output.generation.usage.get("prompt_tokens", 0)
    completion_tokens = output.generation.usage.get("completion_tokens", 0)
    total_tokens = output.generation.usage.get(
        "total_tokens", prompt_tokens + completion_tokens
    )
    return CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def has_tool_calls(output: ModelOutputThunk) -> bool:
    """Check if a model output has tool calls.

    Args:
        output: Model output thunk that may expose a `tool_calls` mapping.

    Returns:
        `True` if the output has non-empty tool calls, `False` otherwise.
    """
    return (
        hasattr(output, "tool_calls")
        and output.tool_calls is not None
        and isinstance(output.tool_calls, dict)
        and bool(output.tool_calls)
    )


def build_tool_calls(output: ModelOutputThunk) -> list[ToolCallDict] | None:
    """Build OpenAI-compatible tool calls from a model output, if available.

    Args:
        output: Model output thunk that may expose a `tool_calls` mapping.

    Returns:
        List of `ToolCallDict` objects when tool calls are present,
        otherwise `None`.
    """
    if not has_tool_calls(output):
        return None

    assert output.tool_calls is not None
    tool_calls: list[ToolCallDict] = []
    for model_tool_call in output.tool_calls.values():
        # Generate a unique ID for this tool call
        tool_call_id = f"call_{uuid.uuid4().hex[:24]}"

        # Serialize arguments to JSON with str fallback for non-serializable types
        args_json = json.dumps(model_tool_call.args, default=str)

        tool_call: ToolCallDict = {
            "id": tool_call_id,
            "type": "function",
            "function": {"name": model_tool_call.name, "arguments": args_json},
        }
        tool_calls.append(tool_call)

    return tool_calls
