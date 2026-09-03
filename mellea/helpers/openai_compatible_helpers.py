# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A file for helper functions that deal with OpenAI API compatible helpers."""

from __future__ import annotations

import asyncio
import copy
import json
import uuid
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from pydantic import BaseModel

from ..core.base import AudioBlock, AudioUrlBlock, ImageUrlBlock
from ..core.utils import MelleaLogger

if TYPE_CHECKING:
    from ..core import Formatter, ModelToolCall
    from ..core.base import AbstractMelleaTool, ModelOutputThunk
    from ..stdlib.components import Document, Message


# The providers whose request/wire serialization is OpenAI-compatible — the set a
# `provider_fields` `{"openai": ...}` declaration reaches. This is the *serialization*
# family: `message_to_openai_message` is the shared serializer for `openai`, `litellm`,
# and `huggingface` (huggingface.py), and `watsonx` emits the same wire shape via its
# own inline loop. It is deliberately NOT `Message._parse`'s response tuple
# `("openai", "watsonx", "litellm")`, which groups by *response* shape and excludes HF
# (HF returns token tensors, not a `choices[0].message` dict). Keeping these separate
# prevents a `{"openai": ...}` declaration from wrongly raising on the HuggingFace path.
OPENAI_COMPATIBLE_WIRE_PROVIDERS = frozenset(
    {"openai", "litellm", "watsonx", "huggingface"}
)


def merge_provider_fields(
    base: dict[str, Any],
    provider_fields: dict[str, dict[str, Any]] | None,
    provider: str,
) -> dict[str, Any]:
    """Merge author-declared `provider_fields` into a wire message dict.

    A key matches `provider` iff it is `"*"`, equals `provider` exactly, or is
    `"openai"` and `provider` is in `OPENAI_COMPATIBLE_WIRE_PROVIDERS`. Fields from
    every matching key are merged into `base`, but only for keys `base` did not
    already set — Mellea's known fields always win, and a dropped colliding key is
    debug-logged. `"*"` always matches, so its presence never triggers the mismatch
    error.

    Merged values are deep-copied on the way in, so the wire dict never aliases the
    source `provider_fields`. Mutating a nested value on the originating `Message`
    after serialization (or vice versa) cannot leak across the boundary.

    Whether an unmatched-but-declared field actually reaches the model depends on
    the backend SDK, not this function. This helper only raises when a `provider`
    key names a target the request did not hit (see `Raises`); a field that *does*
    match but that the provider does not recognize is left in the wire dict and
    handled downstream. Ollama and HuggingFace re-validate messages through their
    own schemas and silently drop unknown keys (Ollama via pydantic's default
    `extra="ignore"`; HuggingFace via `apply_chat_template`), so on those paths an
    unrecognized field is dropped rather than surfaced as an error. Only providers
    whose SDK rejects unknown keys will raise, and that error originates in the SDK.

    Args:
        base: The wire message dict built from Mellea's known fields; mutated and
            returned.
        provider_fields: The author's provider-keyed extra fields, or `None`. Must
            be a `dict` mapping each provider target to a `dict` of field name to
            value.
        provider: The provider string of the backend performing serialization.

    Returns:
        `base`, with fields from every matching provider key merged in.

    Raises:
        ValueError: If `provider_fields` is non-empty, contains no `"*"`, and no key
            matches `provider` — the component targeted a backend it did not hit.
        TypeError: If `provider_fields` itself is not a `dict`, or a matching value
            within it is not a `dict` — each level must map keys to their values so
            the merge fails with a named error rather than a bare `AttributeError`.
    """
    if not provider_fields:
        return base

    if not isinstance(provider_fields, dict):
        raise TypeError(
            "provider_fields must be a dict of provider target to field mapping, "
            f"got {type(provider_fields).__name__}."
        )

    def _matches(key: str) -> bool:
        return (
            key == "*"
            or key == provider
            or (key == "openai" and provider in OPENAI_COMPATIBLE_WIRE_PROVIDERS)
        )

    matched_any = False
    for key, fields in provider_fields.items():
        if not _matches(key):
            continue
        matched_any = True
        if not isinstance(fields, dict):
            raise TypeError(
                f"provider_fields[{key!r}] must be a dict of field name to value, "
                f"got {type(fields).__name__}."
            )
        for field, value in fields.items():
            if field in base:
                MelleaLogger.get_logger().debug(
                    "provider_fields[%r][%r] collides with a Mellea-known field; "
                    "dropping the author value (known field wins).",
                    key,
                    field,
                )
                continue
            # Deep-copy so the wire dict never aliases the source Message's
            # provider_fields; later mutation of either side stays isolated.
            base[field] = copy.deepcopy(value)

    if not matched_any:
        raise ValueError(
            f"provider_fields declares target(s) {sorted(provider_fields)} but the "
            f"request is running on provider {provider!r}, which none of them match. "
            'Add a "*" key to declare the field valid on every backend, or target the '
            "provider this component actually runs on."
        )

    return base


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
) -> list[ModelToolCall] | None:
    """Extract tool calls from the dict representation of an OpenAI-like chat response object.

    Args:
        tools: Mapping of tool name to `AbstractMelleaTool` for lookup.
        response: Dict representation of an OpenAI-compatible chat completion message
            (must contain a `"message"` key).

    Returns:
        List of `ModelToolCall` for each requested tool call (order preserved),
        or `None` if no tool calls were found.
    """
    from ..backends.tools import validate_tool_arguments
    from ..core import MelleaLogger, ModelToolCall

    model_tool_calls: list[ModelToolCall] = []
    calls = response["message"].get("tool_calls", None)
    if calls:
        for tool_call in calls:
            try:
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
                model_tool_calls.append(
                    ModelToolCall(
                        tool_name,
                        func,
                        validated_args,
                        tool_call_id=tool_call.get("id"),
                    )
                )
            except (KeyError, TypeError, ValueError) as e:
                MelleaLogger.get_logger().warning(
                    f"Failed to extract tool call from malformed response: {e}; "
                    f"raw tool_call: {tool_call!r}"
                )
                continue

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

                # id and type arrive only on the first chunk of a tool call.
                if tool_call.get("id") is not None:
                    current_tool["id"] = tool_call["id"]
                if tool_call.get("type") is not None:
                    current_tool["type"] = tool_call["type"]

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
    detected by the message's own `tool_calls` field — and stripped on plain
    follow-up turns. Keying off `tool_calls` rather than a trailing `tool`-role
    message means reasoning is still replayed for a turn that requested a tool
    call even if the tool was never executed. Non-assistant messages and
    assistant messages without reasoning always return `False`.

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
    for msg in messages:
        if msg.role != "assistant" or not msg.thinking:
            flags.append(False)
            continue
        # The turn "had a tool call" iff the assistant message itself carries
        # tool calls — independent of whether the tool was later executed.
        flags.append(bool(msg.tool_calls))
    return flags


async def prefetch_audio_urls(messages: list[Message]) -> None:
    """Warm the audio download cache for any `AudioUrlBlock` in `messages`.

    No provider accepts audio by URL, so `message_to_openai_message` resolves such
    blocks to inline base64. That resolution is blocking, and the serializer is sync;
    awaiting this first moves the fetch onto a worker thread so the event loop is not
    stalled, leaving the serializer's call a cache hit. Mirrors how the Ollama backend
    offloads `ImageUrlBlock` downloads with `asyncio.to_thread`.

    Safe to call when there is nothing to fetch, and safe to skip — the serializer still
    produces correct output either way, just with a blocking download.

    Args:
        messages: The messages about to be serialised.

    Raises:
        ValueError: If a URL cannot be downloaded or exceeds the size cap.
    """
    pending = [
        a for m in messages for a in (m.audio or []) if isinstance(a, AudioUrlBlock)
    ]
    if not pending:
        return
    await asyncio.gather(
        *(asyncio.to_thread(a.resolve_base64) for a in pending), return_exceptions=False
    )


def message_to_openai_message(
    msg: Message,
    formatter: Formatter | None = None,
    *,
    replay_reasoning: bool = False,
    provider: str = "openai",
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
        provider: The calling backend's provider string, used to match the message's
            `provider_fields` declaration. Defaults to `"openai"`; OpenAI-compatible
            callers pass their own provider (`"litellm"`, `"watsonx"`, `"huggingface"`)
            so an `"openai"`-family declaration still reaches the wire.

    Returns:
        A dict with `"role"` and `"content"` fields. When the message carries
        images or audio, `"content"` is a list of content-part dicts; otherwise
        is a plain string. For tool-only assistant turns, `"content"` is `None`
        and `"tool_calls"` carries the structured call list. When content is
        present alongside tool calls, both keys are included. For a tool-result
        turn whose message carries a provider-supplied `tool_call_id` (a
        `ToolMessage`, which forwards it from its `ModelToolCall`, or a plain
        `Message` that declared it directly), the dict also carries
        `"tool_call_id"` (matching the assistant tool call), as spec-strict
        OpenAI-compatible providers require on `role: "tool"` messages. When
        `replay_reasoning` is `True` and reasoning is present, the dict also
        carries a `"reasoning_content"` field.

    An `AudioUrlBlock` is resolved to inline base64 here, because the OpenAI Chat
    Completions audio schema has no audio-by-URL content part. That resolution is
    blocking; callers should `await prefetch_audio_urls` first so it is a cache hit.

    Raises:
        ValueError: If an `AudioUrlBlock`'s audio cannot be downloaded or exceeds the
            size cap (see `AudioUrlBlock.resolve_base64`).
        ValueError: If the message's `provider_fields` names a target that does not
            match `provider` and includes no `"*"` (see `merge_provider_fields`).
        TypeError: If `provider_fields` is not a `dict`, or a matching value within
            it is not a `dict` (see `merge_provider_fields`).
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
                    # OpenAI Chat Completions has no audio-by-URL content part, so
                    # resolve it to inline base64 the way Ollama does for images.
                    # Normally a cache hit: `prefetch_audio_urls` warms it off-thread
                    # before serialisation. Falls back to a blocking fetch if some
                    # caller path did not prefetch.
                    parts.append(
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio.resolve_base64(),
                                "format": audio.format,
                            },
                        }
                    )

        result: dict[str, Any] = {"role": msg.role, "content": parts}
    else:
        result = {"role": msg.role, "content": content}

    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        result["tool_calls"] = tool_calls
        if msg.images is None and not content:
            result["content"] = None

    # Tool-result turns: spec-strict OpenAI-compatible providers require a
    # `role: "tool"` message to reference the assistant's originating tool call
    # via `tool_call_id` (issue #1389). Emit it when the message carries a
    # provider-supplied id. Both a `ToolMessage` (which forwards its
    # `ModelToolCall.tool_call_id` to the base `Message` at construction) and a
    # plain `Message` built from a component that declared `role="tool"` +
    # `tool_call_id` in its template representation expose it via `msg.tool_call_id`.
    # The optional `name` field is deliberately omitted: it is a legacy carryover
    # from `role: "function"`, not part of the OpenAI tool-message schema, and is
    # not required to satisfy the result-turn contract. Gate on truthiness
    # (matching `build_tool_calls` below) so an empty id is treated as absent.
    if msg.tool_call_id:
        result["tool_call_id"] = msg.tool_call_id

    if replay_reasoning and msg.thinking:
        result["reasoning_content"] = msg.thinking

    return merge_provider_fields(result, msg.provider_fields, provider)


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
        and isinstance(output.tool_calls, list)
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
    for model_tool_call in output.tool_calls:
        # Reuse the provider-supplied call id when one exists so a downstream
        # tool-result turn (which reads the same id via `_tool.tool_call_id`) can
        # reference the same call — spec-strict providers reject a mismatch
        # (issue #1389). The live case is `cli/serve` forwarding an upstream id
        # back downstream; on the openai/litellm paths the assistant turn's tool
        # calls come straight from the provider payload rather than through here.
        # Fall back to a fabricated id only when no id was supplied (e.g.
        # raw-string tool parsing).
        tool_call_id = model_tool_call.tool_call_id or f"call_{uuid.uuid4().hex[:24]}"

        # Serialize arguments to JSON with str fallback for non-serializable types
        args_json = json.dumps(model_tool_call.args, default=str)

        tool_call: ToolCallDict = {
            "id": tool_call_id,
            "type": "function",
            "function": {"name": model_tool_call.name, "arguments": args_json},
        }
        tool_calls.append(tool_call)

    return tool_calls
