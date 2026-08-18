# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chat primitives: the `Message` and `ToolMessage` components.

Defines `Message`, the `Component` subtype used to represent a single turn in a
chat history with a `role` (`user`, `assistant`, `system`, or `tool`),
text `content`, and optional `images` and `documents` attachments. Also provides
`ToolMessage` (a `Message` subclass that carries the tool name and arguments), and
utilities for converting a `Context` into a flat list of `Message` objects:
`as_chat_history` (strict typing) and `as_generic_chat_history` (flexible with
configurable formatter).
"""

import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal, cast, get_args

from ...core import (
    AudioBlock,
    AudioUrlBlock,
    CBlock,
    Component,
    Context,
    ImageBlock,
    ImageUrlBlock,
    ModelOutputThunk,
    ModelToolCall,
    Span,
    TemplateRepresentation,
)
from .docs.document import Document, _coerce_to_documents

_logger = logging.getLogger(__name__)


class Message(Component["Message"]):
    """A single Message in a Chat history.

    Args:
        role (str): The role that this message came from (e.g., `"user"`,
            `"assistant"`).
        content (str): The content of the message.
        images (list[ImageBlock | ImageUrlBlock] | None): Optional images
            associated with the message. Use `ImageBlock` for base64-encoded
            images (supported by all vision backends) or `ImageUrlBlock` for
            URL-referenced images (passed directly to OpenAI-compatible
            backends; backends that require base64, such as Ollama, download
            and encode the image automatically).
        audio (list[AudioBlock | AudioUrlBlock] | None): Optional audio
            associated with the message.
        documents (list[Document] | None): Optional documents associated with
            the message.
        tool_calls (list[dict[str, Any]] | None): Optional OpenAI-compatible
            assistant tool calls associated with the message.
        tool_call_id (str | None): Optional provider-supplied tool-call id that a
            `role="tool"` message references, so a tool-result turn can be linked
            back to the assistant tool call that produced it (issue #1389).
            `ToolMessage` sources this from its `ModelToolCall` instead; set it
            directly only when constructing a bare `role="tool"` `Message`.
        tool_name (str | None): Optional name of the tool whose result a
            `role="tool"` message carries. Some backends (e.g. Ollama) key their
            tool-result turn on the tool name rather than a call id. A component
            declaring `role="tool"` can supply it directly; `ToolMessage` instead
            carries it as `.name`.
        thinking (str | None): Optional reasoning trace produced by a thinking
            model on the turn that generated this message. Populated by `_parse`
            from `ModelOutputThunk.thinking`; carried through `as_chat_history`
            so backends can round-trip it on subsequent turns per their replay
            policy. `None` or empty for messages that carry no reasoning (e.g.
            user turns, or assistant turns from non-thinking models); the replay
            policy and serializers treat both falsy cases identically.

    Attributes:
        Role (type): Type alias for the allowed role literals: `"system"`,
            `"user"`, `"assistant"`, or `"tool"`.
    """

    Role = Literal["system", "user", "assistant", "tool"]

    def __init__(
        self,
        role: "Message.Role",
        content: str,
        *,
        images: None | list[ImageBlock | ImageUrlBlock] = None,
        audio: None | list[AudioBlock | AudioUrlBlock] = None,
        documents: None | Iterable[str | Document] = None,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        thinking: str | None = None,
    ):
        """Initialize a Message with a role, text content, optional images, audio, documents, tool calls, an optional tool-call id, an optional tool name, and an optional reasoning trace."""
        if role not in get_args(Message.Role):
            raise ValueError(
                f"Invalid role {role!r}. Must be one of: {list(get_args(Message.Role))}"
            )
        self.role = role
        self.content = content  # TODO this should be private.
        self.thinking = thinking
        self._content_cblock = CBlock(self.content)
        self._images = images
        self._audio = audio
        self._docs = _coerce_to_documents(documents)
        self._tool_calls = tool_calls
        self._tool_call_id = tool_call_id
        self._tool_name = tool_name

    @property
    def images(self) -> None | list[ImageBlock | ImageUrlBlock]:
        """Returns the images associated with this message."""
        return self._images

    @property
    def audio(self) -> None | list[AudioBlock | AudioUrlBlock]:
        """Returns the audio associated with this message."""
        return self._audio

    @property
    def tool_calls(self) -> list[dict[str, Any]] | None:
        """Returns the OpenAI-compatible tool calls associated with this message."""
        return self._tool_calls

    @property
    def tool_call_id(self) -> str | None:
        """Returns the tool-call id this `role="tool"` message references, if any."""
        return self._tool_call_id

    @property
    def tool_name(self) -> str | None:
        """Returns the name of the tool whose result this `role="tool"` message carries, if any."""
        return self._tool_name

    def parts(self) -> list[Span]:
        """Return the constituent parts of this message, including content, documents, images, and audio.

        Returns:
            list[Span]: A list beginning with the content block,
            followed by any attached documents, image blocks, and audio blocks.
        """
        parts: list[Span] = [self._content_cblock]
        if self._docs is not None:
            parts.extend(self._docs)
        if self._images is not None:
            parts.extend(self._images)
        if self._audio is not None:
            parts.extend(self._audio)
        return parts

    def format_for_llm(self) -> TemplateRepresentation:
        """Formats the content for a Language Model.

        Declares this message's `role` and carries its `thinking`, `tool_calls`,
        and `tool_call_id` so the representation is a faithful, self-describing
        view of the message (see `message_from_template_representation`).

        Returns:
            The formatted output suitable for language models.
        """
        return TemplateRepresentation(
            obj=self,
            args={"content": self._content_cblock, "documents": self._docs},
            template_order=["*", "Message"],
            images=self._images,
            audio=self._audio,
            role=self.role,
            thinking=self.thinking,
            tool_calls=self._tool_calls,
            tool_call_id=self._tool_call_id,
            tool_name=self._tool_name,
        )

    def __repr__(self) -> str:
        """Pretty representation of messages, because they are a special case."""
        images = []
        if self._images is not None:
            images = [f"{str(i.value)[:20]}..." for i in self._images]

        audio = []
        if self._audio is not None:
            audio = [f"{str(a.value)[:20]}..." for a in self._audio]

        docs = []
        if self._docs is not None:
            # Do a quick format of each document.
            docs = [
                # Equivalent to: "[Document <ID>] <TITLE>: <TEXT>...".
                f"[Document{' ' + str(doc.doc_id) if doc.doc_id else ''}] {str(doc.title) + ': ' if doc.title else ''}{doc.text}"[
                    :20
                ]
                + "..."
                for doc in self._docs
            ]
        return f'mellea.Message(role="{self.role}", content="{self.content}", images="{images}", audio="{audio}", documents="{docs}")'

    def _parse(self, computed: ModelOutputThunk) -> "Message":
        """Parse the model output into a Message."""
        # TODO: There's some specific logic for tool calls. Storing that here for now.
        # We may eventually need some generic parsing logic that gets run for all Component types...
        provider = computed.raw.provider
        response = computed.raw.response

        if computed.tool_calls is not None:
            # A tool was successfully requested.
            # Assistant responses for tool calling differ by backend. Preserve
            # OpenAI-compatible tool calls separately from message content when
            # the provider gives us structured tool-call data.
            # Carry any captured reasoning onto the tool-issuing assistant Message: this is the
            # exact turn the replay policy round-trips (see `should_replay_reasoning`), so the
            # reasoning must survive parsing here or the round-trip has nothing to replay.
            if provider == "ollama" and response is not None:
                from ...helpers.openai_compatible_helpers import build_tool_calls

                tool_calls = cast(
                    list[dict[str, Any]], build_tool_calls(computed) or []
                )
                thinking = (
                    computed.thinking if response.message.role == "assistant" else None
                )
                return Message(
                    role=response.message.role,
                    content=getattr(response.message, "content", "") or "",
                    tool_calls=tool_calls,
                    thinking=thinking,
                )
            if provider in ("openai", "watsonx", "litellm") and isinstance(
                response, dict
            ):
                choice = response["choices"][0]
                msg = choice["message"]
                thinking = computed.thinking if msg["role"] == "assistant" else None
                return Message(
                    role=msg["role"],
                    content=msg.get("content") or "",
                    tool_calls=msg.get("tool_calls") or None,
                    thinking=thinking,
                )
            # Hugging Face (or others). There are no guarantees on how the model represented the function calls.
            # Output it in the same format we received the tool call request.
            assert computed.value is not None
            return Message(
                role="assistant", content=computed.value, thinking=computed.thinking
            )

        # No tool call on this turn: carry any captured reasoning onto the parsed
        # assistant Message so it can round-trip on subsequent turns. `thinking` is
        # None for non-thinking models and for role="tool" recoveries.
        if provider == "ollama" and response is not None:
            # Ollama can return role="tool"; preserve role recovery from the response.
            thinking = (
                computed.thinking if response.message.role == "assistant" else None
            )
            return Message(
                role=response.message.role,
                content=response.message.content,
                thinking=thinking,
            )
        if provider in ("openai", "watsonx", "litellm") and isinstance(response, dict):
            msg = response["choices"][0].get("message", {})
            role = msg.get("role", "assistant")
            content = msg.get("content") or ""
            thinking = computed.thinking if role == "assistant" else None
            return Message(role=role, content=content, thinking=thinking)

        # Hugging Face: raw.response is token tensors with no role/content to parse.
        # Unknown provider: nothing to switch on. Both fall back to the decoded text.
        assert computed.value is not None
        return Message(
            role="assistant", content=computed.value, thinking=computed.thinking
        )


class ToolMessage(Message):
    """Adds the name field for function name.

    Args:
        role (str): The role of this message; most backends use `"tool"`.
        content (str): The content of the message; should be a stringified
            version of `tool_output`.
        tool_output (Any): The output of the tool or function call.
        name (str): The name of the tool or function that was called.
        args (Mapping[str, Any]): The arguments passed to the tool.
        tool (ModelToolCall): The `ModelToolCall` representation.

    Attributes:
        arguments (Mapping[str, Any]): The arguments that were passed to the
            tool; stored from the `args` constructor parameter.
    """

    def __init__(
        self,
        role: Message.Role,
        content: str,
        tool_output: Any,
        name: str,
        args: Mapping[str, Any],
        tool: ModelToolCall,
    ):
        """Initialize a ToolMessage with role, content, tool output, name, args, and tool call."""
        super().__init__(role, content, tool_call_id=tool.tool_call_id)
        self.name = name
        self.arguments = args
        self._tool_output = tool_output
        self._tool = tool

    def __repr__(self) -> str:
        """Pretty representation of messages, because they are a special case."""
        return f'mellea.ToolMessage(role="{self.role}", content="{self.content}", name="{self.name}")'


def message_from_template_representation(
    tr: TemplateRepresentation, *, default_role: "Message.Role", content: str
) -> Message:
    """Build a `Message` from a component's `TemplateRepresentation`.

    Shared by `ChatFormatter.to_chat_messages` and `as_generic_chat_history` so a
    component's declared role and tool metadata are honored consistently across both
    conversion paths. The representation's `role` overrides `default_role` when set;
    role validation is deferred to `Message`, which raises `ValueError` for anything
    outside `Message.Role`. `thinking`, `tool_calls`, and (for `role="tool"`)
    `tool_call_id`/`tool_name` are carried onto the resulting message.

    Args:
        tr: The template representation returned by the component's `format_for_llm`.
        default_role: The positional role guess to use when `tr.role` is `None`.
        content: The already-rendered text content for the message.

    Returns:
        A `Message` with the resolved role, content, attachments, and tool metadata.
    """
    role = tr.role if tr.role is not None else default_role
    return Message(
        role=role,  # type: ignore[arg-type]  # validated by Message.__init__
        content=content,
        images=tr.images,
        audio=tr.audio,
        tool_calls=tr.tool_calls,
        tool_call_id=tr.tool_call_id,
        tool_name=tr.tool_name,
        thinking=tr.thinking,
    )


def as_chat_history(ctx: Context) -> list[Message]:
    """Returns a list of Messages corresponding to a Context.

    Args:
        ctx: A linear `Context` whose entries are `Message` or `ModelOutputThunk`
            objects with `Message` parsed representations.

    Returns:
        List of `Message` objects in conversation order.

    Raises:
        ValueError: If the context history is non-linear and cannot be cast to a
            flat list.
        AssertionError: If any entry in the context cannot be converted to a
            `Message`.
    """

    def _to_msg(c: Span) -> Message | None:
        match c:
            case Message():
                return c
            case ModelOutputThunk():
                match c.parsed_repr:
                    case Message():
                        return c.parsed_repr
                    case _:
                        return None
            case _:
                return None

    all_ctx_events = ctx.as_list()
    if all_ctx_events is None:
        raise ValueError("Trying to cast a non-linear history into a chat history.")
    else:
        history = [_to_msg(c) for c in all_ctx_events]
        assert None not in history, "Could not render this context as a chat history."
        return history  # type: ignore


def _default_formatter(obj: object) -> str:
    """Default formatter for unknown component types.

    Logs a warning and converts the object to a string representation.
    """
    _logger.warning(
        f"Unknown component type {type(obj).__name__} in as_generic_chat_history; "
        f"converting to string representation."
    )
    return str(obj)


def as_generic_chat_history(
    ctx: Context, formatter: Callable[[object], str] | None = None
) -> list[Message]:
    """Returns a list of Messages corresponding to a Context, with flexible type handling.

    This function is more permissive than `as_chat_history()`, allowing arbitrary
    component types. Unknown types are converted to strings using a configurable
    formatter, making it suitable for general-purpose use where context composition
    may be heterogeneous.

    The formatter is applied to:
    - `ModelOutputThunk` with non-Message `parsed_repr`
    - `CBlock` subclasses (subclasses only; plain `CBlock` is stringified)
    - Other unknown component types

    Existing `Message` objects are preserved as-is; their content is not formatted.
    This design preserves Message fidelity while providing an escape hatch for unknown types.

    Args:
        ctx: A linear `Context` that may contain `Message`, `ModelOutputThunk`,
            or other `Component` types.
        formatter: Optional callable that converts unknown types to strings.
            Defaults to `_default_formatter` which logs a warning and stringifies.

    Returns:
        List of `Message` objects in conversation order.

    Raises:
        ValueError: If the context history is non-linear and cannot be cast to a
            flat list.
    """
    if formatter is None:
        formatter = _default_formatter

    def _to_msg(c: Span) -> Message:
        match c:
            case Message():
                return c
            case ModelOutputThunk():
                if isinstance(c.parsed_repr, Message):
                    return c.parsed_repr
                if isinstance(c.parsed_repr, str):
                    return Message(role="assistant", content=c.parsed_repr)
                # Use value if parsed_repr is None
                if c.parsed_repr is None:
                    if c.value is None:
                        raise ValueError(
                            "ModelOutputThunk has no value and no parsed_repr — was it evaluated?"
                        )
                    content = str(c.value)
                else:
                    _logger.warning(
                        f"ModelOutputThunk.parsed_repr is {type(c.parsed_repr).__name__}, "
                        f"not a Message; falling back to value."
                    )
                    content = formatter(c.parsed_repr)
                return Message(role="assistant", content=content)
            case CBlock():
                if type(c) is not CBlock:
                    content = formatter(c)
                else:
                    content = str(c)
                return Message(role="user", content=content)
            case Component():
                # A component may declare its own role (and tool metadata) via its
                # template representation; honor it over the `user` default, keeping
                # this path consistent with `ChatFormatter.to_chat_messages`. Stay
                # permissive: some components (e.g. `Intrinsic`) intentionally raise
                # from `format_for_llm` because they are only ever the action, never
                # part of the rendered context. Fall back to stringifying them via
                # the formatter — the pre-existing behavior for arbitrary components.
                content = formatter(c)
                try:
                    tr = c.format_for_llm()
                except NotImplementedError:
                    tr = None
                if isinstance(tr, TemplateRepresentation):
                    return message_from_template_representation(
                        tr, default_role="user", content=content
                    )
                return Message(role="user", content=content)
            case _:
                content = formatter(c)
                return Message(role="user", content=content)

    all_ctx_events = ctx.as_list()
    if all_ctx_events is None:
        raise ValueError("Trying to cast a non-linear history into a chat history.")
    return [_to_msg(c) for c in all_ctx_events]
