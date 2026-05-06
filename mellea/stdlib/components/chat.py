"""Chat primitives: the ``Message`` and ``ToolMessage`` components.

Defines ``Message``, the ``Component`` subtype used to represent a single turn in a
chat history with a ``role`` (``user``, ``assistant``, ``system``, or ``tool``),
text ``content``, and optional ``images`` and ``documents`` attachments. Also provides
``ToolMessage`` (a ``Message`` subclass that carries the tool name and arguments), and
utilities for converting a ``Context`` into a flat list of ``Message`` objects:
``as_chat_history`` (strict typing) and ``as_generic_chat_history`` (flexible with
configurable formatter).
"""

import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal

from ...core import (
    CBlock,
    Component,
    Context,
    ImageBlock,
    ModelOutputThunk,
    ModelToolCall,
    TemplateRepresentation,
)
from .docs.document import Document, _coerce_to_documents

_logger = logging.getLogger(__name__)


class Message(Component["Message"]):
    """A single Message in a Chat history.

    Args:
        role (str): The role that this message came from (e.g., ``"user"``,
            ``"assistant"``).
        content (str): The content of the message.
        images (list[ImageBlock] | None): Optional images associated with the
            message.
        documents (list[Document] | None): Optional documents associated with
            the message.

    Attributes:
        Role (type): Type alias for the allowed role literals: ``"system"``,
            ``"user"``, ``"assistant"``, or ``"tool"``.
    """

    Role = Literal["system", "user", "assistant", "tool"]

    def __init__(
        self,
        role: "Message.Role",
        content: str,
        *,
        images: None | list[ImageBlock] = None,
        documents: None | Iterable[str | Document] = None,
    ):
        """Initialize a Message with a role, text content, and optional images and documents."""
        self.role = role
        self.content = content  # TODO this should be private.
        self._content_cblock = CBlock(self.content)
        self._images = images
        self._docs = _coerce_to_documents(documents)

    @property
    def images(self) -> None | list[str]:
        """Returns the images associated with this message as list of base 64 strings."""
        if self._images is not None:
            return [str(i) for i in self._images]
        return None

    def parts(self) -> list[Component | CBlock]:
        """Return the constituent parts of this message, including content, documents, and images.

        Returns:
            list[Component | CBlock]: A list beginning with the content block,
            followed by any attached documents and image blocks.
        """
        parts: list[Component | CBlock] = [self._content_cblock]
        if self._docs is not None:
            parts.extend(self._docs)
        if self._images is not None:
            parts.extend(self._images)
        return parts

    def format_for_llm(self) -> TemplateRepresentation:
        """Formats the content for a Language Model.

        Returns:
            The formatted output suitable for language models.
        """
        return TemplateRepresentation(
            obj=self,
            args={
                "role": self.role,
                "content": self._content_cblock,
                "images": self._images,
                "documents": self._docs,
            },
            template_order=["*", "Message"],
        )

    def __repr__(self) -> str:
        """Pretty representation of messages, because they are a special case."""
        images = []
        if self.images is not None:
            images = [f"{i[:20]}..." for i in self.images]

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
        return f'mellea.Message(role="{self.role}", content="{self.content}", images="{images}", documents="{docs}")'

    def _parse(self, computed: ModelOutputThunk) -> "Message":
        """Parse the model output into a Message."""
        # TODO: There's some specific logic for tool calls. Storing that here for now.
        # We may eventually need some generic parsing logic that gets run for all Component types...
        if computed.tool_calls is not None:
            # A tool was successfully requested.
            # Assistant responses for tool calling differ by backend. For the default formatter,
            # we put all of the function data into the content field in the same format we received it.

            # Chat backends should provide an openai-like object in the _meta chat response, which we can use to properly format this output.
            if "chat_response" in computed._meta:
                # Ollama.
                return Message(
                    role=computed._meta["chat_response"].message.role,
                    content=str(computed._meta["chat_response"].message.tool_calls),
                )
            elif "oai_chat_response" in computed._meta:
                # OpenAI and Watsonx.
                return Message(
                    role=computed._meta["oai_chat_response"]["choices"][0]["message"][
                        "role"
                    ],
                    content=str(
                        computed._meta["oai_chat_response"]["choices"][0][
                            "message"
                        ].get("tool_calls", [])
                    ),
                )
            else:
                # HuggingFace (or others). There are no guarantees on how the model represented the function calls.
                # Output it in the same format we received the tool call request.
                assert computed.value is not None
                return Message(role="assistant", content=computed.value)

        if "chat_response" in computed._meta:
            # Chat backends should provide an openai-like object in the _meta chat response, which we can use to properly format this output.
            return Message(
                role=computed._meta["chat_response"].message.role,
                content=computed._meta["chat_response"].message.content,
            )
        elif "oai_chat_response" in computed._meta:
            role = (
                computed._meta["oai_chat_response"]
                .get("choices", [{}])[0]
                .get("message", {})
                .get("role", "")
            )
            if role == "":
                role = (
                    computed._meta["oai_chat_response"]
                    .get("message", {})
                    .get("role", "")
                )

            content = (
                computed._meta["oai_chat_response"]
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            if content == "":
                content = (
                    computed._meta["oai_chat_response"]
                    .get("message", {})
                    .get("content", "")
                )
            return Message(role=role, content=content)
        else:
            assert computed.value is not None
            return Message(role="assistant", content=computed.value)


class ToolMessage(Message):
    """Adds the name field for function name.

    Args:
        role (str): The role of this message; most backends use ``"tool"``.
        content (str): The content of the message; should be a stringified
            version of ``tool_output``.
        tool_output (Any): The output of the tool or function call.
        name (str): The name of the tool or function that was called.
        args (Mapping[str, Any]): The arguments passed to the tool.
        tool (ModelToolCall): The ``ModelToolCall`` representation.

    Attributes:
        arguments (Mapping[str, Any]): The arguments that were passed to the
            tool; stored from the ``args`` constructor parameter.
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
        super().__init__(role, content)
        self.name = name
        self.arguments = args
        self._tool_output = tool_output
        self._tool = tool

    def format_for_llm(self) -> TemplateRepresentation:
        """Return the same representation as ``Message`` with a ``name`` field added to the args dict.

        Returns:
            TemplateRepresentation: Template representation including the tool
            name alongside the standard message fields.
        """
        message_repr = super().format_for_llm()
        args = message_repr.args
        args["name"] = self.name

        return TemplateRepresentation(
            obj=self, args=args, template_order=["*", "Message"]
        )

    def __repr__(self) -> str:
        """Pretty representation of messages, because they are a special case."""
        return f'mellea.ToolMessage(role="{self.role}", content="{self.content}", name="{self.name}")'


def as_chat_history(ctx: Context) -> list[Message]:
    """Returns a list of Messages corresponding to a Context.

    Args:
        ctx: A linear ``Context`` whose entries are ``Message`` or ``ModelOutputThunk``
            objects with ``Message`` parsed representations.

    Returns:
        List of ``Message`` objects in conversation order.

    Raises:
        Exception: If the context history is non-linear and cannot be cast to a
            flat list.
        AssertionError: If any entry in the context cannot be converted to a
            ``Message``.
    """

    def _to_msg(c: CBlock | Component | ModelOutputThunk) -> Message | None:
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
        raise Exception("Trying to cast a non-linear history into a chat history.")
    else:
        history = [_to_msg(c) for c in all_ctx_events]
        assert None not in history, "Could not render this context as a chat history."
        return history  # type: ignore


def _default_formatter(obj: Any) -> str:
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

    This function is more permissive than ``as_chat_history()``, allowing arbitrary
    component types. Unknown types are converted to strings using a configurable
    formatter, making it suitable for general-purpose use where context composition
    may be heterogeneous.

    Args:
        ctx: A linear ``Context`` that may contain ``Message``, ``ModelOutputThunk``,
            or other ``Component`` types.
        formatter: Optional callable that converts unknown types to strings.
            Defaults to ``_default_formatter`` which logs a warning and stringifies.

    Returns:
        List of ``Message`` objects in conversation order.

    Raises:
        Exception: If the context history is non-linear and cannot be cast to a
            flat list.
    """
    if formatter is None:
        formatter = _default_formatter

    def _to_msg(c: CBlock | Component | ModelOutputThunk) -> Message:
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
            case _:
                content = formatter(c)
                return Message(role="user", content=content)

    all_ctx_events = ctx.as_list()
    if all_ctx_events is None:
        raise Exception("Trying to cast a non-linear history into a chat history.")
    return [_to_msg(c) for c in all_ctx_events]
