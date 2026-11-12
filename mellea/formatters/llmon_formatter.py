# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`LLMONFormatter`: sends components to models trained on LLMON.

LLMON (LLM Object Notation, arXiv:2603.22519) marks a prompt up with special-token
blocks, so an LLMON-trained model can tell what it should do (`instr`) from what it
should only read (`data`, `doc`) and what its answer must satisfy (`req`). The
granite LLMON models' chat template writes that markup itself from the fields
of each chat message: a user message's `content` and `instruction` items become
`instr` blocks, its `data` items `data` blocks, its `requirement` items `req` blocks,
and its `documents` `doc` blocks, each wired into the `exec` block the template adds.
`LLMONFormatter` maps an `Instruction`'s `format_for_llm()` args onto those fields,
sends a requirement's LLM-as-a-judge check as an instruction with what it judges as
data, and sends every other turn as its plain text, so it writes no LLMON itself.
Content is sanitized by the `llmon` library first, since the template inserts it as
it is.

Only `LocalHFBackend` supports LLMON, since only it hands the chat template each
message with the fields the template reads; every other backend raises `ValueError`
when given an LLMON model or this formatter. `LocalHFBackend` uses this formatter by
default for the LLMON models registered here (see `default_formatter`); pass one to it
explicitly to use it with any other model. It needs the `llmon` extra:
`pip install "mellea[llmon]"`.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Iterable
from typing import Any, cast

from ..backends.model_ids import (
    IBM_GRANITE_4_1_3B_LLMON_PREVIEW,
    IBM_GRANITE_4_1_8B_LLMON_PREVIEW,
    ModelIdentifier,
)
from ..core import (
    CBlock,
    Component,
    ModelOutputThunk,
    Requirement,
    Span,
    TemplateRepresentation,
)
from ..stdlib.components.chat import Message, message_from_template_representation
from ..stdlib.components.docs.document import Document
from ..stdlib.components.instruction import Instruction
from .chat_formatter import ChatFormatter
from .template_formatter import TemplateFormatter

# Models that speak LLMON, matched exactly against a model id. Only `LocalHFBackend`
# runs them; other backends raise for them. The catalog constants in
# `mellea.backends.model_ids` are the single source of truth; every provider name they
# carry counts as an LLMON model id.
_LLMON_MODEL_IDS = (IBM_GRANITE_4_1_3B_LLMON_PREVIEW, IBM_GRANITE_4_1_8B_LLMON_PREVIEW)
_LLMON_MODELS = frozenset(
    getattr(model_id, f.name)
    for model_id in _LLMON_MODEL_IDS
    for f in dataclasses.fields(model_id)
    if getattr(model_id, f.name) is not None
    and isinstance(getattr(model_id, f.name), str)
)

# Only the HuggingFace backend hands each message to the chat template as it is.
# OpenAI-compatible servers such as vLLM rebuild every message from its standard keys
# and drop the rest, so the template's fields are declared for HuggingFace alone, the
# one backend that takes this formatter.
_TEMPLATE_FIELDS_PROVIDER = "huggingface"
# The instruction for a requirement's LLM-as-a-judge check: the wording of Mellea's
# `Requirement` template, with the output and the requirement sent as labeled `data`
# items rather than inline. A requirement in a `req` item would instead constrain the
# judge's own reply.
_JUDGE_INSTRUCTION = (
    "Check whether the output satisfies the requirement. "
    "Reply with 'yes' if the requirement is satisfied and 'no' otherwise. "
    "Do not include any other text in your response."
)
# A tool call as the model writes it in its reply, with the whitespace before it.
_WRITTEN_TOOL_CALL = re.compile(r"\s*<tool_call>.*?</tool_call>", re.DOTALL)


def is_llmon_model(model_id: str | ModelIdentifier | None) -> bool:
    """Return whether a model is a registered LLMON model.

    Args:
        model_id: A model id, or a `ModelIdentifier`, which matches when any of its
            provider names is registered.

    Returns:
        `True` for a registered LLMON model, otherwise `False`.
    """
    if isinstance(model_id, ModelIdentifier):
        names = [getattr(model_id, f.name) for f in dataclasses.fields(model_id)]
    else:
        names = [model_id]
    return any(name in _LLMON_MODELS for name in names)


def default_formatter(model_id: str | ModelIdentifier) -> ChatFormatter:
    """Return the formatter `LocalHFBackend` uses when it is not given one.

    Args:
        model_id: The model id the backend was constructed with.

    Returns:
        An `LLMONFormatter` for a registered LLMON model, otherwise a
        `TemplateFormatter`.
    """
    if is_llmon_model(model_id):
        return LLMONFormatter(model_id)
    return TemplateFormatter(model_id=model_id)


class _LLMONMessage(Message):
    """A chat message whose content `LLMONFormatter` has already rendered.

    Backends print every message from `to_chat_messages` again to serialize it;
    marking these lets `LLMONFormatter.print` return the content as it is instead of
    rendering it a second time.
    """


@dataclasses.dataclass
class _Turn:
    """One chat turn: its role, its sanitized text, and its chat template fields."""

    role: str
    content: str
    fields: dict[str, list[dict[str, Any]]]
    rep: TemplateRepresentation | None


class LLMONFormatter(ChatFormatter):
    """Formatter that sends components to LLMON-trained models through their chat template.

    The granite LLMON models' chat template writes the LLMON markup from each
    message's fields, so this formatter fills those fields rather than writing LLMON.
    An `Instruction` becomes a user message from its `format_for_llm()` args: its
    prefix and description, joined, are the message's content, which the template
    takes as its instruction; a repair (the feedback `RepairTemplateStrategy` adds
    after a failed check) is a further `instruction` item; each requirement is a
    `requirement` item; each in-context
    example and each grounding-context value is a `data` item, a `Document` as its
    text alone. Every item is wired into the turn's
    `exec` block. A user `Message` is its content, with its documents as `documents`
    items, and a `CBlock` is its text, as the instruction. A `Requirement` being
    validated (its LLM-as-a-judge check) is an instruction to judge the output, with
    the output and the requirement as labeled `data` items: in a `requirement` item,
    the requirement would constrain the judge's own reply. Any other component raises.

    System, assistant, and tool turns are sent as their text alone (a message's
    content, or a model output's reply), with their tool calls and tool-call id as
    message fields, because the template wraps each in its own LLMON block and writes
    an assistant's `tool_calls` out itself. When a reply holds tool calls the model
    wrote out (as the HuggingFace backend leaves them) and the backend parsed every
    one, they are moved from the text into `tool_calls`, so the template also wires
    that assistant turn into the next `exec` as an input.

    All text is sanitized with the `llmon` library's `LLMONConverters.sanitize`, which
    replaces each LLMON special token with a look-alike that is not one, because the
    template inserts content as it is.

    The template's fields reach it only through `LocalHFBackend`, which hands it each
    message as it is, so they are declared in the message's `provider_fields` for that
    backend alone, and it is the only backend that takes this formatter: every other
    backend raises `ValueError` when given it or a registered LLMON model. `print`
    raises for a turn with fields, since a single string cannot carry them. A reply
    needs no extracting: `LocalHFBackend` decodes the generation without special
    tokens, LLMON's among them, so the output's value is the model's text.

    Args:
        model_id (str | ModelIdentifier): The model the prompts are for.

    Example:
        ```python
        from mellea.formatters import LLMONFormatter
        from mellea.stdlib.components import Instruction

        formatter = LLMONFormatter("ibm-granite/granite-4.1-3b-llmon-preview")
        (message,) = formatter.to_chat_messages(
            [Instruction("Summarize the review.", grounding_context={"review": "..."})]
        )
        ```
    """

    def __init__(self, model_id: str | ModelIdentifier):
        """Initialize the formatter for a model and load the `llmon` sanitizer.

        Raises:
            ImportError: If the `llmon` library is not installed.
        """
        try:
            from llmon.lang.json.llmon_converters import LLMONConverters
        except ImportError as e:
            raise ImportError(
                "LLMONFormatter requires the `llmon` library. "
                'Please install it with: pip install "mellea[llmon]"'
            ) from e

        self.model_id = model_id
        self._sanitize: Callable[[str], str] = LLMONConverters().sanitize

    def print(self, c: Span) -> str:
        """Render the text a turn is sent as.

        That is an instruction's prefix and description, a message's content, a
        content block's text, or a model output's reply. A message that
        `to_chat_messages` already rendered is returned as it is. Any other value,
        such as a tool's output that `call_tools` prints for its tool message, is not
        a turn and is returned as plain text. All text is sanitized.

        Args:
            c (Span): The content block, model output, instruction, or message to
                render.

        Returns:
            str: The text of the turn.

        Raises:
            TypeError: If `c` is a component other than an `Instruction`, a
                `Message`, or a `Requirement` being validated, or an instruction holds
                one in its args.
            ValueError: If `c` has parts that reach the model only as chat template
                fields (an instruction's requirements, examples, grounding context,
                or repair, a user message's documents, or what a requirement judges),
                or is an instruction with an output prefix.
        """
        if isinstance(c, _LLMONMessage):
            return c.content
        if not isinstance(c, Component | CBlock):
            return "" if c is None else self._sanitize(str(c))
        turn = self._turn(c)
        if turn.fields:
            raise ValueError(
                f"This {type(c).__name__} has parts ({', '.join(turn.fields)}) that "
                "reach an LLMON model only as chat template message fields, so it "
                "cannot be printed as one string; send it as a chat message instead."
            )
        return turn.content

    def to_chat_messages(self, cs: list[Span]) -> list[Message]:
        """Convert a linearized chat history into messages for the LLMON chat template.

        Each item becomes one message, as described for the class: roles and
        attachments are resolved as in `ChatFormatter.to_chat_messages`, the turn's
        text is the content, and the parts the template takes as fields are declared
        in the message's `provider_fields` for the HuggingFace backend. `print`
        returns the content of the resulting messages unchanged.

        Args:
            cs (list[Span]): The linearized sequence of context components, content
                blocks, and model outputs to convert.

        Returns:
            list[Message]: One message per item in `cs`, in order.

        Raises:
            TypeError: If an item is a component other than an `Instruction`, a
                `Message`, or a `Requirement` being validated, or an instruction holds
                one in its args.
            ValueError: If an item is an instruction with an output prefix.
        """
        messages: list[Message] = []
        for c in cs:
            turn = self._turn(c)
            content, tool_calls = turn.content, None
            if isinstance(c, ModelOutputThunk) and not (
                turn.rep and turn.rep.tool_calls
            ):
                content, tool_calls = _move_written_tool_calls(c, content)
            provider_fields = _with_template_fields(
                turn.rep.provider_fields if turn.rep else None, turn.fields
            )
            if turn.rep is None:
                messages.append(
                    _LLMONMessage(
                        role=cast(Message.Role, turn.role),
                        content=content,
                        tool_calls=tool_calls,
                        provider_fields=provider_fields,
                    )
                )
                continue
            rep = dataclasses.replace(turn.rep, provider_fields=provider_fields)
            if tool_calls is not None:
                rep = dataclasses.replace(rep, tool_calls=tool_calls)
            messages.append(
                message_from_template_representation(
                    rep,
                    default_role=cast(Message.Role, turn.role),
                    content=content,
                    message_cls=_LLMONMessage,
                )
            )
        return messages

    def _turn(self, c: Span) -> _Turn:
        """Resolve one turn's role, sanitized text, and chat template fields."""
        if isinstance(c, ModelOutputThunk):
            assert c.is_computed(), "a model output must be computed to be rendered"
            # A chat reply's parsed `Message` carries the turn's tool metadata.
            parsed = c.parsed_repr
            rep = _representation(parsed) if isinstance(parsed, Message) else None
            return _Turn(
                _role(rep, "assistant"), self._sanitize(c.value or ""), {}, rep
            )
        if isinstance(c, CBlock):
            return _Turn("user", self._sanitize(c.value or ""), {}, None)
        if isinstance(c, Instruction):
            return self._instruction_turn(_representation(c))
        if isinstance(c, Message):
            rep = _representation(c)
            role = _role(rep, "user")
            fields: dict[str, list[dict[str, Any]]] = {}
            # The chat template reads documents on user messages only.
            if role == "user":
                documents = cast(list[Document] | None, rep.args.get("documents"))
                fields["documents"] = [self._document(d) for d in documents or []]
            return _Turn(role, self._sanitize(c.content), _non_empty(fields), rep)
        if isinstance(c, Requirement):
            return self._judge_turn(_representation(c))
        raise TypeError(
            "LLMONFormatter sends only instructions, messages, requirements being "
            "validated, content blocks, and model outputs to an LLMON model; got a "
            f"{type(c).__name__}."
        )

    def _judge_turn(self, rep: TemplateRepresentation) -> _Turn:
        """Map a requirement's LLM-as-a-judge check onto an instruction and its data."""
        args = cast(dict[str, Any], rep.args)
        # A chat reply is judged by its parsed `Message`.
        output = args.get("output")
        text = output.content if isinstance(output, Message) else _plain_text(output)
        description = _plain_text(args.get("description"))
        fields = {
            "data": self._items([f"Output: {text}", f"Requirement: {description}"])
        }
        return _Turn(_role(rep, "user"), _JUDGE_INSTRUCTION, fields, rep)

    def _instruction_turn(self, rep: TemplateRepresentation) -> _Turn:
        """Map an instruction's args onto the chat template's user-message fields.

        Every grounding-context value is a `data` item labeled with its key, as
        `key: value`; a `Document` is its text alone, without its title.
        """
        args = cast(dict[str, Any], rep.args)
        if _plain_text(args.get("output_prefix")):
            raise ValueError(
                "An LLMON chat template cannot start the reply with a prefix, so an "
                "Instruction's output_prefix cannot be sent to an LLMON model."
            )
        instruction = "\n\n".join(
            text
            for text in (
                _plain_text(args.get("prefix")),
                _plain_text(args.get("description")),
            )
            if text
        )
        grounding = []
        for key, value in (args.get("grounding_context") or {}).items():
            text = _plain_text(value.text if isinstance(value, Document) else value)
            if text:
                grounding.append(f"{key}: {text}")
        fields = {
            "instruction": self._items([args.get("repair")]),
            "data": self._items([*(args.get("icl_examples") or []), *grounding]),
            "requirement": self._items(args.get("requirements") or []),
        }
        return _Turn(
            _role(rep, "user"), self._sanitize(instruction), _non_empty(fields), rep
        )

    def _items(self, values: Iterable[object]) -> list[dict[str, Any]]:
        """Build a chat template item for each value with text, wired into the `exec`."""
        texts = (_plain_text(value) for value in values)
        return [{"value": self._sanitize(text), "exec": True} for text in texts if text]

    def _document(self, document: Document) -> dict[str, Any]:
        """Build a chat template `documents` item, wired into the `exec`."""
        item: dict[str, Any] = {}
        if document.title:
            item["title"] = self._sanitize(document.title)
        item["text"] = self._sanitize(document.text or "")
        item["exec"] = True
        return item


def _representation(component: Component) -> TemplateRepresentation:
    """Return an instruction's or a message's `TemplateRepresentation`."""
    rep = component.format_for_llm()
    assert isinstance(rep, TemplateRepresentation), (
        f"a {type(component).__name__} must format to a TemplateRepresentation"
    )
    return rep


def _role(rep: TemplateRepresentation | None, default: str) -> str:
    """Return the role a representation declares, else `default`."""
    return rep.role if rep is not None and rep.role is not None else default


def _plain_text(value: object) -> str:
    """Return a value from an instruction's args as text.

    Raises:
        TypeError: If `value` is a component, which has no text of its own.
    """
    match value:
        case None:
            return ""
        case str():
            return value
        case CBlock():
            return value.value or ""
        case Component():
            raise TypeError(
                f"LLMONFormatter cannot send a {type(value).__name__} in this part "
                "of an instruction to an LLMON model: only text, content blocks, and "
                "model outputs can be sent, and documents in the grounding context."
            )
        case _:
            return str(value)


def _non_empty(
    fields: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Drop the fields that have no items."""
    return {name: items for name, items in fields.items() if items}


def _with_template_fields(
    provider_fields: dict[str, dict[str, Any]] | None,
    fields: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]] | None:
    """Declare a turn's chat template fields in its message's `provider_fields`."""
    if not fields:
        return provider_fields
    merged = dict(provider_fields or {})
    merged[_TEMPLATE_FIELDS_PROVIDER] = {
        **merged.get(_TEMPLATE_FIELDS_PROVIDER, {}),
        **fields,
    }
    return merged


def _move_written_tool_calls(
    mot: ModelOutputThunk, text: str
) -> tuple[str, list[dict[str, Any]] | None]:
    """Move the tool calls written in a reply's text into OpenAI-shaped `tool_calls`.

    The chat template writes an assistant's `tool_calls` into its block as
    `<tool_call>...</tool_call>`, and only then wires the turn into the next `exec`
    as an input. A reply whose calls are still in its text, as the HuggingFace
    backend leaves them, has them moved, but only when the backend parsed every
    one, so no call is dropped.

    Returns:
        The text without its calls and the calls, or `text` unchanged and `None`.
    """
    written = _WRITTEN_TOOL_CALL.findall(text)
    if not mot.tool_calls or len(written) != len(mot.tool_calls):
        return text, None

    from ..helpers.openai_compatible_helpers import build_tool_calls

    tool_calls = cast(list[dict[str, Any]], build_tool_calls(mot))
    return _WRITTEN_TOOL_CALL.sub("", text).strip(), tool_calls
