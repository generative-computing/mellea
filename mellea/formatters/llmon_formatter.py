# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`LLMONFormatter`: renders components as LLMON for models trained on it.

LLMON (LLM Object Notation, arXiv:2603.22519) marks a prompt up with special-token
blocks, so an LLMON-trained model can tell what it should do (`instr`) from what it
should only read (`data`, `doc`) and what its answer must satisfy (`req`).
`LLMONFormatter` builds that markup from the same `format_for_llm()` args that
`TemplateFormatter` renders through Jinja2: each component is lowered to a nested
dict, its sub-components are lowered the same way, and the tree is converted once
by the `llmon` library's `LLMONConverters`, which flattens the nesting into dotted
tag names. Each user turn ends with an `exec` block that wires its blocks to the
`assistant` output. System, assistant, and tool turns are sent as their plain text,
since the model's chat template wraps those in their LLMON blocks itself, and
`parse_output` extracts the assistant's reply from a generation that carries LLMON
markup. All content is sanitized by the library first, so LLMON special tokens
inside it cannot open or close blocks.

Backends use this formatter by default for the served models registered here (see
`default_formatter`); pass one to a backend explicitly to use it with any other
model. It needs the `llmon` extra: `pip install "mellea[llmon]"`.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any, cast

import pydantic

from ..backends.model_ids import ModelIdentifier
from ..core import CBlock, Component, ModelOutputThunk, Span, TemplateRepresentation
from ..stdlib.components.chat import Message, message_from_template_representation
from .chat_formatter import ChatFormatter
from .template_formatter import TemplateFormatter


@dataclasses.dataclass(frozen=True)
class LLMONDialect:
    """The `LLMONConverters` options for the LLMON variant a model was trained on.

    Each field sets the converter option of the same name. A field left as `None`
    keeps the converter's own default; those defaults are the LLMON format Mellea
    recommends, so set a field only for a model trained on a different variant.

    Args:
        no_type_tags_for_literals (bool | None): Leave `int`, `float`, `bool` and
            `None` values without their `int`/`float`/`bool`/`none` type tags.
        no_composite_tags (bool | None): Leave out the `object` and `list` blocks,
            which flattens list items into `tag<|:|>index` blocks.
        no_openend_tagname (bool | None): Close every block with a bare
            `<|open_end|>` instead of repeating its tag name.
        generate_nested_tagnames (bool | None): Also emit a block for each nesting
            level that holds only other blocks.
    """

    no_type_tags_for_literals: bool | None = None
    no_composite_tags: bool | None = None
    no_openend_tagname: bool | None = None
    generate_nested_tagnames: bool | None = None


# Served models that speak LLMON, with the dialect each was trained on, matched
# exactly against a model id. `llmon95:latest` uses the converter's defaults.
_LLMON_MODELS: dict[str, LLMONDialect] = {"llmon95:latest": LLMONDialect()}

# Instructions and messages are the turns LLMON models were trained on: their args
# become a turn's top-level blocks, under the tag names the models expect (from the
# granite LLMON preview models' chat template). Args not listed keep their own name.
_TURN_ARG_TAGS: dict[str, dict[str, str]] = {
    "instruction": {
        "description": "instr",
        "requirements": "req",
        "grounding_context": "data",
    },
    "message": {"documents": "doc"},
}
# The `exec` slot that each of those args is wired into. Args not listed are inputs.
_EXEC_SLOTS: dict[str, dict[str, str]] = {
    "instruction": {
        "description": "instruction",
        "prefix": "instruction",
        "repair": "instruction",
        "output_prefix": "instruction",
        "requirements": "requirement",
    },
    "message": {"content": "instruction"},
}
# The order of the slots in an `exec` block, as in the granite LLMON preview models'
# chat template. The `output` slot always comes last.
_EXEC_SLOT_ORDER = ("instruction", "system", "input", "requirement")
# A message's content is tagged by its role.
_ROLE_TAGS: dict[str, str] = {
    "user": "instr",
    "assistant": "assistant",
    "system": "system",
    "tool": "data",
}
# Other components whose tag is renamed to the one the models expect.
_COMPONENT_TAGS: dict[str, str] = {"document": "doc"}
# Whitespace and special tokens (e.g. `<|end_of_text|>`) only.
_ONLY_SPECIAL_TOKENS = re.compile(r"(?:\s|<\|[^<>|]*\|>)*")
# A tool call as the model writes it in its reply, with the whitespace before it.
_WRITTEN_TOOL_CALL = re.compile(r"\s*<tool_call>.*?</tool_call>", re.DOTALL)


def llmon_dialect_for(model_id: str | ModelIdentifier | None) -> LLMONDialect | None:
    """Return the LLMON dialect registered for a model, if it is an LLMON model.

    Args:
        model_id: A model id, or a `ModelIdentifier`, which matches when any of its
            provider names is registered.

    Returns:
        The registered `LLMONDialect`, or `None` when the model is not an LLMON
        model.
    """
    if isinstance(model_id, ModelIdentifier):
        names = [getattr(model_id, f.name) for f in dataclasses.fields(model_id)]
    else:
        names = [model_id]
    for name in names:
        if name in _LLMON_MODELS:
            return _LLMON_MODELS[name]
    return None


def default_formatter(model_id: str | ModelIdentifier) -> ChatFormatter:
    """Return the formatter a backend uses when it is not given one.

    Args:
        model_id: The model id the backend was constructed with.

    Returns:
        An `LLMONFormatter` for a registered LLMON model, otherwise a
        `TemplateFormatter`.
    """
    dialect = llmon_dialect_for(model_id)
    if dialect is None:
        return TemplateFormatter(model_id=model_id)
    return LLMONFormatter(model_id, dialect=dialect)


class _LLMONMessage(Message):
    """A chat message whose content `LLMONFormatter` has already rendered.

    Backends print every message from `to_chat_messages` again to serialize it;
    marking these lets `LLMONFormatter.print` return the content as it is instead of
    encoding it a second time.
    """


class LLMONFormatter(ChatFormatter):
    """Formatter that renders components as LLMON for LLMON-trained models.

    A user turn is lowered from its component's `format_for_llm()` args: every
    non-empty arg is output, sub-components inside the args are lowered the same way,
    and a model output inside the args is lowered from its parsed representation. An
    instruction or a user message uses the tag names the models were trained on
    (`instr`, `req` and `data` for an instruction's description, requirements and
    grounding context; `instr` for a message's content; `doc` for documents), and its
    other args are tagged with their arg names. Any other component nests its args
    under its own tag, named after its template (e.g. `query`). Every user turn ends
    with an `exec` block that references its blocks by tag.

    System, assistant, and tool turns are sent as their text alone (a message's
    content, or a model output's reply), with their tool calls and tool-call id as
    message fields, because the model's chat template wraps each in its own LLMON
    block and writes an assistant's `tool_calls` out itself. When a reply holds tool
    calls the model wrote out (as the HuggingFace backend leaves them) and the
    backend parsed every one, they are moved from the text into `tool_calls`, so the
    template also wires that assistant turn into the next `exec` as an input.

    Content is sanitized with the `llmon` library's `LLMONConverters.sanitize`, which
    replaces each LLMON special token with a look-alike that is not one, so content
    such as a retrieved document cannot open or close blocks. That covers every
    string value and key in a user turn and the text of system, assistant, and tool
    turns.

    Args:
        model_id (str | ModelIdentifier): The model the prompts are for.
        dialect (LLMONDialect | None): The converter options for the LLMON variant
            the model was trained on. Defaults to the dialect registered for
            `model_id`, else the converter defaults.

    Attributes:
        dialect (LLMONDialect): The dialect in use, resolved from `model_id` when
            none was given. Options it leaves unset take the converter's defaults.

    Example:
        ```python
        from mellea.formatters import LLMONFormatter
        from mellea.stdlib.components import Instruction

        formatter = LLMONFormatter("llmon95:latest")
        text = formatter.print(
            Instruction("Summarize the review.", grounding_context={"review": "..."})
        )
        ```
    """

    def __init__(
        self, model_id: str | ModelIdentifier, *, dialect: LLMONDialect | None = None
    ):
        """Initialize the formatter for a model and load the `llmon` converters.

        Raises:
            ImportError: If the `llmon` library is not installed.
        """
        try:
            from lang.common import common
            from lang.json.llmon_converters import LLMONConverters
        except ImportError as e:
            raise ImportError(
                "LLMONFormatter requires the `llmon` library. "
                'Please install it with: pip install "mellea[llmon]"'
            ) from e

        self.model_id = model_id
        self.dialect: LLMONDialect = (
            dialect if dialect is not None else llmon_dialect_for(model_id)
        ) or LLMONDialect()
        self._converters = LLMONConverters
        self._sanitize: Callable[[str], str] = LLMONConverters().sanitize
        # The converter options in effect: those the dialect sets, else the
        # converter's own defaults.
        defaults = inspect.signature(LLMONConverters).parameters
        self._options: dict[str, bool] = {}
        for field in dataclasses.fields(self.dialect):
            value = getattr(self.dialect, field.name)
            self._options[field.name] = (
                defaults[field.name].default if value is None else value
            )
        self._markup = (
            common.OPEN_TAG,
            common.CLOSE_TAG,
            common.OPEN_END_TAG,
            common.SELF_CLOSE_TAG,
        )
        self._open_tag = common.OPEN_TAG
        self._open_end_tag = common.OPEN_END_TAG
        # The library keeps the raw separator in JSON keys, e.g. "assistant<|:|>12".
        self._assistant_key_prefix = "assistant" + common.COLON_SEPARATOR_TAG
        open_tag, close, open_end, colon = (
            re.escape(tag)
            for tag in (
                common.OPEN_TAG,
                common.CLOSE_TAG,
                common.OPEN_END_TAG,
                common.COLON_SEPARATOR_TAG,
            )
        )
        # A complete assistant block, `<|open|>[data<|:|>]assistant[<|:|>ID]<|close|>
        # TEXT <|open_end|>`, with an id of any characters; the group is the text.
        self._assistant_block = re.compile(
            f"{open_tag}(?:data{colon})?assistant(?:{colon}[^<]*)?{close}(.*?){open_end}",
            re.DOTALL,
        )

    def print(self, c: Span) -> str:
        """Render a component, content block, or model output as one turn's content.

        A user turn is LLMON ending in an `exec` block; a system, assistant, or tool
        turn is its text alone. A message that `to_chat_messages` already rendered is
        returned as it is. Any other value, such as a tool's output that `call_tools`
        prints for its tool message, is not a turn and is returned as plain text. All
        content is sanitized.

        Args:
            c (Span): The component, content block, or model output to render.

        Returns:
            str: The content of the turn.
        """
        if isinstance(c, _LLMONMessage):
            return c.content
        if not isinstance(c, Component | CBlock | ModelOutputThunk):
            lowered = self._lower(c)
            if lowered is None:
                return ""
            return lowered if isinstance(lowered, str) else str(lowered)
        subject, rep, role = self._resolve(c)
        if role == "user":
            return self._render_user_turn(subject, rep)
        return self._turn_text(c, subject)

    def to_chat_messages(self, cs: list[Span]) -> list[Message]:
        """Convert a linearized chat history into messages with LLMON content.

        Roles, attachments, and tool metadata are resolved as in
        `ChatFormatter.to_chat_messages`, except that a `Message` is rendered too
        rather than passed through. `print` returns the content of the resulting
        messages unchanged.

        Args:
            cs (list[Span]): The linearized sequence of context components, content
                blocks, and model outputs to convert.

        Returns:
            list[Message]: One message per item in `cs`, in order.

        Raises:
            ValueError: If a component declares a `role` (via its
                `TemplateRepresentation`) outside `Message.Role`.
        """
        messages: list[Message] = []
        for c in cs:
            subject, rep, role = self._resolve(c)
            tool_calls = None
            if role == "user":
                content = self._render_user_turn(subject, rep)
            else:
                content = self._turn_text(c, subject)
                if isinstance(c, ModelOutputThunk) and not (rep and rep.tool_calls):
                    content, tool_calls = _move_written_tool_calls(c, content)
            if rep is None:
                messages.append(
                    _LLMONMessage(
                        role=cast(Message.Role, role),
                        content=content,
                        tool_calls=tool_calls,
                    )
                )
            else:
                if tool_calls is not None:
                    rep = dataclasses.replace(rep, tool_calls=tool_calls)
                messages.append(
                    message_from_template_representation(
                        rep,
                        default_role=cast(Message.Role, role),
                        content=content,
                        message_cls=_LLMONMessage,
                    )
                )
        return messages

    def parse_output(self, text: str) -> str:
        """Extract the assistant's reply from a generation that may carry LLMON markup.

        Text with no LLMON markup is returned unchanged. Otherwise the reply is,
        trying each in turn: the `assistant` block (or the only block) of a
        well-formed generation, as converted by the `llmon` library; the last
        complete `assistant` block of a partial generation, since a served model can
        start mid-structure or echo context before answering; and, in a dialect with
        bare closers, the text before the closer that ends a generation which is the
        rest of an `assistant` block the prompt opened (as the model's chat template
        opens it), allowing only special tokens such as end-of-text after the closer.
        When none applies, `text` is returned unchanged rather than stripped into
        fragments. Tool calls the model wrote in its reply are part of the reply.

        Args:
            text (str): The generated text, exactly as the model returned it.

        Returns:
            str: The assistant's reply.
        """
        if not any(tag in text for tag in self._markup):
            return text

        try:
            converter = self._converters(
                converting_to_json=True,
                no_openend_tagname=self._options["no_openend_tagname"],
            )
            reply = self._assistant_value(
                json.loads(converter.llmon_str_to_json_str(text))
            )
        except Exception:
            # The converter is strict, so a partial or garbled generation raises.
            reply = None
        if reply is not None:
            return reply

        blocks = self._assistant_block.findall(text)
        if blocks:
            return blocks[-1]

        if self._options["no_openend_tagname"]:
            head, closer, tail = text.partition(self._open_end_tag)
            if (
                closer
                and head.strip()
                and self._open_tag not in head
                and _ONLY_SPECIAL_TOKENS.fullmatch(tail)
            ):
                return head
        return text

    def _resolve(self, c: Span) -> tuple[Span, TemplateRepresentation | None, str]:
        """Return what to lower for one turn, its representation, and its role.

        Mirrors `ChatFormatter.to_chat_messages`: a model output is an assistant
        turn of its parsed component, if it has one, and a component's
        representation can declare a different role.
        """
        role = "user"
        if isinstance(c, ModelOutputThunk):
            role = "assistant"
            assert c.is_computed(), "a model output must be computed to be rendered"
            if isinstance(c.parsed_repr, Component):
                c = c.parsed_repr
        rep = c.format_for_llm() if isinstance(c, Component) else None
        if not isinstance(rep, TemplateRepresentation):
            return c, None, role
        return c, rep, rep.role if rep.role is not None else role

    def _render_user_turn(self, c: Span, rep: TemplateRepresentation | None) -> str:
        """Render a user turn as LLMON ending in its `exec` block."""
        blocks, slots = self._user_turn_blocks(c, rep)
        if slots:
            blocks["exec"] = _exec_block(slots)
        converter = self._converters(**self._options)
        return converter.json_dict_to_llmon_str(blocks, compact=True)

    def _turn_text(self, c: Span, subject: Span) -> str:
        """Return the sanitized text of a system, assistant, or tool turn.

        The chat template wraps these turns in their own block, so LLMON here would
        nest inside it. A model output is its reply; a message is its content.
        """
        if isinstance(c, ModelOutputThunk):
            return self._sanitize(c.value or "")
        if isinstance(subject, Message):
            return self._sanitize(subject.content)
        # Another component declaring one of these roles has no text of its own,
        # so it is sent as LLMON of its args.
        lowered = self._lower(subject)
        if isinstance(lowered, dict):
            converter = self._converters(**self._options)
            return converter.json_dict_to_llmon_str(lowered, compact=True)
        return "" if lowered is None else str(lowered)

    def _user_turn_blocks(
        self, c: Span, rep: TemplateRepresentation | None
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Lower a user turn to its top-level blocks and the `exec` slot of each."""
        if rep is None:
            lowered = self._lower(c)
        else:
            tag = _component_tag(rep, c)
            if tag in _TURN_ARG_TAGS:
                return self._arg_blocks(tag, rep, "user")
            lowered = self._lower_representation(rep, c)

        if isinstance(lowered, dict) and len(lowered) == 1:
            # A component's own block: the request itself, unless it is a document.
            (tag,) = lowered
            return lowered, {tag: "input" if tag == "doc" else "instruction"}
        return {"instr": lowered}, {"instr": "instruction"}

    def _arg_blocks(
        self, tag: str, rep: TemplateRepresentation, role: str
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Lower an instruction's or a message's args to blocks named as trained."""
        names = _TURN_ARG_TAGS[tag]
        if tag == "message":
            names = {**names, "content": _ROLE_TAGS.get(role, role)}
        blocks: dict[str, Any] = {}
        slots: dict[str, str] = {}
        for key, value in rep.args.items():
            lowered = self._lower(value)
            if _is_empty(lowered):
                continue
            name = names.get(key, self._sanitize(key))
            blocks[name] = _unwrap(lowered, name)
            slots[name] = _EXEC_SLOTS[tag].get(key, "input")
        return blocks, slots

    def _lower_representation(
        self, rep: TemplateRepresentation, component: object
    ) -> Any:
        """Lower a component's representation from its args."""
        tag = _component_tag(rep, component)
        if tag in _TURN_ARG_TAGS:
            blocks, _ = self._arg_blocks(tag, rep, rep.role or "user")
            return blocks
        args: dict[str, Any] = {}
        for key, value in rep.args.items():
            lowered = self._lower(value)
            if not _is_empty(lowered):
                args[self._sanitize(key)] = lowered
        # A component with nothing to show still marks its place, as a self-closing
        # block.
        return {_COMPONENT_TAGS.get(tag, tag): args or ""}

    def _lower(self, value: object) -> Any:
        """Lower a value to the dicts, lists and scalars `LLMONConverters` reads.

        Mirrors `TemplateFormatter._stringify`: strings and content blocks become
        text, components are lowered from their args, model outputs from their
        parsed representation, and mappings and iterables keep their structure,
        without their empty entries. All text, keys included, is sanitized.
        """
        match value:
            case str():
                return self._sanitize(value)
            case None | bool() | int() | float():
                return value
            case ModelOutputThunk():
                parsed = value.parsed_repr
                if parsed is None or isinstance(parsed, ModelOutputThunk):
                    return self._lower(value.value)
                return self._lower(parsed)
            case CBlock():
                return self._lower(value.value)
            case Component():
                rep = value.format_for_llm()
                if isinstance(rep, TemplateRepresentation):
                    return self._lower_representation(rep, value)
                return self._lower(rep)
            case TemplateRepresentation():
                return self._lower_representation(value, value.obj)
            case pydantic.BaseModel():
                return self._lower(value.model_dump())
            case Mapping():
                lowered: dict[str, Any] = {}
                for key, item in value.items():
                    lowered_item = self._lower(item)
                    if not _is_empty(lowered_item):
                        lowered[self._sanitize(str(key))] = lowered_item
                return lowered
            case Iterable():
                return [
                    lowered_item
                    for lowered_item in map(self._lower, value)
                    if not _is_empty(lowered_item)
                ]
            case _:
                return self._sanitize(str(value))

    def _assistant_value(self, parsed: object) -> str | None:
        """Return the assistant block's value from the converter's JSON, if any."""
        if isinstance(parsed, str):
            return parsed
        if not isinstance(parsed, dict):
            return None
        for key, value in parsed.items():
            if key == "assistant" or key.startswith(self._assistant_key_prefix):
                return _as_text(value)
        if len(parsed) == 1:
            # A reply that is a single block of another tag is that block's value.
            (value,) = parsed.values()
            return _as_text(value)
        return None


def _component_tag(rep: TemplateRepresentation, component: object) -> str:
    """Name a component's block after the template it is rendered with, in snake_case.

    The first template named in `template_order` is the one a component shares with
    its subclasses (`Instruction` for any `Instruction`); a component that names
    none is named after its class.
    """
    names = [name for name in rep.template_order or [] if name != "*"]
    name = names[0] if names else type(component).__name__
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def _exec_block(slots: Mapping[str, str]) -> dict[str, Any]:
    """Build a user turn's `exec` block, which references its blocks by tag."""
    block: dict[str, Any] = {}
    for slot in _EXEC_SLOT_ORDER:
        tags = [tag for tag, tag_slot in slots.items() if tag_slot == slot]
        if tags:
            block[slot] = tags[0] if len(tags) == 1 else tags
    block["output"] = "assistant"
    return block


def _unwrap(lowered: Any, tag: str) -> Any:
    """Drop the block of `tag` that each item of a block of that same tag carries.

    A message's documents each lower to a `doc` block, which would otherwise nest
    inside the message's own `doc` block.
    """
    if isinstance(lowered, list):
        return [_unwrap(item, tag) for item in lowered]
    if isinstance(lowered, dict) and lowered.keys() == {tag}:
        return lowered[tag]
    return lowered


def _is_empty(value: object) -> bool:
    """Whether a lowered value has nothing to show: `None`, or an empty string or container."""
    return value is None or (isinstance(value, str | list | dict) and not value)


def _as_text(value: object) -> str:
    """Return a converted block's value as text, as JSON if it is structured."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


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
