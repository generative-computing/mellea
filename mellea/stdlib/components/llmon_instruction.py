# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

r"""LLMON-aware `Instruction` for the soft-instruct round-trip.

`LLMONInstruction` is an opt-in `Instruction` subclass that llmonizes itself on
the way out and parses a llmonized reply back into a `Component` on the way in,
closing the round-trip described in
`LLMON-FORMATTER-AND-COMPONENT-FOR-SOFT-INSTRUCT-ROUND-TRIP.md` (issue #1623):

- `as_llmon_document()` maps the instruction's parts onto id-tagged LLMON blocks
  (`instr:<id>`, `data:<id>`, `req:<id>`) plus an `exec:<id>` object whose
  children reference those blocks by id (`exec.instruction`, `exec.input`,
  `exec.requirement`, `exec.output`). `LLMONFormatter` calls this to render the
  request.
- `_parse()` runs `parse_llmon` on the reply and returns an `LLMONResult`
  component holding the parsed node tree; `str(result)` yields the text of the
  reply's `exec.output` / `assistant` block.

This is soft mode: the model is prompted to emit LLMON but decoding is not
constrained to the grammar. Nothing here is backend-specific — serialization is
formatter-owned and parsing runs through the generic `Component._parse` dispatch.

TODO(#1623 follow-up): render with the Machine-LLMON special-token table
(`<|open|>`, `<|:|>`, `<|.|>`, `<|open_end|>`, `<|close|>`) instead of the
backslash form. The raw-string to special-token go-between may become an imported
library. For now `to_llmon` produces the Phase 1 backslash dialect.

TODO(#1623 follow-up): `user_variables` are Jinja-rendered into the description by
`Instruction.__init__` before `as_llmon_document` runs, so a `data:<id>` block
currently carries the rendered value rather than the raw `{{var}}` placeholder.
Round-tripping unrendered variables is deferred.
"""

from __future__ import annotations

import secrets

from ...core import (
    Component,
    LLMONDocument,
    LLMONNode,
    LLMONParseError,
    LLMONString,
    LLMONUserType,
    ModelOutputThunk,
    Span,
    TemplateRepresentation,
    parse_llmon,
    to_llmon,
)
from .instruction import Instruction

# The user-type tags the Instruction mapping introduces beyond the LLMON paper.
_INSTR_TAG = "instr"
_DATA_TAG = "data"
_REQ_TAG = "req"
_EXEC_TAG = "exec"
_ASSISTANT_TAG = "assistant"
# `exec` object slots, referencing the id-tagged blocks by id.
_EXEC_INSTRUCTION = "exec.instruction"
_EXEC_INPUT = "exec.input"
_EXEC_REQUIREMENT = "exec.requirement"
_EXEC_OUTPUT = "exec.output"


def _new_id() -> str:
    """Return a fresh 4-digit block id, stable for the render that captures it."""
    return f"{secrets.randbelow(9000) + 1000}"


def _block(kind: str, block_id: str, text: str) -> LLMONUserType:
    r"""Build an id-tagged content block, e.g. `\instr:3456\ "..." /instr:3456/`."""
    return LLMONUserType(f"{kind}:{block_id}", [LLMONString(text)])


def _ref(slot: str, target_id: str) -> LLMONUserType:
    r"""Build an `exec` reference to a block id, e.g. `\exec.instruction\ "instr:3456" /exec.instruction/`."""
    return LLMONUserType(slot, [LLMONString(target_id)])


def instruction_to_llmon_document(instruction: Instruction) -> LLMONDocument:
    """Map any `Instruction` onto an LLMON document of id-tagged blocks + `exec`.

    This is the reusable OUT-direction builder: `LLMONFormatter` calls it for every
    `Instruction` it renders (so a plain `m.instruct` is llmonized by default when
    the session uses an `LLMONFormatter`), and `LLMONInstruction.as_llmon_document`
    delegates to it.

    The document holds one `instr:<id>` block (the description), one `req:<id>`
    block per non-`check_only` requirement, and a final `exec:<id>` object whose
    children reference those blocks by id. The `exec.output` slot points at a
    placeholder `assistant:<id>` id — where the model generates.

    Args:
        instruction: The instruction (plain or `LLMONInstruction`) to map.

    Returns:
        The `LLMONDocument` for the instruction. Block ids are freshly generated
        per call and are reused by the `exec` references within the same document.
    """
    items: list[LLMONNode] = []
    exec_children: list[LLMONNode] = []

    description = (
        str(instruction._description) if instruction._description is not None else ""
    )
    instr_id = _new_id()
    items.append(_block(_INSTR_TAG, instr_id, description))
    exec_children.append(_ref(_EXEC_INSTRUCTION, f"{_INSTR_TAG}:{instr_id}"))

    for req in instruction._requirements:
        if req.description in (None, "") or req.check_only:
            continue
        req_id = _new_id()
        items.append(_block(_REQ_TAG, req_id, str(req.description)))
        exec_children.append(_ref(_EXEC_REQUIREMENT, f"{_REQ_TAG}:{req_id}"))

    output_id = _new_id()
    exec_children.append(_ref(_EXEC_OUTPUT, f"{_ASSISTANT_TAG}:{output_id}"))

    exec_id = _new_id()
    items.append(LLMONUserType(f"{_EXEC_TAG}:{exec_id}", exec_children))
    return LLMONDocument(items)


class LLMONResult(Component[str]):
    """The parsed result of a llmonized reply.

    Holds the full parsed LLMON node tree and the text extracted from the reply's
    `exec.output` / `assistant` block. `str(result)` and `format_for_llm()` yield
    that output text, so the result behaves like an assistant message downstream.

    Args:
        document: The parsed LLMON document for the whole reply.
        output_text: The text of the reply's output block. When the reply carries
            no recognizable output block, this is the reply's full text.
    """

    def __init__(self, document: LLMONDocument, output_text: str) -> None:
        """Store the parsed document and extracted output text."""
        self._document = document
        self._output_text = output_text

    @property
    def document(self) -> LLMONDocument:
        """The parsed LLMON node tree for the reply."""
        return self._document

    def parts(self) -> list[Span]:
        """An `LLMONResult` has no sub-`Span` parts.

        Returns:
            An empty list.
        """
        return []

    def format_for_llm(self) -> TemplateRepresentation:
        """Render the result as an assistant message carrying the output text.

        Returns:
            A `TemplateRepresentation` with `role="assistant"` whose content is
            the extracted output text.
        """
        return TemplateRepresentation(
            obj=self,
            args={"content": self._output_text},
            template_order=["*"],
            role="assistant",
        )

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Return the extracted output text.

        Returns:
            The reply's output-block text.
        """
        return self._output_text

    def __str__(self) -> str:
        """Return the extracted output text."""
        return self._output_text


class LLMONInstruction(Instruction):
    """An `Instruction` that llmonizes its request and parses a llmonized reply.

    Use this in place of `Instruction` (with an `LLMONFormatter` on the backend)
    for the soft-instruct LLMON round-trip. `format=` (Pydantic constrained
    decoding) is incompatible with LLMON's own grammar and is rejected here.

    Args:
        format: Must be `None`. LLMON has its own grammar, so a Pydantic `format`
            alongside it is contradictory.

    See `Instruction` for the remaining constructor arguments.
    """

    def __init__(self, *args, format: object = None, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """Initialize like `Instruction`, rejecting a non-`None` `format`.

        Raises:
            ValueError: If `format` is not `None`; LLMON mode has its own grammar
                and cannot also constrain to a Pydantic schema.
        """
        if format is not None:
            raise ValueError(
                "format= is not supported in LLMON mode: LLMON has its own grammar, "
                "so constraining to a Pydantic schema is contradictory."
            )
        super().__init__(*args, **kwargs)

    def as_llmon_document(self) -> LLMONDocument:
        """Map this instruction onto an LLMON document of id-tagged blocks + `exec`.

        Delegates to `instruction_to_llmon_document`, the shared OUT-direction
        builder (also used by `LLMONFormatter` for plain instructions).

        Returns:
            The `LLMONDocument` for this instruction. Block ids are freshly
            generated per call and are reused by the `exec` references within the
            same document.
        """
        return instruction_to_llmon_document(self)

    def _parse(self, computed: ModelOutputThunk) -> LLMONResult:  # type: ignore[override]
        """Parse a llmonized reply into an `LLMONResult` component.

        Runs `parse_llmon` on the reply text and extracts the output-block text
        (the reply's `assistant:*` block, else its first user-type block, else the
        full reply text). A reply that is not valid LLMON is treated as plain text.

        Returns:
            An `LLMONResult` holding the parsed document and extracted output text.
        """
        text = computed.value if computed.value is not None else ""
        try:
            document = parse_llmon(text)
        except LLMONParseError:
            # Soft mode: the model was prompted, not constrained, so a non-LLMON
            # reply is expected to occur. Fall back to treating the reply as text.
            return LLMONResult(LLMONDocument([]), text)

        return LLMONResult(document, _extract_output_text(document, fallback=text))


def _extract_output_text(document: LLMONDocument, *, fallback: str) -> str:
    """Extract the reply's output text from a parsed LLMON document.

    Prefers an `assistant:*` block, then any top-level user-type block, then the
    fallback text.

    Args:
        document: The parsed reply document.
        fallback: Text to return when no user-type block is found.

    Returns:
        The extracted output text.
    """
    user_types = [n for n in document.items if isinstance(n, LLMONUserType)]

    def _text_of(node: LLMONUserType) -> str:
        return "".join(
            child.value for child in node.children if isinstance(child, LLMONString)
        )

    for node in user_types:
        if node.tag == _ASSISTANT_TAG or node.tag.startswith(f"{_ASSISTANT_TAG}:"):
            return _text_of(node)
    if user_types:
        return _text_of(user_types[0])
    return fallback
