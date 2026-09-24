# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

r"""`LLMONFormatter` for rendering components as LLMON on the way to the model.

`LLMONFormatter` renders every `Instruction` as LLMON text (so a plain
`m.instruct` is llmonized by default when the session uses this formatter), also
renders any other component that exposes an `as_llmon_document()` method, and
delegates every other span to the standard Jinja2 template rendering. It extends
`TemplateFormatter` — which supplies the concrete `print` that `ChatFormatter`
leaves abstract and is itself a `ChatFormatter` — so chat backends drive it
through the inherited `to_chat_messages` with no backend changes.
`to_chat_messages` calls `self.print(c)` for message content, so overriding
`print` alone llmonizes the request everywhere the formatter is used.

Construct a backend with `formatter=LLMONFormatter(model_id=...)` to llmonize
every action on the way out. This alone gives LLMON *out*; to also parse the reply
back into a Component, use `LLMONInstruction` as the action (e.g.
`m.instruct(..., instruction_cls=LLMONInstruction)`). This is the soft
(prompt-only) LLMON path; decoding is not constrained to the grammar. See
`LLMON-FORMATTER-AND-COMPONENT-FOR-SOFT-INSTRUCT-ROUND-TRIP.md` (issue #1623).

TODO(#1623 follow-up): the component's `as_llmon_document()` currently renders via
the Phase 1 backslash dialect (`to_llmon`). Swapping to the Machine-LLMON
special-token table is deferred.
"""

from typing import Protocol, runtime_checkable

from ..core import LLMONDocument, Span, to_llmon
from ..stdlib.components.instruction import Instruction
from ..stdlib.components.llmon_instruction import instruction_to_llmon_document
from .template_formatter import TemplateFormatter


@runtime_checkable
class _LLMONizable(Protocol):
    """A component that can render itself as an LLMON document."""

    def as_llmon_document(self) -> LLMONDocument:
        """Return the component's LLMON document."""
        ...


class LLMONFormatter(TemplateFormatter):
    """Formatter that renders instructions as LLMON, others via templates.

    Extends `TemplateFormatter` (a `ChatFormatter`), so it plugs into chat
    backends through the inherited `to_chat_messages` with no backend edits. Every
    `Instruction` is llmonized on the way out (so a plain `m.instruct` needs no
    special component), as is any component exposing `as_llmon_document()`.
    """

    def print(self, c: Span) -> str:
        """Render `c` as LLMON when it is an instruction, else via the base formatter.

        Args:
            c (Span): The component, content block, or model output to render.

        Returns:
            str: The LLMON text of the instruction's mapped document when `c` is an
                `Instruction` (or exposes `as_llmon_document()`), otherwise the
                standard template rendering from `TemplateFormatter`.
        """
        if isinstance(c, _LLMONizable):
            return to_llmon(c.as_llmon_document())
        if isinstance(c, Instruction):
            return to_llmon(instruction_to_llmon_document(c))
        return super().print(c)
