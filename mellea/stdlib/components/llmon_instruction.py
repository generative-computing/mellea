# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

r"""LLMON-aware `Instruction` for the served round-trip.

`LLMONInstruction` is an opt-in `Instruction` subclass for talking to an LLMON
model (arXiv:2603.22519) that is **served behind a chat-completions endpoint**
with its `chat_template.jinja` installed server-side.

On that path the server owns serialization: it llmonizes the prompt (drawing the
block ids, emitting the `instr`/`data`/`req` blocks and wiring the `exec` graph)
and detokenizes the reply. Mellea therefore sends a plain `{role, content}`
message through the ordinary chat backend and needs **no formatter swap and no
new backend** — a plain `m.instruct` already round-trips.

Note (unverified — see `LLMON-STATUS.md`, "Open design question"): the LLMON
model owners describe the outgoing message dict as carrying structured
`instruction`/`data` fields (each with an `exec` flag), not a flat `content`
string, with the chat template reading those named fields. If confirmed, the
"plain `{role, content}`, no out-direction work" claim above is incomplete and an
out-direction field mapping is needed. The wire shape is not yet confirmed, so it
is flagged here rather than assumed either way. This subclass's behavior is
unaffected regardless — it only touches the reply-in direction.

What this subclass adds is on the way in: `_parse` runs `parse_llmon_reply`
(backed by the `llmon` library), which detects raw Machine LLMON in the reply
and, if present, extracts the assistant block's text; a reply the server already
detokenized passes through unchanged. The constructor also rejects `format=`,
since LLMON has its own grammar and cannot also be constrained to a Pydantic
schema.

Pass it via `m.instruct(..., instruction_cls=LLMONInstruction)`.
"""

from __future__ import annotations

from ...core import Component, ModelOutputThunk, Span, TemplateRepresentation
from .instruction import Instruction
from .llmon import parse_llmon_reply


class LLMONResult(Component[str]):
    """The parsed result of an LLMON reply.

    Holds the assistant text extracted from the reply. `str(result)` and
    `format_for_llm()` yield that text, so the result behaves like an assistant
    message downstream.

    Args:
        output_text: The assistant text extracted from the reply (already
            stripped of any Machine-LLMON markup).
    """

    def __init__(self, output_text: str) -> None:
        """Store the extracted output text."""
        self._output_text = output_text

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
            The reply's assistant text.
        """
        return self._output_text

    def __str__(self) -> str:
        """Return the extracted output text."""
        return self._output_text


class LLMONInstruction(Instruction):
    """An `Instruction` that parses an LLMON reply back into a `Component`.

    Use this against an LLMON model served with its `chat_template.jinja`
    installed server-side (see the module docstring). `format=` (Pydantic
    constrained decoding) is incompatible with LLMON's own grammar and is
    rejected here.

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

    def _parse(self, computed: ModelOutputThunk) -> LLMONResult:  # type: ignore[override]
        """Parse an LLMON reply into an `LLMONResult` component.

        Runs `parse_llmon_reply` on the reply text: a reply carrying raw Machine
        LLMON has its assistant block extracted; a reply the server already
        detokenized passes through unchanged.

        Returns:
            An `LLMONResult` holding the extracted assistant text.
        """
        text = computed.value if computed.value is not None else ""
        return LLMONResult(parse_llmon_reply(text))
