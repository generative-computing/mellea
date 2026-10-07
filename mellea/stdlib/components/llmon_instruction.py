# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

r"""LLMON-aware `Instruction` for talking to an LLMON model.

`LLMONInstruction` is an opt-in `Instruction` subclass for talking to a model
(arXiv:2603.22519) trained on LLMON (LLM Object Notation) over a chat-completions
endpoint. Pass it via `m.instruct(..., instruction_cls=LLMONInstruction)`.

Where the LLMONizing happens — verified against `llmon95:latest`. An earlier
design assumed the *server's* `chat_template.jinja` owns serialization (drawing
block ids, emitting `instr`/`data`/`req` blocks, wiring the `exec` graph, and
detokenizing the reply). That is **false for the tested deployment**: the
`llmon95:latest` Ollama template is a bare passthrough (`{{ .Prompt }}`), with no
system prompt and no `chat_template` in the GGUF metadata. The LLMON behavior is
baked into the fine-tuned Granite weights — the model emits Machine LLMON on its
own and will even infer `instr`/`exec`/`data` structure from a plain prompt. So
neither Mellea nor this server LLMONizes the input by default. See
`LLMON-SUMMARY.md` for the full verification.

This subclass handles both directions over the ordinary chat backend (no new
backend):

- **Reply-in (default, always on).** `_parse` runs `parse_llmon_reply` (backed by
  the `llmon` library), which strips raw Machine LLMON from the reply and returns
  the plain assistant text as an `LLMONResult`. A reply that is already plain prose
  passes through unchanged.
- **Out-direction (opt-in via `llmonize_input=True`).** When set,
  `format_for_llm` encodes the rendered instruction (and any `grounding_context`)
  into a Machine-LLMON `instr`/`data` object — structured input the model was
  trained to read — instead of the plain prose `Instruction` sends by default.
  Default is `False`: the plain-prompt path round-trips, and whether structured
  input improves behavior is deployment-dependent (see `LLMON-SUMMARY.md`).

Either way the *returned* value is plain text (`LLMONResult`), never LLMON markup.

The constructor also rejects `format=`, since LLMON has its own grammar and cannot
also be constrained to a Pydantic schema.
"""

from __future__ import annotations

from ...core import Component, ModelOutputThunk, Span, TemplateRepresentation
from .instruction import Instruction
from .llmon import encode_llmon_input, parse_llmon_reply


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

        Falls back to the shared `Message` template (`template_order=["*",
        "Message"]`), which emits the `content` verbatim. This keeps the result
        renderable wherever a component is stringified — notably the
        requirement-validation re-render in `Requirement.format_for_llm`, which
        feeds a judged output's `parsed_repr` back through the formatter — so the
        judge sees the detokenized text rather than raw Machine-LLMON. Without the
        `Message` fallback there is no `LLMONResult.jinja2`, and
        `TemplateFormatter._load_template` raises `could not find template
        candidate`.

        Returns:
            A `TemplateRepresentation` with `role="assistant"` whose content is
            the extracted output text.
        """
        return TemplateRepresentation(
            obj=self,
            args={"content": self._output_text},
            template_order=["*", "Message"],
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
    """An `Instruction` for an LLMON model: parses the reply, optionally encodes input.

    Use this against a model trained on LLMON over a chat-completions endpoint (see
    the module docstring). On the way back, `_parse` strips any raw Machine LLMON
    and returns plain text as an `LLMONResult`. On the way out, `llmonize_input`
    controls whether the instruction is encoded as a Machine-LLMON `instr`/`data`
    object or sent as the plain prose `Instruction` renders by default.

    `format=` (Pydantic constrained decoding) is incompatible with LLMON's own
    grammar and is rejected.

    Args:
        format: Must be `None`. LLMON has its own grammar, so a Pydantic `format`
            alongside it is contradictory.
        llmonize_input: If `True`, `format_for_llm` encodes the instruction (and
            any single-document `grounding_context`) as a Machine-LLMON object so
            the model receives explicit instruction/data boundaries. If `False`
            (the default), the instruction is sent as plain prose — which already
            round-trips, since the model infers LLMON structure from the weights.

    See `Instruction` for the remaining constructor arguments.
    """

    def __init__(  # type: ignore[no-untyped-def]
        self, *args, format: object = None, llmonize_input: bool = False, **kwargs
    ) -> None:
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
        self._llmonize_input = llmonize_input

    def format_for_llm(self) -> TemplateRepresentation:
        """Render the instruction, encoding it as a Machine-LLMON object when enabled.

        When `llmonize_input` is `False` (default) this defers to `Instruction`'s
        normal prose rendering. When `True`, the instruction's description — and a
        single grounding document, if exactly one is present, as the `data`
        member — is encoded via `encode_llmon_input` into a Machine-LLMON object.
        The pre-encoded string is emitted verbatim through the shared `Message`
        template (`template_order=["*", "Message"]`), so no `LLMONInstruction`
        template file is needed.

        Returns:
            The parent `TemplateRepresentation` (plain mode), or one whose content
            is the Machine-LLMON encoding of the instruction (llmonized mode).
        """
        if not self._llmonize_input:
            return super().format_for_llm()

        instruction = str(self._description) if self._description is not None else ""
        # Use the sole grounding document as the data member when there is exactly
        # one; with none or several, the mapping is ambiguous, so encode the
        # instruction alone and leave the context to the parent prose path.
        data: str | None = None
        if len(self._grounding_context) == 1:
            (doc,) = self._grounding_context.values()
            data = str(doc)

        return TemplateRepresentation(
            obj=self,
            args={"content": encode_llmon_input(instruction, data)},
            template_order=["*", "Message"],
            role="user",
        )

    def _parse(self, computed: ModelOutputThunk) -> LLMONResult:  # type: ignore[override]
        """Parse an LLMON reply into an `LLMONResult` component.

        Runs `parse_llmon_reply` on the reply text: a reply carrying raw Machine
        LLMON has its assistant block extracted; a reply that is already plain
        prose passes through unchanged.

        Returns:
            An `LLMONResult` holding the extracted assistant text.
        """
        text = computed.value if computed.value is not None else ""
        return LLMONResult(parse_llmon_reply(text))
