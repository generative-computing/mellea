# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

r"""LLMON-aware `Instruction` for opting into hard-mode grammar.

For a served model trained on LLMON (arXiv:2603.22519, LLM Object Notation), the
round-trip is now **automatic and model-driven** — no subclass needed. When the
backend's model is in the LLMON allowlist (`llmon_models.is_llmon_model`), Mellea
encodes the instruction into Machine LLMON on the way out (in the shared backend
chokepoint `Backend.generate_from_context`) and strips any Machine LLMON out of the
reply on the way back (in `ModelOutputThunk._set_parsed_repr`). So a plain
`m.instruct(prompt)` against an LLMON model round-trips to clean text on its own.

`LLMONInstruction` remains only for one opt-in behavior that is *not* model-driven:
**hard-mode output constraint**. Passing `constrain_output=True` makes
`output_grammar` return the LLMON reply-object grammar (`llmon_grammar`), which a
grammar-aware backend uses to constrain decoding to valid Machine LLMON. Enforcement
needs a backend with a grammar hook — the HuggingFace llguidance path, or a vLLM
server via `guided_grammar`; on Ollama or official OpenAI it is a silent no-op (soft
fallback). The automatic input-encode still applies to an `LLMONInstruction` (it is
an `Instruction`), so grammar and encode compose.

The constructor rejects `format=`, since LLMON has its own grammar and cannot also be
constrained to a Pydantic schema. (The same conflict is rejected for the automatic
path by `Backend.generate_from_context`.)

Use it via `m.instruct(..., instruction_cls=LLMONInstruction,
instruction_kwargs={"constrain_output": True})`.
"""

from __future__ import annotations

from .instruction import Instruction
from .llmon import llmon_grammar


class LLMONInstruction(Instruction):
    """An `Instruction` that can opt into LLMON hard-mode output constraint.

    The LLMON reply-parse and input-encode are automatic for an allowlisted model
    (see the module docstring), so this subclass exists only to opt into hard-mode
    grammar via `constrain_output`.

    Args:
        format: Must be `None`. LLMON has its own grammar, so a Pydantic `format`
            alongside it is contradictory.
        constrain_output: If `True`, `output_grammar` returns the LLMON grammar so a
            grammar-aware backend constrains decoding to valid Machine LLMON (hard
            mode). If `False` (the default), generation is unconstrained (soft mode).

    See `Instruction` for the remaining constructor arguments.
    """

    def __init__(  # type: ignore[no-untyped-def]
        self, *args, format: object = None, constrain_output: bool = False, **kwargs
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
        self._constrain_output = constrain_output

    def output_grammar(self) -> tuple[str, str] | None:
        """Return the output grammar for constrained decoding, or `None`.

        A grammar-aware backend (the HuggingFace llguidance path; vLLM via
        `guided_grammar`) calls this on the action it is generating. Returning a
        grammar there masks generation to the LLMON reply-object shape; returning
        `None` leaves decoding unconstrained. The hook is duck-typed — any component
        exposing `output_grammar` participates, with no backend changes. The
        automatic input-encode preserves this method (the encoding view subclasses
        this class), so grammar survives the encode.

        Returns:
            `(grammar_text, syntax)` with `syntax == "gbnf"` when `constrain_output`
            is `True`, else `None`.
        """
        if not self._constrain_output:
            return None
        return (llmon_grammar(), "gbnf")
