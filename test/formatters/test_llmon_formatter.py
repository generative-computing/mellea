# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `LLMONFormatter`.

The formatter renders llmonizable components (those exposing
`as_llmon_document()`) as LLMON text and delegates every other span to the
standard template rendering.
"""

from mellea.core.llmon import LLMONDocument, LLMONString, LLMONUserType, parse_llmon
from mellea.formatters import LLMONFormatter, TemplateFormatter
from mellea.stdlib.components.instruction import Instruction
from mellea.stdlib.components.llmon_instruction import LLMONInstruction

_MODEL_ID = "ibm-granite/granite-4.1-3b"


def _formatter() -> LLMONFormatter:
    return LLMONFormatter(model_id=_MODEL_ID)


class TestLLMONizableRendering:
    """A llmonizable component renders to LLMON via `print`."""

    def test_print_renders_summarize_feedback_example(self):
        instr = LLMONInstruction(
            "Summarise this customer feedback in one sentence: {{text}}",
            requirements=[
                "The summary must be a single sentence.",
                "Include both positive and negative aspects if both are present.",
            ],
            user_variables={"text": "Great product, slow shipping."},
        )
        rendered = _formatter().print(instr)

        doc = parse_llmon(rendered)
        assert isinstance(doc, LLMONDocument)

        tags = [n.tag for n in doc.items if isinstance(n, LLMONUserType)]
        assert any(t.startswith("instr:") for t in tags)
        assert sum(t.startswith("req:") for t in tags) == 2
        assert any(t.startswith("exec:") for t in tags)

    def test_exec_references_match_defined_block_ids(self):
        instr = LLMONInstruction("Do the thing.", requirements=["Be terse."])
        doc = parse_llmon(_formatter().print(instr))

        blocks = [n for n in doc.items if isinstance(n, LLMONUserType)]
        defined_ids = {b.tag for b in blocks if not b.tag.startswith("exec:")}
        exec_block = next(b for b in blocks if b.tag.startswith("exec:"))

        referenced = [
            child.children[0].value
            for child in exec_block.children
            if isinstance(child, LLMONUserType)
            and child.children
            and isinstance(child.children[0], LLMONString)
        ]
        # Every instr/req reference must resolve to a defined block; the output
        # reference points at a placeholder assistant id (not a defined block).
        for ref in referenced:
            if ref.startswith("assistant:"):
                continue
            assert ref in defined_ids, f"dangling reference: {ref}"

    def test_render_is_stable_across_repeated_print(self):
        # Ids are generated once per document build, so a single print's
        # references are internally consistent (checked above); a fresh print
        # rebuilds ids but stays structurally valid.
        instr = LLMONInstruction("Hello.")
        first = parse_llmon(_formatter().print(instr))
        second = parse_llmon(_formatter().print(instr))
        first_tags = sorted(t.split(":")[0] for t in _user_tags(first))
        second_tags = sorted(t.split(":")[0] for t in _user_tags(second))
        assert first_tags == second_tags


class TestPlainInstructionLLMONized:
    """A plain `Instruction` is llmonized by default (no special component)."""

    def test_plain_instruction_renders_as_llmon(self):
        plain = Instruction(
            "Summarize this.", requirements=["Keep it to one sentence."]
        )
        rendered = _formatter().print(plain)

        # The formatter llmonizes any Instruction, so the base template rendering
        # is NOT used.
        template_out = TemplateFormatter(model_id=_MODEL_ID).print(plain)
        assert rendered != template_out

        doc = parse_llmon(rendered)
        tags = [n.tag for n in doc.items if isinstance(n, LLMONUserType)]
        assert any(t.startswith("instr:") for t in tags)
        assert any(t.startswith("req:") for t in tags)
        assert any(t.startswith("exec:") for t in tags)


class TestFallThrough:
    """Non-instruction spans use the standard template rendering."""

    def test_cblock_falls_through(self):
        from mellea.core import CBlock

        assert _formatter().print(CBlock("raw text")) == "raw text"


def _user_tags(doc: LLMONDocument) -> list[str]:
    return [n.tag for n in doc.items if isinstance(n, LLMONUserType)]
