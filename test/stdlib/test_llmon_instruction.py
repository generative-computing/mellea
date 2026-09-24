# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `LLMONInstruction` and `LLMONResult`.

Covers the soft-instruct LLMON round-trip: an instruction llmonizes itself via
`as_llmon_document()`, and a llmonized reply parses back into an `LLMONResult`
component through `_parse`.
"""

import pytest

from mellea.core.llmon import (
    LLMONDocument,
    LLMONString,
    LLMONUserType,
    parse_llmon,
    to_llmon,
)
from mellea.stdlib.components.llmon_instruction import LLMONInstruction, LLMONResult


class _FakeThunk:
    """Minimal stand-in for a computed `ModelOutputThunk` (only `.value` used)."""

    def __init__(self, value: str | None) -> None:
        self.value = value


class TestAsLLMONDocument:
    """`as_llmon_document` maps the instruction onto id-tagged blocks + `exec`."""

    def test_blocks_and_exec_present(self):
        instr = LLMONInstruction(
            "Summarise this feedback: {{text}}",
            requirements=["One sentence.", "Cover pros and cons."],
            user_variables={"text": "loved it, shipping slow"},
        )
        doc = instr.as_llmon_document()
        tags = [n.tag for n in doc.items if isinstance(n, LLMONUserType)]
        assert sum(t.startswith("instr:") for t in tags) == 1
        assert sum(t.startswith("req:") for t in tags) == 2
        assert sum(t.startswith("exec:") for t in tags) == 1

    def test_description_has_user_variables_rendered(self):
        # NOTE: user_variables are Jinja-rendered by Instruction.__init__, so the
        # instr block carries the rendered value (raw {{var}} round-trip is a TODO).
        instr = LLMONInstruction("Echo: {{name}}", user_variables={"name": "Ada"})
        doc = instr.as_llmon_document()
        instr_block = next(
            n
            for n in doc.items
            if isinstance(n, LLMONUserType) and n.tag.startswith("instr:")
        )
        assert instr_block.children[0].value == "Echo: Ada"

    def test_check_only_requirements_excluded(self):
        from mellea.core import Requirement

        instr = LLMONInstruction(
            "Task.",
            requirements=[
                Requirement("visible"),
                Requirement("hidden", check_only=True),
            ],
        )
        doc = instr.as_llmon_document()
        req_blocks = [
            n
            for n in doc.items
            if isinstance(n, LLMONUserType) and n.tag.startswith("req:")
        ]
        assert len(req_blocks) == 1
        assert req_blocks[0].children[0].value == "visible"

    def test_exec_references_resolve_to_defined_blocks(self):
        instr = LLMONInstruction("Task.", requirements=["r1"])
        doc = instr.as_llmon_document()
        blocks = [n for n in doc.items if isinstance(n, LLMONUserType)]
        defined = {b.tag for b in blocks if not b.tag.startswith("exec:")}
        exec_block = next(b for b in blocks if b.tag.startswith("exec:"))
        for child in exec_block.children:
            assert isinstance(child, LLMONUserType)
            target = child.children[0].value
            if child.tag == "exec.output":
                assert target.startswith("assistant:")
            else:
                assert target in defined

    def test_document_serializes_and_reparses_stably(self):
        instr = LLMONInstruction("Task.", requirements=["a", "b"])
        doc = instr.as_llmon_document()
        assert parse_llmon(to_llmon(doc)) == doc


class TestParseBack:
    """`_parse` turns a llmonized reply into an `LLMONResult` component."""

    def test_parses_assistant_block(self):
        instr = LLMONInstruction("Task.")
        reply = to_llmon(
            LLMONDocument(
                [LLMONUserType("assistant:2921", [LLMONString("the answer")])]
            )
        )
        result = instr._parse(_FakeThunk(reply))
        assert isinstance(result, LLMONResult)
        assert str(result) == "the answer"

    def test_non_llmon_reply_falls_back_to_text(self):
        instr = LLMONInstruction("Task.")
        result = instr._parse(_FakeThunk("plain text answer"))
        assert isinstance(result, LLMONResult)
        assert str(result) == "plain text answer"

    def test_none_value_is_empty(self):
        instr = LLMONInstruction("Task.")
        result = instr._parse(_FakeThunk(None))
        assert str(result) == ""

    def test_result_format_for_llm_is_assistant_role(self):
        instr = LLMONInstruction("Task.")
        result = instr._parse(_FakeThunk("hi"))
        tr = result.format_for_llm()
        assert tr.role == "assistant"
        assert tr.args["content"] == "hi"


class TestFormatRejection:
    """LLMON mode rejects a Pydantic `format`."""

    def test_format_raises(self):
        with pytest.raises(ValueError, match="format="):
            LLMONInstruction("Task.", format=int)

    def test_format_none_ok(self):
        # Should construct without error.
        LLMONInstruction("Task.", format=None)
