# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the library-backed LLMON reply round-trip.

`LLMONInstruction` is an opt-in `Instruction` for talking to an LLMON model
(arXiv:2603.22519) served behind a chat-completions endpoint: the server owns
serialization, so on the way in Mellea sends a plain message and on the way back
`_parse` runs `parse_llmon_reply` to strip any raw Machine-LLMON markup. These
tests cover the reply helper and the component's `_parse`/`format=` contract; the
served endpoint itself is exercised by the e2e example, not here.
"""

import pytest

# LLMON support is an optional extra (`pip install "mellea[llmon]"`); skip the
# whole module when the `llmon` library (import root `converters`) is absent.
pytest.importorskip("converters.JSON.llmon_converters")

from mellea.stdlib.components.llmon import is_llmon, parse_llmon_reply
from mellea.stdlib.components.llmon_instruction import LLMONInstruction, LLMONResult

# A raw Machine-LLMON assistant block, as a server that returns un-detokenized
# tokens would send it: `<|open|>assistant<|:|>ID<|close|> TEXT <|open_end|>...`.
_ASSISTANT_BLOCK = (
    "<|open|>assistant<|:|>2921<|close|>the answer"
    "<|open_end|>assistant<|:|>2921<|close|>"
)


class _FakeThunk:
    """Minimal stand-in for a computed `ModelOutputThunk` (only `.value` used)."""

    def __init__(self, value: str | None) -> None:
        self.value = value


class TestParseLLMONReply:
    """`parse_llmon_reply` handles both raw-LLMON and already-detokenized replies."""

    def test_is_llmon_detects_open_token(self):
        assert is_llmon(_ASSISTANT_BLOCK)
        assert not is_llmon("plain prose with no markup")

    def test_assistant_block_extracts_text(self):
        assert parse_llmon_reply(_ASSISTANT_BLOCK) == "the answer"

    def test_plain_prose_passes_through(self):
        # Server already detokenized: no open token, returned verbatim.
        assert parse_llmon_reply("just a plain answer") == "just a plain answer"

    def test_empty_string_passes_through(self):
        assert parse_llmon_reply("") == ""

    def test_malformed_llmon_falls_back_to_raw(self):
        # An open token but non-conforming structure must not raise; the raw text
        # comes back so the caller degrades to prose rather than losing the reply.
        malformed = "<|open|>oops no close"
        assert parse_llmon_reply(malformed) == malformed

    def test_single_non_assistant_block_uses_its_value(self):
        block = "<|open|>data<|:|>1<|close|>hi<|open_end|>data<|:|>1<|close|>"
        assert parse_llmon_reply(block) == "hi"


class TestParseBack:
    """`_parse` turns a llmonized reply into an `LLMONResult` component."""

    def test_parses_assistant_block(self):
        instr = LLMONInstruction("Task.")
        result = instr._parse(_FakeThunk(_ASSISTANT_BLOCK))
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
    """LLMON mode rejects a Pydantic `format` (LLMON has its own grammar)."""

    def test_format_raises(self):
        with pytest.raises(ValueError, match="format="):
            LLMONInstruction("Task.", format=int)

    def test_format_none_ok(self):
        # Should construct without error.
        LLMONInstruction("Task.", format=None)
