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
# whole module when the `llmon` library (import root `lang`) is absent.
pytest.importorskip("lang.json.llmon_converters")

from mellea.stdlib.components.llmon import (
    encode_llmon_input,
    is_llmon,
    llmon_grammar,
    parse_llmon_reply,
)
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


class TestLenientFallback:
    """When the strict library conversion fails, a block scan recovers the answer.

    A served model whose chat template consumed the opening blocks can return a
    *partial* Machine-LLMON fragment that the library converter cannot parse
    whole (it begins mid-stream, carries unescaped control characters, or echoes
    instructions as preamble). These shapes are drawn from real `llmon95:latest`
    replies.
    """

    def test_preamble_then_data_assistant_block(self):
        # Reply opens with echoed instructions (no leading <|open|>), then the
        # real answer sits in a data<|:|>assistant block. The library converter
        # raises on this; the scan recovers the block text.
        reply = (
            ". customer feedback blah"
            "<|open_end|>instr<|:|>user<|:|>2175<|close|>"
            "<|open|>exec<|:|>3230<|close|>"
            "<|open|>data<|:|>assistant<|:|>0689<|close|>"
            "The customer loved the product but found shipping slow."
            "<|open_end|>data<|:|>assistant<|:|>0689<|close|>"
        )
        assert (
            parse_llmon_reply(reply)
            == "The customer loved the product but found shipping slow."
        )

    def test_last_block_wins_over_earlier_echoed_block(self):
        # Earlier assistant blocks are echoed context; the final one is the answer.
        reply = (
            "<|open|>data<|:|>assistant<|:|>1<|close|>stale echo<|open_end|>x"
            "<|open|>assistant<|:|>2<|close|>the real answer<|open_end|>y"
        )
        assert parse_llmon_reply(reply) == "the real answer"

    def test_control_characters_in_block_do_not_lose_the_answer(self):
        # Unescaped newlines inside a string break json.loads in the strict path;
        # the scan returns the block text (newlines preserved) rather than raw
        # markup.
        reply = (
            "<|open|>data<|:|>assistant<|:|>7<|close|>line one\nline two"
            "<|open_end|>data<|:|>assistant<|:|>7<|close|>"
        )
        assert parse_llmon_reply(reply) == "line one\nline two"

    def test_fragment_with_no_complete_block_falls_back_to_raw(self):
        # A fragment with the open token but no complete assistant block must not
        # be mangled by a blanket token strip; it degrades to raw text.
        fragment = " Explain step by step.<|open_end|>exec<|.|>output<|"
        assert parse_llmon_reply(fragment) == fragment


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

    def test_result_renders_through_template_formatter(self):
        # Regression: an LLMONResult must render through the real formatter without
        # raising "could not find template candidate". The requirement-validation
        # path re-renders a judged output's parsed_repr, so a missing template
        # there crashed every requirements=[...] round-trip. The result falls back
        # to the shared Message template, which emits the content verbatim.
        from mellea.formatters.template_formatter import TemplateFormatter

        result = LLMONInstruction("Task.")._parse(_FakeThunk("the clean answer"))
        rendered = TemplateFormatter(model_id="default").print(result)
        assert rendered == "the clean answer"


class TestFormatRejection:
    """LLMON mode rejects a Pydantic `format` (LLMON has its own grammar)."""

    def test_format_raises(self):
        with pytest.raises(ValueError, match="format="):
            LLMONInstruction("Task.", format=int)

    def test_format_none_ok(self):
        # Should construct without error.
        LLMONInstruction("Task.", format=None)


class TestEncodeLLMONInput:
    """`encode_llmon_input` builds Machine-LLMON input the model was trained to read."""

    def test_instruction_only_round_trips(self):
        encoded = encode_llmon_input("Summarize in one sentence")
        assert is_llmon(encoded)
        # The encoding is the out-direction inverse of parse: the library's
        # reply converter reads it back to the same instr member.
        assert parse_llmon_reply(encoded) == "Summarize in one sentence"

    def test_instruction_and_data_both_present(self):
        encoded = encode_llmon_input("Summarize", data="loved it, slow shipping")
        assert is_llmon(encoded)
        # Both members are present in the encoded object.
        assert "Summarize" in encoded
        assert "loved it, slow shipping" in encoded


class TestLLMONizeInput:
    """`llmonize_input=True` sends a Machine-LLMON object instead of plain prose."""

    def test_default_is_plain_prose(self):
        # Without the flag, format_for_llm defers to Instruction's prose rendering
        # (template_order includes "Instruction"), not a llmonized object.
        instr = LLMONInstruction("Summarize this.")
        tr = instr.format_for_llm()
        assert "Instruction" in (tr.template_order or [])
        assert "content" not in tr.args  # prose path uses description/requirements

    def test_llmonize_input_encodes_instruction(self):
        instr = LLMONInstruction("Summarize this.", llmonize_input=True)
        tr = instr.format_for_llm()
        content = tr.args["content"]
        assert is_llmon(content)
        assert parse_llmon_reply(content) == "Summarize this."
        assert tr.role == "user"

    def test_llmonize_input_renders_through_formatter(self):
        # The pre-encoded Machine-LLMON string must render verbatim via the shared
        # Message template, with no LLMONInstruction.jinja2.
        from mellea.formatters.template_formatter import TemplateFormatter

        instr = LLMONInstruction("Summarize this.", llmonize_input=True)
        rendered = TemplateFormatter(model_id="default").print(instr)
        assert is_llmon(rendered)
        assert parse_llmon_reply(rendered) == "Summarize this."


class TestLLMONGrammar:
    """`llmon_grammar` is a valid GBNF grammar accepting real encoded input."""

    def test_grammar_is_gbnf_and_memoized(self):
        g = llmon_grammar()
        assert g.startswith("root ::=")  # GBNF uses `root ::=`, not Lark's `start:`
        # Memoized: the same object is returned on repeat calls.
        assert llmon_grammar() is g

    def test_grammar_validates_in_llguidance(self):
        llguidance = pytest.importorskip("llguidance")
        err = llguidance.LLMatcher.validate_grammar(
            llguidance.grammar_from("gbnf", llmon_grammar())
        )
        assert err == ""  # empty means valid

    def test_grammar_accepts_real_encoded_input(self):
        # The grammar must accept what encode_llmon_input emits (out-direction) and,
        # by the same object shape, what a constrained model would produce.
        llguidance = pytest.importorskip("llguidance")
        pytest.importorskip("llguidance.hf")
        transformers = pytest.importorskip("transformers")

        tok = transformers.AutoTokenizer.from_pretrained("gpt2")
        llt = llguidance.hf.from_tokenizer(tok, n_vocab=tok.vocab_size)
        for text in (
            encode_llmon_input("Summarize this."),
            encode_llmon_input("Summarize this.", data="loved it, slow shipping"),
        ):
            matcher = llguidance.LLMatcher(
                llt, llguidance.grammar_from("gbnf", llmon_grammar())
            )
            assert matcher.consume_tokens(tok.encode(text))
            assert matcher.is_accepting()


class TestConstrainOutput:
    """`constrain_output` toggles the output-grammar hook (soft vs. hard mode)."""

    def test_default_is_soft_no_grammar(self):
        instr = LLMONInstruction("Task.")
        assert instr._constrain_output is False
        assert instr.output_grammar() is None

    def test_hard_mode_returns_gbnf_grammar(self):
        instr = LLMONInstruction("Task.", constrain_output=True)
        spec = instr.output_grammar()
        assert spec is not None
        grammar_text, syntax = spec
        assert syntax == "gbnf"
        assert grammar_text == llmon_grammar()

    def test_orthogonal_to_llmonize_input(self):
        # The two flags are independent and may be combined.
        instr = LLMONInstruction("Task.", llmonize_input=True, constrain_output=True)
        assert instr._llmonize_input is True
        assert instr._constrain_output is True
        assert instr.output_grammar() is not None
