# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the automatic, model-driven LLMON round-trip.

For a served model in the LLMON allowlist (`llmon_models.is_llmon_model`), Mellea
encodes an `Instruction` into Machine LLMON in the shared backend chokepoint and
strips any Machine LLMON out of the reply in `ModelOutputThunk._set_parsed_repr` —
no `LLMONInstruction` or per-call flag needed. `LLMONInstruction` remains only to
opt into hard-mode output grammar via `constrain_output`.

These tests cover the reply/encode helpers, the allowlist, the encode helper
`encode_instruction_for_llmon`, and the `LLMONInstruction` grammar/`format=`
contract. The two core seams and the served endpoint are exercised by the e2e
example and the backend/base unit tests.
"""

import pytest

# LLMON support is an optional extra (`pip install "mellea[llmon]"`); skip the
# whole module when the `llmon` library (import root `lang`) is absent.
pytest.importorskip("lang.json.llmon_converters")

from mellea.stdlib.components.instruction import Instruction
from mellea.stdlib.components.llmon import (
    encode_instruction_for_llmon,
    encode_llmon_input,
    is_llmon,
    llmon_grammar,
    parse_llmon_reply,
)
from mellea.stdlib.components.llmon_instruction import LLMONInstruction
from mellea.stdlib.components.llmon_models import is_llmon_model

# A raw Machine-LLMON assistant block, as a server that returns un-detokenized
# tokens would send it: `<|open|>assistant<|:|>ID<|close|> TEXT <|open_end|>...`.
_ASSISTANT_BLOCK = (
    "<|open|>assistant<|:|>2921<|close|>the answer"
    "<|open_end|>assistant<|:|>2921<|close|>"
)


class TestIsLLMONModel:
    """The allowlist is dependency-free and gates both seams by resolved model id."""

    def test_known_model_matches(self):
        assert is_llmon_model("llmon95:latest") is True

    def test_none_is_not_a_match(self):
        assert is_llmon_model(None) is False

    def test_arbitrary_model_is_not_a_match(self):
        assert is_llmon_model("granite4.2:3b") is False


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

    def test_requirements_encoded_as_req_member(self):
        encoded = encode_llmon_input(
            "Summarize", requirements=["one sentence", "mention both aspects"]
        )
        assert is_llmon(encoded)
        # Requirements appear in a dedicated `req` list member.
        assert "req<|close|>" in encoded
        assert "one sentence" in encoded
        assert "mention both aspects" in encoded

    def test_empty_requirements_omit_req_member(self):
        # None and [] both yield the two-argument encoding, byte-identical to the
        # no-requirements form — so adding the parameter never changes old callers.
        baseline = encode_llmon_input("Summarize")
        assert encode_llmon_input("Summarize", requirements=None) == baseline
        assert encode_llmon_input("Summarize", requirements=[]) == baseline
        assert "req<|close|>" not in baseline

    def test_dict_data_encodes_keyed_object(self):
        # A dict of named documents becomes a nested keyed `data` object, with
        # every document present and labelled by its key.
        encoded = encode_llmon_input(
            "Summarize", data={"doc1": "alpha text", "doc2": "beta text"}
        )
        assert is_llmon(encoded)
        assert "data<|.|>doc1" in encoded
        assert "data<|.|>doc2" in encoded
        assert "alpha text" in encoded
        assert "beta text" in encoded

    def test_single_string_data_stays_scalar(self):
        # A plain string stays a scalar `data` member — no nested object — so the
        # existing single-document shape is preserved.
        encoded = encode_llmon_input("Summarize", data="only text")
        assert "data<|.|>" not in encoded
        assert "only text" in encoded


class TestEncodeInstructionForLLMON:
    """`encode_instruction_for_llmon` maps an `Instruction`'s fields onto the encoder.

    This is what the backend chokepoint calls to LLMONize an instruction for an
    LLMON model; it replaces the old `LLMONInstruction(llmonize_input=True)` path.
    """

    def test_instruction_only(self):
        content = encode_instruction_for_llmon(Instruction("Summarize this."))
        assert is_llmon(content)
        assert parse_llmon_reply(content) == "Summarize this."

    def test_requirements_encoded(self):
        # Requirements must reach the wire: the model is told the same constraints
        # the sampling strategy validates against.
        content = encode_instruction_for_llmon(
            Instruction("Summarize this.", requirements=["must be one sentence"])
        )
        assert "req<|close|>" in content
        assert "must be one sentence" in content

    def test_check_only_requirements_excluded(self):
        # Mirrors the base Instruction prompt filter: a check_only requirement is
        # validation-only and must not be shown to the model.
        from mellea.core.requirement import Requirement

        content = encode_instruction_for_llmon(
            Instruction(
                "Do it.",
                requirements=[
                    Requirement("internal grading note", check_only=True),
                    "show this to the model",
                ],
            )
        )
        assert "internal grading note" not in content
        assert "show this to the model" in content

    def test_single_grounding_doc_is_scalar(self):
        content = encode_instruction_for_llmon(
            Instruction(
                "Summarize the document.", grounding_context={"doc1": "alpha text"}
            )
        )
        assert "alpha text" in content
        # One document is a scalar data member, not a nested keyed object.
        assert "data<|.|>" not in content

    def test_multiple_grounding_docs_are_keyed(self):
        # All grounding documents must reach the wire, keyed by name.
        content = encode_instruction_for_llmon(
            Instruction(
                "Summarize the documents.",
                grounding_context={
                    "doc1": "alpha text",
                    "doc2": "beta text",
                    "doc3": "gamma text",
                },
            )
        )
        assert "data<|.|>doc1" in content
        assert "data<|.|>doc2" in content
        assert "data<|.|>doc3" in content
        assert "alpha text" in content
        assert "beta text" in content
        assert "gamma text" in content

    def test_repair_string_folded_into_instruction(self):
        # Rejection-sampling repair feedback must reach an LLMON model, not be
        # silently dropped. `copy_and_repair` sets `_repair_string`.
        repaired = Instruction("Summarize this.").copy_and_repair(
            "Your last answer was too long; use one sentence."
        )
        content = encode_instruction_for_llmon(repaired)
        assert "too long" in content

    def test_renders_through_formatter(self):
        # The pre-encoded Machine-LLMON string must render verbatim via the shared
        # Message template emitted by the backend's encoding view.
        from mellea.core.backend import _llmon_encoded
        from mellea.formatters.template_formatter import TemplateFormatter

        wrapped = _llmon_encoded(Instruction("Summarize this."))
        rendered = TemplateFormatter(model_id="default").print(wrapped)
        assert is_llmon(rendered)
        assert parse_llmon_reply(rendered) == "Summarize this."


class TestLLMONEncodedView:
    """The backend's `_llmon_encoded` view renders LLMON without mutating the action."""

    def test_does_not_mutate_original(self):
        from mellea.core.backend import _llmon_encoded

        original = Instruction("Summarize this.")
        wrapped = _llmon_encoded(original)
        assert wrapped is not original
        # The original still renders through the normal Instruction prose template.
        assert "Instruction" in (original.format_for_llm().template_order or [])

    def test_view_renders_machine_llmon(self):
        from mellea.core.backend import _llmon_encoded

        tr = _llmon_encoded(Instruction("Summarize this.")).format_for_llm()
        assert is_llmon(tr.args["content"])
        assert tr.role == "user"

    def test_view_preserves_output_grammar(self):
        # Risk E: wrapping an LLMONInstruction(constrain_output=True) must not drop
        # the hard-mode grammar hook. The view subclasses the origin class, so
        # output_grammar is inherited.
        from mellea.core.backend import _llmon_encoded

        wrapped = _llmon_encoded(LLMONInstruction("Task.", constrain_output=True))
        spec = wrapped.output_grammar()
        assert spec is not None
        assert spec == (llmon_grammar(), "gbnf")


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
