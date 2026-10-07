# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real llguidance enforcement of the LLMON grammar on the HuggingFace backend.

Unlike the mock tests in `test_huggingface_unit.py`, these build the backend's
actual llguidance tokenizer from a tiny local model and prove that
`LLMONInstruction(constrain_output=True)` produces a logits processor that masks
generation to the LLMON grammar. The proof is deterministic — it does not depend
on a trained model complying — so it is a true enforcement test, not a smoke test.

Needs the `hf` extra (torch/transformers/llguidance) and the `llmon` extra; both
are skipped cleanly when absent.
"""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed — install mellea[hf]")
llguidance = pytest.importorskip(
    "llguidance", reason="llguidance not installed — install mellea[hf]"
)
pytest.importorskip("transformers")
pytest.importorskip("lang.json.llmon_converters", reason="install mellea[llmon]")

from mellea.backends.huggingface import LocalHFBackend
from mellea.formatters.granite.base.util import _GuidanceLogitsProcessor
from mellea.stdlib.components.llmon import llmon_grammar
from mellea.stdlib.components.llmon_instruction import LLMONInstruction

# A tiny random Llama: loads in seconds, exposes `model.vocab_size`, and its
# tokenizer builds a working llguidance tokenizer.
_TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def hf_backend():
    """A LocalHFBackend on a tiny model (module-scoped; the load is the slow part)."""
    return LocalHFBackend(model_id=_TINY_MODEL)


@pytest.mark.slow
class TestLLMONGrammarEnforcement:
    """The LLMON grammar genuinely masks generation on the llguidance path."""

    def test_processor_masks_all_but_grammar_legal_first_token(self, hf_backend):
        # At the first position the LLMON grammar permits only the start of
        # `<|open|>object...`, so the processor must mask (-inf) the overwhelming
        # majority of the vocabulary. This is a deductive enforcement proof: it
        # holds regardless of the model's weights.
        llt = hf_backend._llguidance_tokenizer
        # The grammar is GBNF; LLMatcher assumes Lark unless compiled for its
        # syntax, so wrap it exactly as the backend helper does.
        grammar = llguidance.grammar_from("gbnf", llmon_grammar())
        processor = _GuidanceLogitsProcessor(grammar, llt)
        processor.new_sampling = True

        vocab = llt.vocab_size
        scores = torch.zeros((1, vocab))
        input_ids = torch.zeros((1, 0), dtype=torch.long)

        masked = processor(input_ids, scores)

        allowed = int((masked[0] > float("-inf")).sum().item())
        # Far fewer than the full vocab remain legal — the mask constrains hard.
        assert 0 < allowed < vocab
        assert allowed < vocab // 100  # a tiny, grammar-dictated set

    def test_broken_grammar_rejects_everything(self, hf_backend):
        # Negative control: a grammar that cannot start masks the whole vocab,
        # proving the mask reflects the grammar rather than always allowing a token.
        llt = hf_backend._llguidance_tokenizer
        # A GBNF grammar requiring a literal no tokenizer can emit as a first token.
        grammar = llguidance.grammar_from(
            "gbnf", 'root ::= "\\uE000UNREACHABLE\\uE001"'
        )
        processor = _GuidanceLogitsProcessor(grammar, llt)
        processor.new_sampling = True

        vocab = llt.vocab_size
        masked = processor(
            torch.zeros((1, 0), dtype=torch.long), torch.zeros((1, vocab))
        )
        allowed = int((masked[0] > float("-inf")).sum().item())
        # Either nothing is allowed, or only an EOS-like token — never the broad
        # set a permissive grammar would leave. Contrast with the LLMON test.
        assert allowed <= 1

    @pytest.mark.asyncio
    async def test_constrained_generation_begins_with_llmon_object(self, hf_backend):
        # End-to-end: a constrained generation is forced into the LLMON object
        # shape from its first token, which an unconstrained one is not.
        from mellea.backends.model_options import ModelOption
        from mellea.stdlib.context import ChatContext

        instr = LLMONInstruction("summarize this", constrain_output=True)
        mot, _ = await hf_backend.generate_from_context(
            instr,
            ChatContext(),
            model_options={ModelOption.MAX_NEW_TOKENS: 20, ModelOption.STREAM: False},
        )
        await mot.avalue()
        assert (mot.value or "").startswith("<|open|>object<|close|>")
