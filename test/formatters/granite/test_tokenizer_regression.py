# SPDX-License-Identifier: Apache-2.0

"""Regression tests for granite-4.0-micro tokenizer stability.

Transformers v5 removed ``tokenization_gpt2_fast.py`` and merged GPT2Tokenizer
into a single class that inherits from ``TokenizersBackend``.  The new
GPT2Tokenizer constructs BPE from ``vocab.json`` + ``merges.txt``
(``VOCAB_FILES_NAMES`` in ``tokenization_gpt2.py``), whereas the old
GPT2TokenizerFast loaded the pre-built ``tokenizer.json``
(``VOCAB_FILES_NAMES`` in ``tokenization_gpt2_fast.py``).  The two file sources
produce different token IDs for granite-4.0-micro, causing regressions in 3/6
RAG intrinsics.  Because the v5 GPT2Tokenizer still uses the Rust backend,
``is_fast`` reports ``True`` — so the attribute is unreliable and golden token
ID assertions are the real guard.

Fix: use ``PreTrainedTokenizerFast.from_pretrained()`` directly, which loads
``tokenizer.json``.

References (line numbers approximate):
  tf4 GPT2TokenizerFast — tokenization_gpt2_fast.py:
    VOCAB_FILES_NAMES = {..., "tokenizer_file": "tokenizer.json"}
  tf5 GPT2Tokenizer — tokenization_gpt2.py:
    VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}
"""

# Third Party
import pytest

transformers = pytest.importorskip(
    "transformers", reason="transformers not installed -- install mellea[hf]"
)

pytestmark = pytest.mark.integration

MODEL_ID = "ibm-granite/granite-4.0-micro"

GOLDEN_TOKEN_IDS = {
    "2023": [2366, 18],
    "d.o.o": [67, 14778, 14778],
    "60-138-3818": [1399, 12, 10350, 12, 19162, 23],
    "Hello world": [9906, 1917],
    "relevant": [98673],
    "irrelevant": [404, 98673],
    "partially relevant": [4581, 34575, 9959],
    "answerable": [9399, 481],
    "unanswerable": [359, 9399, 481],
    "CLEAR": [91449],
    "faithful": [75710, 1285],
    "unfaithful": [359, 75710, 1285],
}


@pytest.fixture(scope="module")
def tokenizer():
    return transformers.AutoTokenizer.from_pretrained(MODEL_ID)


class TestTokenizerType:
    def test_tokenizer_is_fast(self, tokenizer):
        assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast), (
            f"Expected a PreTrainedTokenizerFast subclass but got "
            f"{type(tokenizer).__name__}. The v5 GPT2Tokenizer constructs BPE "
            f"from vocab.json/merges.txt instead of loading tokenizer.json, "
            f"producing different token IDs that break RAG intrinsics."
        )


class TestTokenizationStability:
    @pytest.mark.parametrize(
        "text, expected_ids",
        list(GOLDEN_TOKEN_IDS.items()),
        ids=list(GOLDEN_TOKEN_IDS.keys()),
    )
    def test_encode_golden_ids(self, tokenizer, text, expected_ids):
        actual = tokenizer.encode(text, add_special_tokens=False)
        assert actual == expected_ids

    @pytest.mark.parametrize(
        "text, expected_ids",
        list(GOLDEN_TOKEN_IDS.items()),
        ids=list(GOLDEN_TOKEN_IDS.keys()),
    )
    def test_roundtrip_decode(self, tokenizer, text, expected_ids):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded)
        assert decoded == text

    def test_special_tokens(self, tokenizer):
        assert tokenizer.eos_token_id == 100257
        assert tokenizer.bos_token_id == 100257
