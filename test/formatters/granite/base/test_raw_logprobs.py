# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the raw-logit readout in `generate_with_transformers`.

One-token decisions read label probabilities from the first generated token.
`generate(output_scores=True)` returns *processed* scores (after repetition
penalties, bad-word masks, sequence bias and sampling warpers), so any such
option would silently move the probabilities. With `raw_logprobs` set the
function reads `output_logits` instead and reports the exact logprob of each
requested token id, so neither generation options nor the top-K cut can
change or hide a label.

These tests use a tiny randomly initialised model (no download) and compare
the reported logprobs against a plain forward pass.
"""

# Third Party
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch", reason="torch not installed — install mellea[hf]")
pytest.importorskip(
    "transformers", reason="transformers not installed — install mellea[hf]"
)
from transformers import LlamaConfig, LlamaForCausalLM

# First Party
from mellea.formatters.granite.base.util import generate_with_transformers

_PROMPT = [10, 11, 12, 13, 14, 10]
_LABEL_IDS = [20, 21]
_VOCAB = 64


class _Tokenizer:
    """Minimal tokenizer double: every id decodes to a stable string."""

    eos_token_id = 9999  # outside the vocabulary: never generated

    def decode(self, tokens, *args, **kwargs):
        if isinstance(tokens, int) or getattr(tokens, "dim", lambda: 1)() == 0:
            return f"<{int(tokens)}>"
        return "".join(f"<{int(t)}>" for t in tokens)

    def batch_decode(self, seqs, *args, **kwargs):
        return [self.decode(s) for s in seqs]


@pytest.fixture(scope="module")
def model() -> LlamaForCausalLM:
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=_VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    return LlamaForCausalLM(config).eval()


@pytest.fixture(scope="module")
def reference(model) -> Any:
    """Log-softmax of the model's own next-token distribution."""
    with torch.no_grad():
        logits = model(torch.tensor([_PROMPT])).logits[0, -1].float()
    return torch.log_softmax(logits, -1)


def _readout(model, *, raw: bool, top_logprobs: int | None = None, **generate_kwargs):
    """Run the production path for one greedy token; return {token_id: logprob}."""
    generate_input = {
        "input_tokens": torch.tensor([_PROMPT]),
        "max_new_tokens": 1,
        "return_dict_in_generate": True,
        "output_scores": True,
        "do_sample": False,
        **generate_kwargs,
    }
    other_input: dict = {}
    if top_logprobs is not None:
        other_input["top_logprobs"] = top_logprobs
    if raw:
        generate_input["output_logits"] = True
        other_input["raw_logprobs"] = True
        other_input["logprob_token_ids"] = _LABEL_IDS
    with torch.no_grad():
        response = generate_with_transformers(
            cast(Any, _Tokenizer()), model, generate_input, other_input
        )
    logprobs = response.choices[0].logprobs
    assert logprobs is not None and logprobs.content is not None
    entries = logprobs.content[0].top_logprobs
    return {e.token: e.logprob for e in entries}


def _tok(i: int) -> str:
    return f"<{i}>"


# Options that change processed scores on a greedy one-token call. The
# positive control (`raw=False`) must show the distortion, otherwise the
# negative result below would prove nothing.
_DISTORTING_OPTIONS = [
    pytest.param({"repetition_penalty": 1.5}, id="repetition_penalty"),
    pytest.param({"bad_words_ids": [[_LABEL_IDS[0]]]}, id="bad_words_ids"),
    pytest.param({"sequence_bias": {(_LABEL_IDS[1],): 5.0}}, id="sequence_bias"),
    pytest.param({"do_sample": True, "temperature": 0.5}, id="sampling_temperature"),
    pytest.param({"do_sample": True, "top_k": 3}, id="sampling_top_k"),
]


class TestRawLogprobs:
    def test_raw_matches_plain_forward_pass(self, model, reference):
        got = _readout(model, raw=True)
        for label_id in _LABEL_IDS:
            assert got[_tok(label_id)] == pytest.approx(
                reference[label_id].item(), abs=1e-5
            )

    @pytest.mark.parametrize("options", _DISTORTING_OPTIONS)
    def test_processed_scores_are_distorted(self, model, reference, options):
        """Positive control: the legacy path is moved by these options."""
        got = _readout(model, raw=False, top_logprobs=_VOCAB, **options)
        shifted = [
            abs(got[_tok(i)] - reference[i].item())
            for i in _LABEL_IDS
            if _tok(i) in got
        ]
        # A label masked to -inf drops out of the finite readout; either a
        # shift or a disappearance counts as distortion.
        assert len(shifted) < len(_LABEL_IDS) or max(shifted) > 1e-3

    @pytest.mark.parametrize("options", _DISTORTING_OPTIONS)
    def test_raw_logits_ignore_generation_options(self, model, reference, options):
        got = _readout(model, raw=True, **options)
        for label_id in _LABEL_IDS:
            assert got[_tok(label_id)] == pytest.approx(
                reference[label_id].item(), abs=1e-5
            )

    def test_label_outside_top_k_is_still_reported(self, model, reference):
        """A requested id is appended even when top-K would have cut it."""
        order = torch.argsort(reference, descending=True).tolist()
        buried = [i for i in _LABEL_IDS if order.index(i) >= 3]
        assert buried, "tiny model needs a label outside its top 3 for this test"
        legacy = _readout(model, raw=False, top_logprobs=3)
        assert all(_tok(i) not in legacy for i in buried)
        got = _readout(model, raw=True, top_logprobs=3)
        for label_id in buried:
            assert got[_tok(label_id)] == pytest.approx(
                reference[label_id].item(), abs=1e-5
            )

    def test_requested_id_in_top_k_is_not_duplicated(self, model, reference):
        top = int(torch.argmax(reference))
        other_input = {
            "top_logprobs": 3,
            "raw_logprobs": True,
            "logprob_token_ids": [top],
        }
        with torch.no_grad():
            response = generate_with_transformers(
                cast(Any, _Tokenizer()),
                model,
                {
                    "input_tokens": torch.tensor([_PROMPT]),
                    "max_new_tokens": 1,
                    "return_dict_in_generate": True,
                    "output_logits": True,
                    "do_sample": False,
                },
                other_input,
            )
        logprobs = response.choices[0].logprobs
        assert logprobs is not None and logprobs.content is not None
        tokens = [e.token for e in logprobs.content[0].top_logprobs]
        assert tokens.count(_tok(top)) == 1

    def test_default_path_is_unchanged(self, model):
        """Without `raw_logprobs` other intrinsics still read processed scores."""
        got = _readout(model, raw=False, top_logprobs=5, repetition_penalty=1.5)
        reference_raw = _readout(model, raw=True)
        label = _tok(_LABEL_IDS[0])
        if label in got:
            assert got[label] != pytest.approx(reference_raw[label], abs=1e-3)
