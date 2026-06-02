"""Unit tests for HuggingFace backend pure-logic helpers — no model load required."""

from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch", reason="torch not installed — install mellea[hf]")

from transformers.generation.utils import GenerateDecoderOnlyOutput

from mellea.backends import ModelOption
from mellea.backends.huggingface import LocalHFBackend
from mellea.core import ModelOutputThunk
from mellea.stdlib.components import Message


def _make_backend(eos_token_id: int | list[int] = 0) -> LocalHFBackend:
    mock_tok = MagicMock(eos_token_id=eos_token_id, vocab_size=32000)
    mock_tok._tokenizer = MagicMock()
    mock_tok._tokenizer.get_vocab_size.return_value = 32000
    mock_tok.__len__ = MagicMock(return_value=32000)
    mock_model = MagicMock(vocab_size=32000)
    with (
        patch("mellea.backends.huggingface.llguidance") as mock_llg,
        patch("mellea.backends.huggingface.set_seed"),
    ):
        mock_llg.hf.from_tokenizer.return_value = MagicMock(vocab_size=32000)
        return LocalHFBackend(
            model_id="ibm-granite/granite-3.3-8b-instruct",
            custom_config=(mock_tok, mock_model, torch.device("cpu")),
        )


@pytest.mark.parametrize(
    "value, last_token, eos, n_completion, model_options, expected",
    [
        # EOS token at end of sequence -> stop
        ("hello", 99, 99, 2, {}, ["stop"]),
        # Multi-EOS list (eos_token_id as list)
        ("x", 99, [42, 99], 2, {}, ["stop"]),
        # Output ends with a configured stop string -> stop (the new branch)
        (
            "answer<END>",
            4,
            99,
            2,
            {ModelOption.STOP_SEQUENCES: ["<END>", "###"]},
            ["stop"],
        ),
        # Hit max_new_tokens -> length
        ("abc", 4, 99, 3, {ModelOption.MAX_NEW_TOKENS: 3}, ["length"]),
        # No terminator hit -> finish_reasons stays None
        (
            "ongoing",
            4,
            99,
            2,
            {ModelOption.MAX_NEW_TOKENS: 999, ModelOption.STOP_SEQUENCES: ["<END>"]},
            None,
        ),
    ],
)
@pytest.mark.asyncio
async def test_finish_reasons_derivation(
    value, last_token, eos, n_completion, model_options, expected
):
    """post_processing derives finish_reasons from sequence/EOS/stop_strings/max_new_tokens."""
    backend = _make_backend(eos_token_id=eos)
    input_ids = torch.tensor([[1]])
    sequences = torch.tensor([[*range(n_completion), last_token]])

    mot = ModelOutputThunk(value=value)
    mot._action = Message("user", "noop")
    mot._model_options = model_options
    mot._meta["hf_output"] = GenerateDecoderOnlyOutput(
        sequences=sequences,
        scores=None,
        logits=None,
        attentions=None,
        hidden_states=None,
        past_key_values=None,
    )

    await backend.post_processing(mot, [], None, False, {}, None, input_ids)

    assert mot.generation.finish_reasons == expected
