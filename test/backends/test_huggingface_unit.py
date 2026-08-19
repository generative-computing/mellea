# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for HuggingFace backend pure-logic helpers — no model load required."""

import asyncio
import gc
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch", reason="torch not installed — install mellea[hf]")
pytest.importorskip(
    "transformers", reason="transformers not installed — install mellea[hf]"
)
pytest.importorskip(
    "llguidance", reason="llguidance not installed — install mellea[hf]"
)

import base64
import struct

from transformers.generation.utils import GenerateDecoderOnlyOutput

from mellea.backends import ModelOption
from mellea.backends.adapters import AdapterMixin, IntrinsicAdapter
from mellea.backends.adapters._core import Identity
from mellea.backends.huggingface import LocalHFBackend
from mellea.core import ModelOutputThunk
from mellea.formatters.granite.base.util import (
    chat_completion_request_to_transformers_inputs,
)
from mellea.stdlib.components import (
    AudioBlock,
    AudioUrlBlock,
    ImageBlock,
    ImageUrlBlock,
    Instruction,
    Intrinsic,
    Message,
)
from mellea.stdlib.context import ChatContext

# Minimal 1x1 PNG for testing
_MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx"
    b"\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00"
    b"\x00\x00\x00IEND\xaeB`\x82"
)
_B64_PNG = base64.b64encode(_MINIMAL_PNG).decode()

# Minimal WAV for testing
_SILENT_SAMPLE = struct.pack("<h", 0)
_WAV_HEADER = (
    b"RIFF"
    + struct.pack("<I", 36 + len(_SILENT_SAMPLE))
    + b"WAVEfmt "
    + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
    + b"data"
    + struct.pack("<I", len(_SILENT_SAMPLE))
)
_MINIMAL_WAV = _WAV_HEADER + _SILENT_SAMPLE
_B64_WAV = base64.b64encode(_MINIMAL_WAV).decode()

# All four multimodal block types — reused by every parametrized guard test below.
_MULTIMODAL_CASES = [
    ([ImageBlock(_B64_PNG)], None),
    ([ImageUrlBlock(value="http://example.com/image.png")], None),
    (None, [AudioBlock(_B64_WAV, format="wav")]),
    (None, [AudioUrlBlock(value="http://example.com/audio.wav", format="wav")]),
]


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
    mot._call.action = Message("user", "noop")
    mot._call.model_options = model_options
    mot.raw.response = GenerateDecoderOnlyOutput(
        sequences=sequences,
        scores=None,
        logits=None,
        attentions=None,
        hidden_states=None,
        past_key_values=None,
    )

    await backend.post_processing(mot, [], None, False, {}, None, input_ids)

    assert mot.generation.finish_reasons == expected


class _FakeRewrittenRequest:
    def __init__(self, temperature=None):
        self.temperature = temperature

    def model_copy(self, update):
        copied = _FakeRewrittenRequest(self.temperature)
        for key, value in update.items():
            setattr(copied, key, value)
        return copied


class _FakeRewriter:
    def __init__(self, *args, **kwargs):
        pass

    def transform(self, request_json, **intrinsic_kwargs):
        return _FakeRewrittenRequest()


class _FakeResultProcessor:
    def __init__(self, *args, **kwargs):
        pass


@pytest.fixture
def stub_backend():
    """Return a stub with the attributes _make_backend_specific_and_remove reads.

    Avoids constructing a real LocalHFBackend (which loads a model from the Hub).
    """
    return SimpleNamespace(
        from_mellea_model_opts_map={
            ModelOption.MAX_NEW_TOKENS: "max_new_tokens",
            ModelOption.STOP_SEQUENCES: "stop_strings",
        }
    )


def _call(stub, opts):
    return LocalHFBackend._make_backend_specific_and_remove(stub, opts)


def _make_intrinsic_adapter_stub():
    adapter = IntrinsicAdapter.__new__(IntrinsicAdapter)
    adapter.name = "answerability"
    adapter.qualified_name = "answerability_alora"
    adapter.config = {}
    # Required for the capability-based lookup introduced in Epic #929 Phase 1.
    # __new__ bypasses __init__; use object.__setattr__ to set frozen-dataclass fields.
    object.__setattr__(
        adapter,
        "identity",
        Identity(
            name="answerability", adapter_type="alora", capability="answerability"
        ),
    )
    return adapter


def _make_intrinsic_backend_stub(stub_backend):
    stub_backend.formatter = SimpleNamespace(
        to_chat_messages=lambda linearized_ctx: [Message("user", "Is the sky blue?")]
    )
    stub_backend._added_adapters = {}
    stub_backend._tokenizer = object()
    stub_backend._model = object()
    stub_backend._llguidance_tokenizer = object()
    stub_backend._model_id = "stub-model"
    stub_backend._provider = "huggingface"
    stub_backend._make_backend_specific_and_remove = lambda opts: (
        LocalHFBackend._make_backend_specific_and_remove(stub_backend, opts)
    )
    stub_backend.post_processing = lambda *args, **kwargs: None
    stub_backend._generate_with_adapter_lock = (
        lambda adapter_name, generate_func, *args: generate_func(*args)
    )
    stub_backend._find_adapter = lambda cap, types=None: AdapterMixin._find_adapter(
        stub_backend, cap, types
    )
    return stub_backend


def test_generate_with_adapter_lock_calls_load_peft_adapter():
    """Regression guard: the internal adapter-lock call site (Epic #929 Phase 2,
    issue #1140) must use the renamed `load_peft_adapter` verb, not the old
    `load_adapter` name.
    """
    backend = _make_backend()
    backend._model.active_adapters.return_value = ["my_adapter"]  # type: ignore[union-attr]

    with patch.object(backend, "load_peft_adapter") as mock_load:
        backend._generate_with_adapter_lock("my_adapter", lambda: "output")

    mock_load.assert_called_once_with("my_adapter")
    # Deliberately no `_model.set_adapter` assertion. Since #1141 that call is
    # reached via `activate_peft_adapter` rather than inlined here, so asserting
    # it would make this test an unannounced guard for the delegation chain --
    # failing on a change to `activate_peft_adapter` without naming it. The chain
    # is covered by `test_generate_with_adapter_lock_uses_activate_deactivate_verbs`
    # and the verb itself by `test_activate_peft_adapter_calls_set_adapter`.


def test_generate_with_adapter_lock_uses_activate_deactivate_verbs():
    """_generate_with_adapter_lock delegates to the new activate/deactivate verbs
    rather than calling `_model.set_adapter` directly (Epic #929 Phase 2, issue #1141).
    """
    backend = _make_backend()
    backend._model.active_adapters.return_value = ["my_adapter"]  # type: ignore[union-attr]

    with (
        patch.object(backend, "load_peft_adapter"),
        patch.object(backend, "activate_peft_adapter") as mock_activate,
        patch.object(backend, "deactivate_peft_adapter") as mock_deactivate,
    ):
        backend._generate_with_adapter_lock("my_adapter", lambda: "output")

    mock_activate.assert_called_once_with("my_adapter")
    mock_deactivate.assert_not_called()

    backend._model.active_adapters.return_value = []  # type: ignore[union-attr]
    with (
        patch.object(backend, "activate_peft_adapter") as mock_activate,
        patch.object(backend, "deactivate_peft_adapter") as mock_deactivate,
    ):
        backend._generate_with_adapter_lock("", lambda: "output")

    mock_activate.assert_not_called()
    mock_deactivate.assert_called_once_with("")


def test_activate_peft_adapter_calls_set_adapter():
    """activate_peft_adapter() is a thin wrapper over `_model.set_adapter`."""
    backend = _make_backend()

    backend.activate_peft_adapter("my_adapter")

    backend._model.set_adapter.assert_called_once_with("my_adapter")  # type: ignore[union-attr]


def test_deactivate_peft_adapter_calls_set_adapter_empty():
    """deactivate_peft_adapter() clears active adapters via `_model.set_adapter([])`."""
    backend = _make_backend()

    backend.deactivate_peft_adapter("my_adapter")

    backend._model.set_adapter.assert_called_once_with([])  # type: ignore[union-attr]


def test_deactivate_peft_adapter_swallows_no_adapter_loaded_error():
    """deactivate_peft_adapter() is a no-op if the model has no adapter loaded yet."""
    backend = _make_backend()
    backend._model.set_adapter.side_effect = ValueError(  # type: ignore[union-attr]
        "No adapter loaded. Please load an adapter first."
    )

    backend.deactivate_peft_adapter("my_adapter")  # must not raise


def test_deactivate_peft_adapter_reraises_other_value_errors():
    """deactivate_peft_adapter() only swallows the specific 'no adapter loaded' error."""
    backend = _make_backend()
    backend._model.set_adapter.side_effect = ValueError("some other failure")  # type: ignore[union-attr]

    with pytest.raises(ValueError, match="some other failure"):
        backend.deactivate_peft_adapter("my_adapter")


def test_adapter_activation_lock_is_the_generation_lock():
    """`_adapter_activation_lock()` reuses `_generation_lock`, not a separate lock.

    `LocalFileBinding.activate()`/`.deactivate()` (driven by `adapter_scope()`)
    hold no lock of their own and rely on this method for the exclusivity
    `_generate_with_adapter_lock` otherwise gets from holding `_generation_lock`
    directly. If this ever returned a different lock, the two callers would no
    longer be mutually exclusive.
    """
    backend = _make_backend()

    assert backend._adapter_activation_lock() is backend._generation_lock


def test_list_adapters_reflects_registration_not_just_loading():
    """list_adapters() must include adapters registered via add_adapter, even
    if they've never been loaded (aligns HF's semantics with OpenAI's).
    """
    backend = _make_backend()
    adapter = _make_intrinsic_adapter_stub()
    adapter.backend = None
    adapter.get_local_hf_path = lambda base_model_name: "/fake/path"

    backend.add_adapter(adapter)

    assert adapter.qualified_name not in backend._loaded_adapters
    assert adapter.qualified_name in backend.list_adapters()


def test_add_non_local_hf_adapter_raises():
    """LocalHFBackend.add_adapter() rejects adapters outside its own reality."""
    backend = _make_backend()
    mock_adapter = MagicMock(spec=[])

    with pytest.raises(TypeError, match="LocalHFAdapter"):
        backend.add_adapter(mock_adapter)


def test_seed_forces_do_sample_true(stub_backend):
    """Issue #40: a seed alone must flip do_sample=True so it isn't ignored."""
    out = _call(stub_backend, {ModelOption.SEED: 42})
    assert out["do_sample"] is True


def test_nonzero_temperature_forces_do_sample_true(stub_backend):
    out = _call(stub_backend, {ModelOption.TEMPERATURE: 0.7})
    assert out["do_sample"] is True
    assert out["temperature"] == 0.7


def test_zero_temperature_does_not_force_do_sample(stub_backend):
    """temperature=0 means greedy; don't override do_sample."""
    out = _call(stub_backend, {ModelOption.TEMPERATURE: 0.0})
    assert "do_sample" not in out


def test_seed_with_zero_temperature_does_not_force_do_sample(stub_backend):
    """temperature=0 wins over seed — do_sample=True with temperature=0 crashes transformers."""
    out = _call(stub_backend, {ModelOption.SEED: 42, ModelOption.TEMPERATURE: 0.0})
    assert "do_sample" not in out


def test_no_seed_no_temperature_leaves_do_sample_unset(stub_backend):
    out = _call(stub_backend, {ModelOption.MAX_NEW_TOKENS: 32})
    assert "do_sample" not in out
    assert out["max_new_tokens"] == 32


def test_user_do_sample_is_not_overridden(stub_backend):
    """If the caller explicitly set do_sample=False, respect it even with a seed."""
    out = _call(stub_backend, {ModelOption.SEED: 42, "do_sample": False})
    assert out["do_sample"] is False


def test_seed_sentinel_is_stripped(stub_backend):
    """SEED is a Mellea sentinel and must not leak into the backend kwargs."""
    out = _call(stub_backend, {ModelOption.SEED: 42})
    assert ModelOption.SEED not in out


async def test_intrinsic_seed_with_zero_temperature_keeps_greedy(stub_backend):
    """The intrinsic path must not let seed override explicit temperature=0."""
    backend = _make_intrinsic_backend_stub(stub_backend)
    adapter = _make_intrinsic_adapter_stub()
    captured = {}

    def fake_transformers_inputs(rewritten, tokenizer, model, ll_tokenizer=None):
        assert rewritten.temperature == 0.0
        generate_input = {"input_tokens": object(), "do_sample": False}
        captured["generate_input"] = generate_input
        return generate_input, {}

    def fake_generate_with_transformers(tokenizer, model, generate_input, other_input):
        return object()

    # Pre-populate the adapter so the capability-based lookup finds it.
    backend._added_adapters = {adapter.qualified_name: adapter}

    with (
        patch(
            "mellea.backends.huggingface.granite_formatters.IntrinsicsRewriter",
            _FakeRewriter,
        ),
        patch(
            "mellea.backends.huggingface.granite_formatters.IntrinsicsResultProcessor",
            _FakeResultProcessor,
        ),
        patch(
            "mellea.formatters.granite.base.util.chat_completion_request_to_transformers_inputs",
            side_effect=fake_transformers_inputs,
        ),
        patch(
            "mellea.formatters.granite.base.util.generate_with_transformers",
            side_effect=fake_generate_with_transformers,
        ),
    ):
        output = await LocalHFBackend._generate_from_intrinsic(
            backend,
            Intrinsic("answerability"),
            ChatContext().add(Message("user", "Is the sky blue?")),
            model_options={ModelOption.SEED: 42, ModelOption.TEMPERATURE: 0.0},
        )
        assert output._gen.generate is not None
        await output._gen.generate

    assert captured["generate_input"]["do_sample"] is False
    assert "temperature" not in captured["generate_input"]


@pytest.mark.asyncio
async def test_logits_populated_when_option_set():
    """generation.logits is populated with (vocab_size,) tensors when ModelOption.LOGITS=True (caching disabled)."""
    backend = _make_backend()
    input_ids = torch.tensor([[1]])
    sequences = torch.tensor([[0, 0]])
    # scores shape: (1, vocab_size) per token — post_processing squeezes to (vocab_size,)
    fake_scores = (torch.zeros(1, 32000), torch.zeros(1, 32000))

    mot = ModelOutputThunk(value="hi")
    mot._call.action = Message("user", "noop")
    mot._call.model_options = {ModelOption.LOGITS: True}
    mot.raw.response = GenerateDecoderOnlyOutput(
        sequences=sequences,
        scores=fake_scores,
        logits=None,
        attentions=None,
        hidden_states=None,
        past_key_values=None,
    )

    await backend.post_processing(mot, [], None, False, {}, None, input_ids)

    assert mot.generation.logits is not None
    assert len(mot.generation.logits) == len(fake_scores)
    assert all(t.shape == (32000,) for t in mot.generation.logits)


@pytest.mark.asyncio
async def test_raw_logits_populated_when_option_set():
    """generation.raw_logits is populated with (vocab_size,) tensors when ModelOption.RAW_LOGITS=True (caching disabled)."""
    backend = _make_backend()
    input_ids = torch.tensor([[1]])
    sequences = torch.tensor([[0, 0]])
    vocab_size = 32000
    fake_raw_logits = (torch.ones(1, vocab_size), torch.ones(1, vocab_size))

    mot = ModelOutputThunk(value="hi")
    mot._call.action = Message("user", "noop")
    mot._call.model_options = {ModelOption.RAW_LOGITS: True}
    mot.raw.response = GenerateDecoderOnlyOutput(
        sequences=sequences,
        scores=None,
        logits=fake_raw_logits,
        attentions=None,
        hidden_states=None,
        past_key_values=None,
    )

    await backend.post_processing(mot, [], None, False, {}, None, input_ids)

    assert mot.generation.raw_logits is not None
    assert len(mot.generation.raw_logits) == len(fake_raw_logits)
    assert all(t.shape == (vocab_size,) for t in mot.generation.raw_logits)
    assert mot.generation.logits is None


@pytest.mark.asyncio
async def test_raw_logits_and_logits_both_populated_when_both_options_set():
    """generation.logits and raw_logits are both populated when both options are set."""
    backend = _make_backend()
    input_ids = torch.tensor([[1]])
    sequences = torch.tensor([[0, 0]])
    vocab_size = 32000
    fake_scores = (torch.zeros(1, vocab_size), torch.zeros(1, vocab_size))
    fake_raw_logits = (torch.ones(1, vocab_size), torch.ones(1, vocab_size))

    mot = ModelOutputThunk(value="hi")
    mot._call.action = Message("user", "noop")
    mot._call.model_options = {ModelOption.LOGITS: True, ModelOption.RAW_LOGITS: True}
    mot.raw.response = GenerateDecoderOnlyOutput(
        sequences=sequences,
        scores=fake_scores,
        logits=fake_raw_logits,
        attentions=None,
        hidden_states=None,
        past_key_values=None,
    )

    await backend.post_processing(mot, [], None, False, {}, None, input_ids)

    assert mot.generation.logits is not None
    assert all(t.shape == (vocab_size,) for t in mot.generation.logits)
    assert mot.generation.raw_logits is not None
    assert all(t.shape == (vocab_size,) for t in mot.generation.raw_logits)


@pytest.mark.asyncio
async def test_logits_populated_when_option_set_caching_enabled():
    """generation.logits is populated via the caching branch (_use_caches=True) when ModelOption.LOGITS=True."""
    backend = _make_backend()
    backend._use_caches = True
    input_ids = torch.tensor([[1]])
    sequences = torch.tensor([[0, 0]])
    fake_scores = (torch.zeros(1, 32000), torch.zeros(1, 32000))

    mot = ModelOutputThunk(value="hi")
    mot._call.action = Message("user", "noop")
    mot._call.model_options = {ModelOption.LOGITS: True}
    mot.raw.response = GenerateDecoderOnlyOutput(
        sequences=sequences,
        scores=fake_scores,
        logits=None,
        attentions=None,
        hidden_states=None,
        past_key_values=None,
    )

    with patch.object(backend, "cache_put"):
        await backend.post_processing(mot, [], None, False, {}, None, input_ids)

    assert mot.generation.logits is not None
    assert len(mot.generation.logits) == len(fake_scores)
    assert all(t.shape == (32000,) for t in mot.generation.logits)


@pytest.mark.asyncio
async def test_logits_not_populated_when_option_not_set():
    """generation.logits stays None when ModelOption.LOGITS is not set."""
    backend = _make_backend()
    input_ids = torch.tensor([[1]])
    sequences = torch.tensor([[0, 0]])
    fake_scores = (torch.zeros(1, 32000), torch.zeros(1, 32000))

    mot = ModelOutputThunk(value="hi")
    mot._call.action = Message("user", "noop")
    mot._call.model_options = {}
    mot.raw.response = GenerateDecoderOnlyOutput(
        sequences=sequences,
        scores=fake_scores,
        logits=None,
        attentions=None,
        hidden_states=None,
        past_key_values=None,
    )

    await backend.post_processing(mot, [], None, False, {}, None, input_ids)

    assert mot.generation.logits is None


@pytest.mark.asyncio
async def test_generate_from_raw_logits_sliced_per_item():
    """generate_from_raw slices outputs.scores per batch item and clones each tensor."""
    backend = _make_backend()

    batch_size = 2
    vocab_size = 32000
    n_tokens = 3
    prompt_len = 1

    # Fake tokenizer encoding: (batch_size, prompt_len) input ids
    fake_input_ids = torch.zeros(batch_size, prompt_len, dtype=torch.long)
    fake_encoding = MagicMock()
    fake_encoding.__getitem__ = lambda self, k: (
        fake_input_ids
        if k == "input_ids"
        else torch.ones(batch_size, prompt_len, dtype=torch.long)
    )
    fake_encoding.to = MagicMock(return_value=fake_encoding)
    backend._tokenizer = MagicMock(eos_token_id=0, vocab_size=vocab_size)
    backend._tokenizer.__len__ = MagicMock(return_value=vocab_size)
    backend._tokenizer.return_value = fake_encoding
    backend._tokenizer.batch_decode = MagicMock(return_value=["result_a", "result_b"])

    # Fake outputs: sequences and scores
    sequences = torch.zeros(batch_size, prompt_len + n_tokens, dtype=torch.long)
    fake_scores = tuple(torch.randn(batch_size, vocab_size) for _ in range(n_tokens))
    fake_outputs = GenerateDecoderOnlyOutput(
        sequences=sequences,
        scores=fake_scores,
        logits=None,
        attentions=None,
        hidden_states=None,
        past_key_values=None,
    )

    actions = [Message("user", "hello"), Message("user", "world")]

    with (
        patch(
            "mellea.backends.huggingface.asyncio.to_thread", return_value=fake_outputs
        ),
        patch.object(backend, "do_generate_walks"),
        patch.object(backend, "formatter") as mock_fmt,
    ):
        mock_fmt.print = MagicMock(return_value="prompt")
        results = await backend.generate_from_raw(
            actions, MagicMock(), model_options={ModelOption.LOGITS: True}
        )

    assert len(results) == batch_size
    for item_idx, result in enumerate(results):
        assert result.generation.logits is not None, (
            f"item {item_idx}: logits should be populated"
        )
        assert len(result.generation.logits) == n_tokens, (
            f"item {item_idx}: one tensor per token"
        )
        for tok_idx, t in enumerate(result.generation.logits):
            assert t.shape == (vocab_size,), (
                f"item {item_idx} token {tok_idx}: expected (vocab_size,)"
            )
            # clone: must not share storage with the original batch tensor
            assert t.data_ptr() != fake_scores[tok_idx][item_idx].data_ptr(), (
                f"item {item_idx} token {tok_idx}: logits must be a clone, not a view"
            )


@pytest.mark.asyncio
async def test_generate_from_raw_logits_not_set_when_option_absent():
    """generate_from_raw leaves logits=None when ModelOption.LOGITS is not set."""
    backend = _make_backend()
    batch_size = 1
    vocab_size = 32000
    n_tokens = 2
    prompt_len = 1

    fake_input_ids = torch.zeros(batch_size, prompt_len, dtype=torch.long)
    fake_encoding = MagicMock()
    fake_encoding.__getitem__ = lambda self, k: (
        fake_input_ids
        if k == "input_ids"
        else torch.ones(batch_size, prompt_len, dtype=torch.long)
    )
    fake_encoding.to = MagicMock(return_value=fake_encoding)
    backend._tokenizer = MagicMock(vocab_size=vocab_size)
    backend._tokenizer.__len__ = MagicMock(return_value=vocab_size)
    backend._tokenizer.return_value = fake_encoding
    backend._tokenizer.batch_decode = MagicMock(return_value=["result"])

    sequences = torch.zeros(batch_size, prompt_len + n_tokens, dtype=torch.long)
    fake_scores = tuple(torch.randn(batch_size, vocab_size) for _ in range(n_tokens))
    fake_outputs = GenerateDecoderOnlyOutput(
        sequences=sequences,
        scores=fake_scores,
        logits=None,
        attentions=None,
        hidden_states=None,
        past_key_values=None,
    )

    with (
        patch(
            "mellea.backends.huggingface.asyncio.to_thread", return_value=fake_outputs
        ),
        patch.object(backend, "do_generate_walks"),
        patch.object(backend, "formatter") as mock_fmt,
    ):
        mock_fmt.print = MagicMock(return_value="prompt")
        results = await backend.generate_from_raw(
            [Message("user", "hi")], MagicMock(), model_options={}
        )

    assert results[0].generation.logits is None


@pytest.mark.asyncio
async def test_logits_none_when_stream_and_logits_both_set():
    """generation.logits stays None when STREAM=True, because the streamer yields no scores.

    The streaming path passes text chunks through an AsyncTextIteratorStreamer
    and never accumulates hf_output.scores, so post_processing receives scores=None
    regardless of ModelOption.LOGITS.
    """
    backend = _make_backend()
    input_ids = torch.tensor([[1]])
    sequences = torch.tensor([[0, 0]])

    mot = ModelOutputThunk(value="hi")
    mot._call.action = Message("user", "noop")
    mot._call.model_options = {ModelOption.LOGITS: True, ModelOption.STREAM: True}
    # Streaming output carries no scores — hf_output.scores is None.
    mot.raw.response = GenerateDecoderOnlyOutput(
        sequences=sequences,
        scores=None,
        logits=None,
        attentions=None,
        hidden_states=None,
        past_key_values=None,
    )

    await backend.post_processing(mot, [], None, False, {}, None, input_ids)

    assert mot.generation.logits is None


@pytest.mark.asyncio
async def test_stream_timeout_signals_generation_thread():
    """Direct streaming signals the HF worker's cooperative cancel event on timeout."""
    backend = _make_backend()
    ctx = ChatContext().add(Message("user", "Hello"))
    cancel_event = MagicMock()

    async def _stalling_stream():
        await asyncio.sleep(1)
        yield "never"  # pragma: no cover

    with (
        patch(
            "mellea.backends.huggingface.AsyncTextIteratorStreamer",
            return_value=_stalling_stream(),
        ),
        patch(
            "mellea.backends.huggingface._install_cancel_stopping_criteria",
            return_value=cancel_event,
        ),
    ):
        output = await backend._generate_from_context_standard(
            Message("assistant", ""),
            ctx,
            model_options={ModelOption.STREAM: True, ModelOption.STREAM_TIMEOUT: 0.05},
        )

        with pytest.raises(TimeoutError, match="Stream timed out"):
            await output.astream()

    cancel_event.set.assert_called_once_with()


@pytest.mark.asyncio
async def test_kv_cache_stream_timeout_signals_generation_thread():
    """Direct KV-cache streaming signals the HF worker on timeout."""
    backend = _make_backend()
    ctx = ChatContext().add(Message("user", "Hello"))
    cancel_event = MagicMock()
    input_ids = torch.tensor([[1]])
    attention_mask = torch.tensor([[1]])

    async def _stalling_stream():
        await asyncio.sleep(1)
        yield "never"  # pragma: no cover

    with (
        patch(
            "mellea.backends.huggingface.AsyncTextIteratorStreamer",
            return_value=_stalling_stream(),
        ),
        patch(
            "mellea.backends.huggingface._install_cancel_stopping_criteria",
            return_value=cancel_event,
        ),
        patch.object(
            backend,
            "_make_merged_kv_cache",
            return_value=("", input_ids, MagicMock(), attention_mask),
        ),
    ):
        output = await backend._generate_from_context_with_kv_cache(
            Message("assistant", ""),
            ctx,
            model_options={ModelOption.STREAM: True, ModelOption.STREAM_TIMEOUT: 0.05},
        )

        with pytest.raises(TimeoutError, match="Stream timed out"):
            await output.astream()

    cancel_event.set.assert_called_once_with()


@pytest.mark.asyncio
async def test_intrinsic_logits_populated_when_option_set(stub_backend):
    """_generate_from_intrinsic populates generation.logits when ModelOption.LOGITS=True.

    generate_with_transformers wraps the raw GenerateDecoderOnlyOutput into a
    ChatCompletionResponse and discards it.  The backend proxies self._model so the
    raw output is intercepted and stashed for post_processing/_surface_logits.
    """
    vocab_size = 32000
    fake_scores = (torch.zeros(1, vocab_size), torch.zeros(1, vocab_size))
    fake_hf_output = GenerateDecoderOnlyOutput(
        sequences=torch.tensor([[1, 2]]),
        scores=fake_scores,
        logits=None,
        attentions=None,
        hidden_states=None,
        past_key_values=None,
    )

    backend = _make_intrinsic_backend_stub(stub_backend)
    # Wire real implementations so the full logits path runs.
    backend.processing = lambda *args, **kwargs: LocalHFBackend.processing(
        backend, *args, **kwargs
    )
    backend.post_processing = lambda *args, **kwargs: LocalHFBackend.post_processing(
        backend, *args, **kwargs
    )
    backend._surface_logits = lambda mot, hf_out: LocalHFBackend._surface_logits(
        backend, mot, hf_out
    )
    backend._use_caches = False
    backend.cache_put = MagicMock()
    backend._tokenizer = MagicMock(eos_token_id=0)
    backend.model_id = "stub-model"

    adapter = _make_intrinsic_adapter_stub()
    backend._added_adapters = {adapter.qualified_name: adapter}

    class _FakeChatCompletionResponse:
        class _Choice:
            class _Message:
                content = '{"score": 0.9}'

            message = _Message()

        choices = [_Choice()]

    def fake_transformers_inputs(rewritten, tokenizer, model, ll_tokenizer=None):
        generate_input = {"input_tokens": torch.tensor([[1]])}
        return generate_input, {}

    def fake_generate_with_transformers(tokenizer, model, generate_input, other_input):
        # Invoke model.generate so the proxy captures the raw output.
        model.generate(inputs=generate_input["input_tokens"])
        return _FakeChatCompletionResponse()

    class _FakeResultProcessorWithOutput:
        def __init__(self, *args, **kwargs):
            pass

        def transform(self, chunk, rewritten):
            return chunk

    with (
        patch(
            "mellea.backends.huggingface.granite_formatters.IntrinsicsRewriter",
            _FakeRewriter,
        ),
        patch(
            "mellea.backends.huggingface.granite_formatters.IntrinsicsResultProcessor",
            _FakeResultProcessorWithOutput,
        ),
        patch(
            "mellea.formatters.granite.base.util.chat_completion_request_to_transformers_inputs",
            side_effect=fake_transformers_inputs,
        ),
        patch(
            "mellea.formatters.granite.base.util.generate_with_transformers",
            side_effect=fake_generate_with_transformers,
        ),
    ):
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value=fake_hf_output)
        backend._model = mock_model

        output = await LocalHFBackend._generate_from_intrinsic(
            backend,
            Intrinsic("answerability"),
            ChatContext().add(Message("user", "Is the sky blue?")),
            model_options={ModelOption.LOGITS: True},
        )
        assert output._gen.generate is not None
        await output._gen.generate

        # Drain the queue to trigger _process (granite_formatters_processing), which
        # stashes the intercepted hf_output in mot._meta["hf_output"].
        while not output._gen.queue.empty():
            item = output._gen.queue.get_nowait()
            if item is not None:
                await output._gen.process(output, item)

        # Simulate the sentinel-driven completion that astream() performs before
        # calling _post_process, so post_processing's assertion mot.value is not None passes.
        output._computed = True

    # hf_output should now be stashed by granite_formatters_processing.
    assert output.raw.response is fake_hf_output, (
        "proxy must have captured the raw GenerateDecoderOnlyOutput"
    )
    input_ids = torch.tensor([[1]])
    await backend.post_processing(output, [], None, False, {}, None, input_ids)

    assert output.generation.logits is not None, (
        "logits must be populated on intrinsic path"
    )
    assert len(output.generation.logits) == len(fake_scores)
    assert all(t.shape == (vocab_size,) for t in output.generation.logits)


@pytest.mark.parametrize("images,audio", _MULTIMODAL_CASES)
@pytest.mark.asyncio
async def test_multimodal_blocks_raise_error(images, audio):
    """LocalHFBackend raises ValueError for image/audio inputs instead of silently dropping them."""
    backend = _make_backend()
    ctx = ChatContext().add(Message("user", "Hello", images=images, audio=audio))

    with pytest.raises(ValueError, match="LocalHFBackend does not support"):
        await backend._generate_from_context_standard(
            Message("assistant", ""), ctx, model_options={}
        )


@pytest.mark.asyncio
async def test_multimodal_blocks_in_action_raise_error():
    """LocalHFBackend raises ValueError when action contains image/audio blocks."""
    backend = _make_backend()
    ctx = ChatContext().add(Message("user", "Hello"))

    with pytest.raises(ValueError, match="LocalHFBackend does not support"):
        await backend._generate_from_context_standard(
            Message("assistant", "", images=[ImageBlock(_B64_PNG)]),
            ctx,
            model_options={},
        )


@pytest.mark.parametrize("images,audio", _MULTIMODAL_CASES)
@pytest.mark.asyncio
async def test_multimodal_blocks_kv_cache_path_raises_error(images, audio):
    """LocalHFBackend KV cache path raises ValueError for image/audio inputs."""
    backend = _make_backend()
    ctx = ChatContext().add(Message("user", "Hello", images=images, audio=audio))

    with pytest.raises(ValueError, match="LocalHFBackend does not support"):
        await backend._generate_from_context_with_kv_cache(
            Message("assistant", ""), ctx, model_options={}
        )


@pytest.mark.parametrize("images,audio", _MULTIMODAL_CASES)
@pytest.mark.asyncio
async def test_multimodal_blocks_in_raw_action_raises_error(images, audio):
    """_generate_from_raw raises ValueError for actions with image/audio blocks instead of silently dropping them."""
    backend = _make_backend()
    ctx = ChatContext().add(Message("user", "Hello"))
    action = Message("assistant", "", images=images, audio=audio)

    with pytest.raises(ValueError, match="LocalHFBackend does not support"):
        await backend._generate_from_raw([action], ctx, model_options={})


@pytest.mark.parametrize("images,audio", _MULTIMODAL_CASES)
@pytest.mark.asyncio
async def test_multimodal_blocks_in_raw_ctx_not_checked(images, audio):
    """_generate_from_raw does not scan ctx for multimodal content.

    ctx is accepted by the signature but never rendered on the raw path — only
    the actions are formatted and sent to the model. Multimodal blocks stored
    in the context do not cause an error here (they are simply unused).
    """
    backend = _make_backend()
    ctx = ChatContext().add(Message("user", "Hello", images=images, audio=audio))
    action = Message("assistant", "")

    # Should not raise — ctx content is not rendered by _generate_from_raw.
    # We mock the model to avoid loading weights; just verify no ValueError is raised.
    mock_outputs = MagicMock()
    mock_outputs.sequences = [MagicMock()]
    mock_outputs.sequences[0].__getitem__ = MagicMock(return_value=MagicMock())
    mock_outputs.scores = None
    mock_outputs.logits = None
    with patch.object(
        backend, "_generate_with_adapter_lock", return_value=mock_outputs
    ):
        with patch.object(
            backend._tokenizer,
            "__call__",
            return_value={
                "input_ids": MagicMock(size=lambda i: 0),
                "attention_mask": MagicMock(),
            },
        ):
            with patch.object(backend._tokenizer, "batch_decode", return_value=[""]):
                await backend._generate_from_raw([action], ctx, model_options={})


@pytest.mark.parametrize("images,audio", _MULTIMODAL_CASES)
@pytest.mark.asyncio
async def test_multimodal_blocks_on_instruction_in_ctx_raise_error(images, audio):
    """LocalHFBackend raises ValueError when an Instruction in ctx carries image/audio blocks.

    The guard uses hasattr(c, "images") / hasattr(c, "audio"), so it must fire
    for Instruction just as it does for Message.
    """
    backend = _make_backend()
    ctx = ChatContext().add(
        Instruction(description="describe this", images=images, audio=audio)
    )

    with pytest.raises(ValueError, match="LocalHFBackend does not support"):
        await backend._generate_from_context_standard(
            Message("assistant", ""), ctx, model_options={}
        )


@pytest.mark.parametrize("images,audio", _MULTIMODAL_CASES)
@pytest.mark.asyncio
async def test_multimodal_blocks_on_instruction_as_action_raise_error(images, audio):
    """LocalHFBackend raises ValueError when an Instruction used as the action carries image/audio.

    The guard checks the action component as well as components in ctx; this test
    exercises the action branch via Instruction instead of Message.
    """
    backend = _make_backend()
    ctx = ChatContext().add(Message("user", "Hello"))

    with pytest.raises(ValueError, match="LocalHFBackend does not support"):
        await backend._generate_from_context_standard(
            Instruction(description="describe this", images=images, audio=audio),
            ctx,
            model_options={},
        )


@pytest.mark.parametrize("images,audio", _MULTIMODAL_CASES)
@pytest.mark.asyncio
async def test_multimodal_blocks_in_intrinsic_ctx_raise_error(
    stub_backend, images, audio
):
    """_generate_from_intrinsic raises ValueError when ctx contains image/audio blocks.

    The guard on the intrinsic path passes `action=None` and scans only the context;
    this test exercises that branch directly.
    """
    backend = _make_intrinsic_backend_stub(stub_backend)
    adapter = _make_intrinsic_adapter_stub()
    backend._added_adapters = {adapter.qualified_name: adapter}
    ctx = ChatContext().add(Message("user", "Hello", images=images, audio=audio))

    with pytest.raises(ValueError, match="LocalHFBackend does not support"):
        await LocalHFBackend._generate_from_intrinsic(
            backend, Intrinsic("answerability"), ctx, model_options={}
        )


# ---------------------------------------------------------------------------
# Regression tests for issue #1510: bounded whitespace_pattern required
# ---------------------------------------------------------------------------
# llguidance's whitespace_flexible=False (compact JSON) interacts badly with
# the backend's default greedy decoding, putting it into states where the
# highest-probability grammar-compatible token closes an array immediately,
# silently collapsing {"result": [...]} to {"result": []}.
# To prevent this, all four grammar_from_json_schema call sites must enforce a
# bounded whitespace_pattern (which allows space and prevents unlimited run-away
# whitespace generation, resolving PR #1513 feedback).
# These tests assert that invariant via mock without loading any real model.


class _FakeSchema:
    """Minimal Pydantic-compatible schema stub."""

    @staticmethod
    def model_json_schema() -> dict:
        return {"type": "object", "properties": {"result": {"type": "array"}}}


def _mock_chat_template_output() -> MagicMock:
    """Return a mock that looks like a tokenizer output dict with a .to() method.

    apply_chat_template returns a BatchEncoding (dict-like) which gets a .to(device)
    call immediately after. Plain dicts don't have .to(), so the mock must.
    """
    ids = torch.zeros(1, 4, dtype=torch.long)
    attn = torch.ones(1, 4, dtype=torch.long)
    obj = MagicMock()
    obj.__getitem__ = lambda s, k: ids if k == "input_ids" else attn
    obj.to = lambda device: obj
    return obj


def _assert_whitespace_pattern_set(captured: list[dict]) -> None:
    assert captured, "grammar_from_json_schema was never called"
    for call_defaults in captured:
        assert call_defaults.get("whitespace_pattern") == r"[\x20\x0A\x0D\x09]{0,20}", (
            f"Expected bounded whitespace_pattern, got {call_defaults!r} — "
            "see issue #1510 and PR #1513 review"
        )


@pytest.mark.asyncio
async def test_whitespace_pattern_set_in_generate_from_context_standard():
    """Regression (#1510): _generate_from_context_standard must call
    grammar_from_json_schema with bounded whitespace_pattern.

    Without the fix, the call passes whitespace_flexible=False, which can cause
    silent array collapse to [] under greedy decoding.
    """
    # _make_backend() patches llguidance during construction; re-patch just for
    # the method call to intercept the grammar_from_json_schema invocation.
    backend = _make_backend()
    backend._tokenizer = MagicMock()
    backend._tokenizer.apply_chat_template.return_value = _mock_chat_template_output()
    backend._model = MagicMock()
    ctx = ChatContext().add(Message("user", "list facts"))

    captured: list[dict] = []

    def _capture_grammar(schema, overrides=None):
        captured.append(overrides or {})
        return "stub-grammar"

    # The real generate() call runs in a background task (output._gen.generate)
    # that this method returns without awaiting, so its mocked result has no
    # bearing on whether the method call itself completes.
    with (
        patch("mellea.backends.huggingface.llguidance") as mock_llg,
        patch(
            "mellea.backends.huggingface.asyncio.to_thread", return_value=MagicMock()
        ),
    ):
        mock_llg.LLMatcher.grammar_from_json_schema.side_effect = _capture_grammar
        output = await backend._generate_from_context_standard(
            Instruction(description="test"), ctx, model_options={}, _format=_FakeSchema
        )
    await output._gen.generate

    _assert_whitespace_pattern_set(captured)


@pytest.mark.asyncio
async def test_whitespace_pattern_set_in_generate_from_raw():
    """Regression (#1510): _generate_from_raw must call grammar_from_json_schema
    with bounded whitespace_pattern.
    """
    backend = _make_backend()
    # _generate_from_raw calls self._tokenizer(prompts, ...).to(device), so the
    # mock tokenizer must be callable and return a .to()-able object.
    tok_output = MagicMock()
    tok_output.to = lambda device: tok_output
    tok_output.__getitem__ = lambda s, k: torch.zeros(1, 4, dtype=torch.long)
    backend._tokenizer = MagicMock(return_value=tok_output)
    backend._tokenizer.batch_decode = MagicMock(return_value=["stub-completion"])
    backend._model = MagicMock()
    ctx = ChatContext().add(Message("user", "list facts"))

    # Unlike the context-based methods, _generate_from_raw awaits the generate
    # call directly, so it needs a realistic GenerateDecoderOnlyOutput result.
    fake_outputs = GenerateDecoderOnlyOutput(
        sequences=torch.zeros(1, 7, dtype=torch.long),
        scores=None,
        logits=None,
        attentions=None,
        hidden_states=None,
        past_key_values=None,
    )

    captured: list[dict] = []

    def _capture_grammar(schema, overrides=None):
        captured.append(overrides or {})
        return "stub-grammar"

    with (
        patch("mellea.backends.huggingface.llguidance") as mock_llg,
        patch(
            "mellea.backends.huggingface.asyncio.to_thread", return_value=fake_outputs
        ),
    ):
        mock_llg.LLMatcher.grammar_from_json_schema.side_effect = _capture_grammar
        await backend._generate_from_raw(
            [Instruction(description="test")], ctx, format=_FakeSchema, model_options={}
        )

    _assert_whitespace_pattern_set(captured)


@pytest.mark.asyncio
async def test_whitespace_pattern_set_in_generate_from_context_with_kv_cache():
    """Regression (#1510): _generate_from_context_with_kv_cache must call
    grammar_from_json_schema with bounded whitespace_pattern.
    """
    backend = _make_backend()
    backend._model = MagicMock()
    ctx = ChatContext().add(Message("user", "list facts"))

    input_ids = torch.tensor([[1]])
    attention_mask = torch.tensor([[1]])

    captured: list[dict] = []

    def _capture_grammar(schema, overrides=None):
        captured.append(overrides or {})
        return "stub-grammar"

    # The real generate() call runs in a background task (output._gen.generate)
    # that this method returns without awaiting, so its mocked result has no
    # bearing on whether the method call itself completes.
    with (
        patch("mellea.backends.huggingface.llguidance") as mock_llg,
        patch.object(
            backend,
            "_make_merged_kv_cache",
            return_value=("", input_ids, MagicMock(), attention_mask),
        ),
        patch(
            "mellea.backends.huggingface.asyncio.to_thread", return_value=MagicMock()
        ),
    ):
        mock_llg.LLMatcher.grammar_from_json_schema.side_effect = _capture_grammar
        output = await backend._generate_from_context_with_kv_cache(
            Instruction(description="test"), ctx, model_options={}, _format=_FakeSchema
        )
    await output._gen.generate

    _assert_whitespace_pattern_set(captured)


def test_whitespace_pattern_set_in_chat_completion_request_to_transformers_inputs():
    """Regression (#1510): chat_completion_request_to_transformers_inputs (the
    OpenAI-compatible /chat/completions path used by `m serve`) must call
    grammar_from_json_schema with bounded whitespace_pattern.
    """
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = torch.zeros(1, 4, dtype=torch.long)
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1

    model = MagicMock()
    model.device = "cpu"

    request = {
        "messages": [{"role": "user", "content": "list facts"}],
        "extra_body": {"structured_outputs": {"json": _FakeSchema.model_json_schema()}},
    }

    captured: list[dict] = []

    def _capture_grammar(schema, overrides=None):
        captured.append(overrides or {})
        return "stub-grammar"

    with patch(
        "llguidance.LLMatcher.grammar_from_json_schema", side_effect=_capture_grammar
    ):
        chat_completion_request_to_transformers_inputs(
            request, tokenizer, model, ll_tokenizer=MagicMock()
        )

    _assert_whitespace_pattern_set(captured)


@pytest.mark.asyncio
async def test_whitespace_pattern_cannot_be_defeated_by_schema():
    """Regression (#1510): Custom schemas attempting to force compact JSON
    (whitespace_flexible=False) must be overridden to our bounded whitespace_pattern across all entry points.
    """

    class _FakeCompactSchema:
        @staticmethod
        def model_json_schema() -> dict:
            return {
                "type": "object",
                "properties": {"result": {"type": "array"}},
                "x-guidance": {"whitespace_flexible": False},
            }

    backend = _make_backend()
    backend._tokenizer = MagicMock()
    backend._tokenizer.apply_chat_template.return_value = _mock_chat_template_output()
    tok_output = MagicMock()
    tok_output.to = lambda device: tok_output
    tok_output.__getitem__ = lambda s, k: torch.zeros(1, 4, dtype=torch.long)
    backend._tokenizer.return_value = tok_output
    backend._tokenizer.batch_decode = MagicMock(return_value=["stub-completion"])
    backend._model = MagicMock()
    backend._model.device = torch.device("cpu")
    ctx = ChatContext().add(Message("user", "list facts"))

    input_ids = torch.tensor([[1]])
    attention_mask = torch.tensor([[1]])

    # We want to trace each of the 4 paths
    for path_name in ("context_standard", "raw", "context_kv_cache", "chat_completion"):
        captured: list[dict] = []

        def _capture_grammar(schema, overrides=None):
            captured.append(overrides or {})
            return "stub-grammar"

        if path_name in ("context_standard", "context_kv_cache"):
            backend._tokenizer.apply_chat_template.return_value = (
                _mock_chat_template_output()
            )
        elif path_name == "chat_completion":
            backend._tokenizer.apply_chat_template.return_value = torch.zeros(
                1, 4, dtype=torch.long
            )
            backend._tokenizer.pad_token_id = 0
            backend._tokenizer.eos_token_id = 1

        with (
            patch("mellea.backends.huggingface.llguidance") as mock_llg,
            patch(
                "mellea.backends.huggingface.asyncio.to_thread",
                return_value=GenerateDecoderOnlyOutput(
                    sequences=torch.zeros(1, 7, dtype=torch.long),
                    scores=None,
                    logits=None,
                    attentions=None,
                    hidden_states=None,
                    past_key_values=None,
                )
                if path_name == "raw"
                else MagicMock(),
            ),
        ):
            mock_llg.LLMatcher.grammar_from_json_schema.side_effect = _capture_grammar

            if path_name == "context_standard":
                output = await backend._generate_from_context_standard(
                    Instruction(description="test"),
                    ctx,
                    model_options={},
                    _format=_FakeCompactSchema,
                )
                await output._gen.generate
            elif path_name == "raw":
                await backend._generate_from_raw(
                    [Instruction(description="test")],
                    ctx,
                    format=_FakeCompactSchema,
                    model_options={},
                )
            elif path_name == "context_kv_cache":
                with patch.object(
                    backend,
                    "_make_merged_kv_cache",
                    return_value=("", input_ids, MagicMock(), attention_mask),
                ):
                    output = await backend._generate_from_context_with_kv_cache(
                        Instruction(description="test"),
                        ctx,
                        model_options={},
                        _format=_FakeCompactSchema,
                    )
                    await output._gen.generate
            elif path_name == "chat_completion":
                request = {
                    "messages": [{"role": "user", "content": "list facts"}],
                    "extra_body": {
                        "structured_outputs": {
                            "json": _FakeCompactSchema.model_json_schema()
                        }
                    },
                }
                with patch(
                    "llguidance.LLMatcher.grammar_from_json_schema",
                    side_effect=_capture_grammar,
                ):
                    chat_completion_request_to_transformers_inputs(
                        request,
                        backend._tokenizer,
                        backend._model,
                        ll_tokenizer=MagicMock(),
                    )

        assert len(captured) == 1, (
            f"grammar_from_json_schema was not called for {path_name}"
        )
        assert captured[0].get("whitespace_pattern") == r"[\x20\x0A\x0D\x09]{0,20}", (
            f"Expected bounded whitespace_pattern to override False in {path_name}"
        )


async def _post_process_holding_only_weakrefs(
    backend: LocalHFBackend, n_steps: int
) -> tuple[ModelOutputThunk, dict[str, list["weakref.ref"]]]:
    """Run post_processing on a MOT and return weakrefs to the raw tensors it cleared.

    Builds a `GenerateDecoderOnlyOutput` whose `scores`, `logits`, and
    `past_key_values` hold fresh tensors, takes weakrefs to those tensors, then
    drops every strong local reference so the only remaining strong references
    are the ones the `ModelOutput` mapping keeps. After `post_processing` clears
    the fields, a correct clear drops the mapping entries and the weakrefs die on
    the next GC; the buggy attribute-set/`del` leaves them alive.

    Args:
        backend: The backend whose `post_processing` is exercised.
        n_steps: Number of decode steps (length of the scores/logits tuples).

    Returns:
        The finalized thunk and a dict mapping field name to the list of
        weakrefs (one per step) to the tensors that field held.
    """
    input_ids = torch.tensor([[1]])
    sequences = torch.tensor([[0, 0]])
    scores = tuple(torch.zeros(1, 32000) for _ in range(n_steps))
    logits = tuple(torch.ones(1, 32000) for _ in range(n_steps))
    kv = tuple(torch.zeros(2, 4) for _ in range(n_steps))

    refs: dict[str, list[weakref.ref]] = {
        "scores": [weakref.ref(t) for t in scores],
        "logits": [weakref.ref(t) for t in logits],
        "past_key_values": [weakref.ref(t) for t in kv],
    }

    mot = ModelOutputThunk(value="hi")
    mot._call.action = Message("user", "noop")
    mot._call.model_options = {}
    mot.raw.response = GenerateDecoderOnlyOutput(
        sequences=sequences,
        scores=scores,
        logits=logits,
        attentions=None,
        hidden_states=None,
        # past_key_values is typed Cache | None, but ModelOutput stores whatever
        # it is handed in its mapping; a tuple of tensors is all this fixture needs
        # to exercise the clearing path.
        past_key_values=kv,  # type: ignore[arg-type]
    )

    # Drop every strong local reference to the tensors and their tuples. After
    # this, the ModelOutput mapping is the only thing keeping them alive.
    del scores, logits, kv

    await backend.post_processing(mot, [], None, False, {}, None, input_ids)

    return mot, refs


@pytest.mark.asyncio
async def test_post_processing_clearing_raw_logits_actually_releases_them():
    """Clearing `hf_output.logits` must drop the tensors, not just the attribute.

    `GenerateDecoderOnlyOutput` is a `ModelOutput`, i.e. an `OrderedDict` subclass
    that mirrors every field into the mapping. `ModelOutput.__setattr__` skips the
    mapping write when the value is `None`, and `ModelOutput` defines no
    `__delattr__`, so `out.logits = None` and `del out.logits` both leave the
    mapping entry — and therefore the tensors — in place. Any code that nulls a
    field to free memory while keeping the container has to clear the mapping too.
    """
    backend = _make_backend(1)
    backend._use_caches = True  # keeps raw.response, so the container survives

    mot, refs = await _post_process_holding_only_weakrefs(backend, n_steps=2)
    gc.collect()
    gc.collect()

    assert mot.raw.response is not None, "test setup: raw.response should be retained"
    assert mot.raw.response.logits is None, "test setup: logits attribute was cleared"
    for step, ref in enumerate(refs["logits"]):
        assert ref() is None, (
            f"raw logits tensor for step {step} is still alive after hf_output.logits "
            "was set to None — the ModelOutput mapping entry still references it"
        )


@pytest.mark.asyncio
async def test_post_processing_clearing_scores_and_kv_actually_releases_them():
    """The caching branch clears `scores`/`past_key_values` off the MOT.

    Those tensors are retained in the LRU cache, so nulling the fields on the
    `ModelOutput` must actually release the container's strong reference (via the
    mapping), otherwise the KV cache is held twice — once in the LRU and once on
    the MOT — defeating the cleanup.
    """
    backend = _make_backend(1)
    backend._use_caches = True

    mot, refs = await _post_process_holding_only_weakrefs(backend, n_steps=2)
    gc.collect()
    gc.collect()

    assert mot.raw.response is not None, "test setup: raw.response should be retained"
    assert mot.raw.response.scores is None, "test setup: scores attribute was cleared"
    assert mot.raw.response.past_key_values is None, (
        "test setup: past_key_values attribute was cleared"
    )
    # With the default zero-capacity LRU, cache_put evicts immediately, so the
    # MOT's ModelOutput mapping is the only thing that could keep the scores
    # tensors alive; clearing the field through the mapping must release them.
    for step, ref in enumerate(refs["scores"]):
        assert ref() is None, (
            f"raw scores tensor for step {step} is still alive after hf_output.scores "
            "was set to None — the ModelOutput mapping entry still references it"
        )
