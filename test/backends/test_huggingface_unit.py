"""Unit tests for HuggingFace backend pure-logic helpers — no model load required."""

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

from transformers.generation.utils import GenerateDecoderOnlyOutput

from mellea.backends import ModelOption
from mellea.backends.adapters import AdapterMixin, IntrinsicAdapter
from mellea.backends.adapters._core import Identity
from mellea.backends.huggingface import LocalHFBackend
from mellea.core import ModelOutputThunk
from mellea.stdlib.components import Intrinsic, Message
from mellea.stdlib.context import ChatContext


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
