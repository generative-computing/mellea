"""Unit tests for HuggingFace backend pure-logic helpers — no model load required."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("torch", reason="torch not installed — install mellea[hf]")
pytest.importorskip(
    "transformers", reason="transformers not installed — install mellea[hf]"
)
pytest.importorskip(
    "llguidance", reason="llguidance not installed — install mellea[hf]"
)

from mellea.backends import ModelOption
from mellea.backends.adapters import IntrinsicAdapter
from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components import Intrinsic, Message
from mellea.stdlib.context import ChatContext


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
    return adapter


def _make_intrinsic_backend_stub(stub_backend):
    stub_backend.formatter = SimpleNamespace(
        to_chat_messages=lambda linearized_ctx: [Message("user", "Is the sky blue?")]
    )
    stub_backend._added_adapters = {}
    stub_backend._tokenizer = object()
    stub_backend._model = object()
    stub_backend._llguidance_tokenizer = object()
    stub_backend._get_hf_model_id = lambda: "stub-model"
    stub_backend._make_backend_specific_and_remove = lambda opts: (
        LocalHFBackend._make_backend_specific_and_remove(stub_backend, opts)
    )
    stub_backend.post_processing = lambda *args, **kwargs: None
    stub_backend._generate_with_adapter_lock = (
        lambda adapter_name, generate_func, *args: generate_func(*args)
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

    with (
        patch(
            "mellea.backends.huggingface.get_adapter_for_intrinsic",
            return_value=adapter,
        ),
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
        assert output._generate is not None
        await output._generate

    assert captured["generate_input"]["do_sample"] is False
    assert "temperature" not in captured["generate_input"]
