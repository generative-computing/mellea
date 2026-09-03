# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for HuggingFace backend pure-logic helpers — no model load required."""

import asyncio
import json
import threading
import time
import warnings
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

torch = pytest.importorskip("torch", reason="torch not installed — install mellea[hf]")
pytest.importorskip(
    "transformers", reason="transformers not installed — install mellea[hf]"
)
pytest.importorskip(
    "llguidance", reason="llguidance not installed — install mellea[hf]"
)

import base64
import gc
import struct
import weakref

from transformers.cache_utils import CacheLayerMixin, DynamicCache
from transformers.generation.utils import (
    GenerateBeamDecoderOnlyOutput,
    GenerateDecoderOnlyOutput,
)

from mellea.backends import ModelOption
from mellea.backends.adapters import (
    AdapterMixin,
    AdapterType,
    EmbeddedBinding,
    IntrinsicAdapter,
    ServerMediatedBinding,
)
from mellea.backends.adapters._core import Identity
from mellea.backends.adapters.adapter import (
    EmbeddedIntrinsicAdapter,
    _ShimWeightsBinding,
)
from mellea.backends.adapters.catalog import IntrinsicsCatalogEntry
from mellea.backends.huggingface import LocalHFBackend
from mellea.core import ModelOutputThunk
from mellea.formatters.granite.base.util import (
    chat_completion_request_to_transformers_inputs,
)
from mellea.plugins.types import HookType
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
from test.backends.test_adapters._hook_capture import (
    capture_adapter_hooks,
    hook_payloads,
)

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

    def model_dump(self):
        return {
            "messages": [],
            "extra_body": {},
            "model": "adapter-model",
            "temperature": self.temperature,
        }


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
    object.__setattr__(adapter, "weights", _ShimWeightsBinding())
    return adapter


def _make_embedded_adapter_stub() -> EmbeddedIntrinsicAdapter:
    """Build an embedded adapter without exposing its deprecation warning in tests."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return EmbeddedIntrinsicAdapter(
            intrinsic_name="answerability",
            config={
                "model": None,
                "response_format": None,
                "transformations": None,
                "instruction": None,
                "parameters": {"max_completion_tokens": 8},
                "sentence_boundaries": None,
            },
            technology="alora",
        )


def _make_intrinsic_backend_stub(stub_backend):
    stub_backend.formatter = SimpleNamespace(
        to_chat_messages=lambda linearized_ctx: [Message("user", "Is the sky blue?")]
    )
    stub_backend._added_adapters = {}
    stub_backend._composed_adapters = {}
    stub_backend._composed_adapter_configs = {}
    stub_backend._tokenizer = object()
    stub_backend._model = object()
    stub_backend._llguidance_tokenizer = object()
    stub_backend._model_id = "stub-model"
    stub_backend._provider = "huggingface"
    stub_backend._make_backend_specific_and_remove = lambda opts: (
        LocalHFBackend._make_backend_specific_and_remove(stub_backend, opts)
    )
    stub_backend.post_processing = lambda *args, **kwargs: None
    # Bypasses locking/adapter_scope entirely — these tests exercise option-merging
    # and logits capture, not activation semantics (see
    # test_generate_intrinsic_with_adapter_scope_* below for that).
    stub_backend._generate_intrinsic_with_adapter_scope = (
        lambda adapter, generate_func, *args, **kwargs: generate_func(*args, **kwargs)
    )
    stub_backend._generate_embedded_with_generation_lock = (
        lambda generate_func, *args, **kwargs: generate_func(*args, **kwargs)
    )
    stub_backend._find_adapter = lambda cap, types=None: AdapterMixin._find_adapter(
        stub_backend, cap, types
    )
    stub_backend._intrinsic_adapter_name_and_config = lambda adapter: (
        LocalHFBackend._intrinsic_adapter_name_and_config(stub_backend, adapter)
    )
    # Composed-Adapter counterpart of _generate_intrinsic_with_adapter_scope
    # above — same bypass, for the same reason (these tests exercise
    # option-merging/logits capture, not activation semantics).
    stub_backend._generate_composed_local_file_with_adapter_scope = (
        lambda adapter, generate_func, *args, **kwargs: generate_func(*args, **kwargs)
    )
    return stub_backend


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("ibm-granite/granite-3.3-8b-instruct", "granite-3.3-8b-instruct"),
        ("granite-switch", "granite-switch"),
        ("/tmp/granite-switch-checkpoint", "granite-switch-checkpoint"),
    ],
)
def test_base_model_name_handles_hub_ids_and_local_checkpoints(model_id, expected):
    """Embedded registration accepts local checkpoint paths and unqualified model IDs."""
    backend = _make_backend()
    backend._model_id = model_id

    assert backend.base_model_name == expected


def test_load_embedded_adapters_registers_checkpoint_adapters():
    """The constructor registers embedded adapters without invoking PEFT loading."""
    adapter = _make_embedded_adapter_stub()
    mock_tok = MagicMock(eos_token_id=0, vocab_size=32000)
    mock_tok._tokenizer = MagicMock()
    mock_tok._tokenizer.get_vocab_size.return_value = 32000
    mock_tok.__len__ = MagicMock(return_value=32000)
    mock_model = MagicMock(vocab_size=32000)

    with (
        patch("mellea.backends.huggingface.llguidance") as mock_llg,
        patch.object(
            EmbeddedIntrinsicAdapter, "from_source", return_value=[adapter]
        ) as mock_from_source,
    ):
        mock_llg.hf.from_tokenizer.return_value = MagicMock(vocab_size=32000)
        backend = LocalHFBackend(
            model_id="ibm-granite/granite-3.3-8b-instruct",
            custom_config=(mock_tok, mock_model, torch.device("cpu")),
            load_embedded_adapters=True,
            adapter_source="/tmp/switch-checkpoint",
        )

    mock_from_source.assert_called_once_with(
        "/tmp/switch-checkpoint", revision="main", cache_dir=None, intrinsic_name=None
    )
    assert backend.list_adapters() == ["answerability_alora"]
    # register_embedded_adapter_model discovers via the non-shim
    # _discover_embedded_adapters factory (Epic #929, issue #1144), which
    # lifts the shim's identity/io_contract/weights into a composed Adapter
    # stored in _composed_adapters, not the shim instance itself.
    registered = backend._composed_adapters["answerability_alora"]
    assert registered.identity == adapter.identity
    assert registered.weights is adapter.weights
    assert isinstance(adapter.weights, EmbeddedBinding)
    # The composed Adapter is registered, not the shim instance, so the
    # shim's own `.backend` is never touched; the binding it shares with the
    # composed Adapter (asserted above) is what actually gets stamped.
    assert adapter.weights.source == backend.base_model_name
    backend._model.load_adapter.assert_not_called()


def test_load_embedded_adapters_requires_granite_switch_package():
    """Automatic Switch checkpoint loading fails with an actionable dependency error."""
    with patch(
        "mellea.backends.huggingface.importlib.import_module",
        side_effect=ImportError("granite_switch is unavailable"),
    ):
        with pytest.raises(ImportError, match=r'pip install "mellea\[hf\]"'):
            LocalHFBackend(
                model_id="ibm-granite/granite-switch-4.1-3b-preview",
                load_embedded_adapters=True,
            )


def test_granite_switch_transformers_override_warning_is_self_retiring(monkeypatch):
    """The compatibility warning appears only while Switch metadata excludes Transformers."""
    import mellea.backends.huggingface as huggingface

    monkeypatch.setattr(huggingface, "_SWITCH_TRANSFORMERS_WARNING_EMITTED", False)
    monkeypatch.setattr(
        huggingface.metadata,
        "requires",
        lambda distribution: ["transformers>=5.5.1,<5.10.0"],
    )
    monkeypatch.setattr(huggingface.metadata, "version", lambda distribution: "5.10.2")

    with pytest.warns(UserWarning, match="explicit compatibility override"):
        huggingface._warn_if_granite_switch_transformers_override_is_active()

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        huggingface._warn_if_granite_switch_transformers_override_is_active()


def test_granite_switch_transformers_override_warning_ignores_bad_metadata(monkeypatch):
    """Malformed Granite Switch metadata must not prevent backend construction."""
    import mellea.backends.huggingface as huggingface

    monkeypatch.setattr(huggingface, "_SWITCH_TRANSFORMERS_WARNING_EMITTED", False)
    monkeypatch.setattr(
        huggingface.metadata, "requires", lambda distribution: ["transformers @@@"]
    )
    monkeypatch.setattr(huggingface.metadata, "version", lambda distribution: "5.10.2")

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        huggingface._warn_if_granite_switch_transformers_override_is_active()


def test_granite_switch_transformers_override_warning_clears_when_metadata_updates(
    monkeypatch,
):
    """No warning remains after Granite Switch declares the installed version compatible."""
    import mellea.backends.huggingface as huggingface

    monkeypatch.setattr(huggingface, "_SWITCH_TRANSFORMERS_WARNING_EMITTED", False)
    monkeypatch.setattr(
        huggingface.metadata,
        "requires",
        lambda distribution: ["transformers>=5.5.1,<6.0.0"],
    )
    monkeypatch.setattr(huggingface.metadata, "version", lambda distribution: "5.10.2")

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        huggingface._warn_if_granite_switch_transformers_override_is_active()


def test_base_model_name_strips_trailing_path_separator():
    """A local checkpoint path keeps its final directory name."""
    backend = _make_backend()
    backend._model_id = "/tmp/granite-switch-checkpoint/"

    assert backend.base_model_name == "granite-switch-checkpoint"


def test_chat_completion_request_forwards_template_kwargs_to_transformers():
    """Template kwargs survive request conversion for local Granite Switch inference."""
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = torch.zeros(1, 4, dtype=torch.long)
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    model = MagicMock()
    model.device = "cpu"

    chat_completion_request_to_transformers_inputs(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "extra_body": {
                "chat_template_kwargs": {
                    "adapter_name": "answerability",
                    "template_option": "preserved",
                }
            },
        },
        tokenizer,
        model,
    )

    tokenizer.apply_chat_template.assert_called_once_with(
        conversation=[{"role": "user", "content": "hello"}],
        add_generation_prompt=True,
        adapter_name="answerability",
        template_option="preserved",
        return_tensors="pt",
    )


@pytest.mark.parametrize(
    "reserved_name",
    ["conversation", "tools", "documents", "add_generation_prompt", "return_tensors"],
)
def test_chat_completion_request_rejects_reserved_template_kwargs(reserved_name):
    """Template kwargs must not overwrite framework-owned chat-template inputs."""
    tokenizer = MagicMock()
    model = MagicMock()

    with pytest.raises(ValueError, match="cannot override Hugging Face"):
        chat_completion_request_to_transformers_inputs(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "extra_body": {"chat_template_kwargs": {reserved_name: "bad"}},
            },
            tokenizer,
            model,
        )

    tokenizer.apply_chat_template.assert_not_called()


def test_chat_completion_request_rejects_non_dict_template_kwargs():
    """Template kwargs must be a mapping before they reach the tokenizer."""
    with pytest.raises(TypeError, match="must be a dict"):
        chat_completion_request_to_transformers_inputs(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "extra_body": {"chat_template_kwargs": ["not", "a", "dict"]},
            },
            MagicMock(),
            MagicMock(),
        )


@pytest.mark.asyncio
async def test_embedded_intrinsic_activates_template_without_peft(stub_backend):
    """Embedded calls select the template control token without PEFT lifecycle work."""
    backend = _make_intrinsic_backend_stub(stub_backend)
    adapter = _make_embedded_adapter_stub()
    backend._added_adapters = {adapter.qualified_name: adapter}
    peft_scope = MagicMock()
    backend._generate_intrinsic_with_adapter_scope = peft_scope
    captured: dict[str, object] = {}

    def fake_transformers_inputs(request, tokenizer, model, ll_tokenizer=None):
        captured["request"] = request
        return {"input_tokens": object()}, {}

    def fake_generate_with_transformers(tokenizer, model, generate_input, other_input):
        return object()

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
            model_options={},
        )
        assert output._gen.generate is not None
        await output._gen.generate

    request = captured["request"]
    assert isinstance(request, dict)
    assert request["extra_body"]["chat_template_kwargs"]["adapter_name"] == (
        "answerability"
    )
    assert "model" not in request
    peft_scope.assert_not_called()


class _FakeChatCompletionResponseWithContent:
    def __init__(self, content: str):
        message = SimpleNamespace(content=content)
        self.choices = [SimpleNamespace(message=message)]


async def _run_embedded_intrinsic_and_collect_invocation_payloads(
    stub_backend, result_processor_cls: type
) -> tuple[list, Exception | None]:
    """Drives `_generate_from_intrinsic` for an embedded adapter to resolution.

    Returns `(payloads, raised)`: the invocation-complete payloads fired via
    `_core.invoke_hook` while `output._gen.process` resolves, and the
    exception `process()` raised (or `None` on success). Callers must assert
    on `raised`, not just the payloads, so a swallowed raise can't pass.
    """
    backend = _make_intrinsic_backend_stub(stub_backend)
    backend.processing = AsyncMock(return_value=None)
    adapter = _make_embedded_adapter_stub()
    backend._added_adapters = {adapter.qualified_name: adapter}

    def fake_transformers_inputs(request, tokenizer, model, ll_tokenizer=None):
        return {"input_tokens": object()}, {}

    def fake_generate_with_transformers(tokenizer, model, generate_input, other_input):
        return _FakeChatCompletionResponseWithContent('{"result": "ok"}')

    with (
        patch(
            "mellea.backends.huggingface.granite_formatters.IntrinsicsRewriter",
            _FakeRewriter,
        ),
        patch(
            "mellea.backends.huggingface.granite_formatters.IntrinsicsResultProcessor",
            result_processor_cls,
        ),
        patch(
            "mellea.formatters.granite.base.util.chat_completion_request_to_transformers_inputs",
            side_effect=fake_transformers_inputs,
        ),
        patch(
            "mellea.formatters.granite.base.util.generate_with_transformers",
            side_effect=fake_generate_with_transformers,
        ),
        patch("mellea.backends.adapters._core.has_plugins", return_value=True),
        patch(
            "mellea.backends.adapters._core.invoke_hook", new_callable=AsyncMock
        ) as mock_invoke,
    ):
        output = await LocalHFBackend._generate_from_intrinsic(
            backend,
            Intrinsic("answerability"),
            ChatContext().add(Message("user", "Is the sky blue?")),
            model_options={},
        )
        assert output._gen.generate is not None
        await output._gen.generate

        assert output._gen.process is not None
        raised: Exception | None = None
        while not output._gen.queue.empty():
            item = output._gen.queue.get_nowait()
            if item is not None:
                # The schema_error/error cases deliberately make
                # granite_formatters_processing raise; capture it (rather than
                # swallowing it) so callers can assert on it directly.
                try:
                    await output._gen.process(output, item)
                except Exception as e:
                    raised = e

    payloads = [
        call.args[1]
        for call in mock_invoke.call_args_list
        if call.args[0] is HookType.ADAPTER_FUNCTION_INVOCATION_COMPLETE
    ]
    return payloads, raised


@pytest.mark.asyncio
async def test_embedded_intrinsic_invocation_complete_fires_success(stub_backend):
    pytest.importorskip("cpex", reason="cpex not installed — install mellea[hooks]")

    class _PassthroughResultProcessor:
        def __init__(self, *args, **kwargs):
            pass

        def transform(self, chunk, rewritten):
            return chunk

    payloads, raised = await _run_embedded_intrinsic_and_collect_invocation_payloads(
        stub_backend, _PassthroughResultProcessor
    )

    assert raised is None
    assert len(payloads) == 1
    assert payloads[0].name == "answerability"
    assert payloads[0].binding_type == "embedded"
    assert payloads[0].adapter_type == "alora"
    assert payloads[0].outcome == "success"
    assert payloads[0].error is None


@pytest.mark.asyncio
async def test_embedded_intrinsic_invocation_complete_fires_schema_error(stub_backend):
    # Pins #1142/#1559: a non-JSON response must record schema_error, not success.
    pytest.importorskip("cpex", reason="cpex not installed — install mellea[hooks]")

    class _JSONDecodeErrorResultProcessor:
        def __init__(self, *args, **kwargs):
            pass

        def transform(self, chunk, rewritten):
            raise json.JSONDecodeError("bad json", "not valid json", 0)

    payloads, raised = await _run_embedded_intrinsic_and_collect_invocation_payloads(
        stub_backend, _JSONDecodeErrorResultProcessor
    )

    assert isinstance(raised, Exception)
    assert "did not return a JSON" in str(raised)
    assert isinstance(raised.__cause__, json.JSONDecodeError)
    assert len(payloads) == 1
    assert payloads[0].outcome == "schema_error"
    assert payloads[0].error is not None


@pytest.mark.asyncio
async def test_embedded_intrinsic_invocation_complete_fires_error(stub_backend):
    pytest.importorskip("cpex", reason="cpex not installed — install mellea[hooks]")

    class _RaisingResultProcessor:
        def __init__(self, *args, **kwargs):
            pass

        def transform(self, chunk, rewritten):
            raise ValueError("boom")

    payloads, raised = await _run_embedded_intrinsic_and_collect_invocation_payloads(
        stub_backend, _RaisingResultProcessor
    )

    assert isinstance(raised, ValueError)
    assert str(raised) == "boom"
    assert len(payloads) == 1
    assert payloads[0].outcome == "error"
    assert payloads[0].error is not None


@pytest.mark.asyncio
async def test_embedded_intrinsic_invocation_complete_fires_error_on_generation_failure(
    stub_backend,
):
    # A failure in the backend's own generation call never reaches
    # granite_formatters_processing — avalue() raises it straight off the
    # queue — so _await_embedded_generation must fire outcome="error" instead.
    pytest.importorskip("cpex", reason="cpex not installed — install mellea[hooks]")
    backend = _make_intrinsic_backend_stub(stub_backend)
    backend.processing = AsyncMock(return_value=None)
    adapter = _make_embedded_adapter_stub()
    backend._added_adapters = {adapter.qualified_name: adapter}

    def fake_transformers_inputs(request, tokenizer, model, ll_tokenizer=None):
        return {"input_tokens": object()}, {}

    def failing_generate_with_transformers(
        tokenizer, model, generate_input, other_input
    ):
        raise RuntimeError("simulated model error")

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
            side_effect=failing_generate_with_transformers,
        ),
        patch("mellea.backends.adapters._core.has_plugins", return_value=True),
        patch(
            "mellea.backends.adapters._core.invoke_hook", new_callable=AsyncMock
        ) as mock_invoke,
    ):
        output = await LocalHFBackend._generate_from_intrinsic(
            backend,
            Intrinsic("answerability"),
            ChatContext().add(Message("user", "Is the sky blue?")),
            model_options={},
        )
        assert output._gen.generate is not None
        with pytest.raises(RuntimeError, match="simulated model error"):
            await output.avalue()

    payloads = [
        call.args[1]
        for call in mock_invoke.call_args_list
        if call.args[0] is HookType.ADAPTER_FUNCTION_INVOCATION_COMPLETE
    ]
    assert len(payloads) == 1
    assert payloads[0].outcome == "error"
    assert isinstance(payloads[0].error, RuntimeError)


@pytest.mark.asyncio
async def test_legacy_peft_intrinsic_never_fires_embedded_invocation_complete(
    stub_backend,
):
    # Drives the legacy PEFT path (embedded_identity=None) and pins that it
    # never fires the embedded helper — guards the isinstance check that
    # gates embedded_identity from silently double-firing.
    pytest.importorskip("cpex", reason="cpex not installed — install mellea[hooks]")
    backend = _make_intrinsic_backend_stub(stub_backend)
    backend.processing = AsyncMock(return_value=None)
    adapter = _make_intrinsic_adapter_stub()
    backend._added_adapters = {adapter.qualified_name: adapter}

    def fake_transformers_inputs(request, tokenizer, model, ll_tokenizer=None):
        return {"input_tokens": object()}, {}

    def fake_generate_with_transformers(tokenizer, model, generate_input, other_input):
        return _FakeChatCompletionResponseWithContent('{"result": "ok"}')

    class _PassthroughResultProcessor:
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
            _PassthroughResultProcessor,
        ),
        patch(
            "mellea.formatters.granite.base.util.chat_completion_request_to_transformers_inputs",
            side_effect=fake_transformers_inputs,
        ),
        patch(
            "mellea.formatters.granite.base.util.generate_with_transformers",
            side_effect=fake_generate_with_transformers,
        ),
        patch("mellea.backends.adapters._core.has_plugins", return_value=True),
        patch(
            "mellea.backends.adapters._core.invoke_hook", new_callable=AsyncMock
        ) as mock_invoke,
    ):
        output = await LocalHFBackend._generate_from_intrinsic(
            backend,
            Intrinsic("answerability"),
            ChatContext().add(Message("user", "Is the sky blue?")),
            model_options={},
        )
        assert output._gen.generate is not None
        await output._gen.generate

        assert output._gen.process is not None
        while not output._gen.queue.empty():
            item = output._gen.queue.get_nowait()
            if item is not None:
                await output._gen.process(output, item)

    fired_hook_types = [call.args[0] for call in mock_invoke.call_args_list]
    assert HookType.ADAPTER_FUNCTION_INVOCATION_COMPLETE not in fired_hook_types


@pytest.mark.asyncio
async def test_composed_adapter_drives_generate_from_intrinsic(stub_backend):
    """A composed `Adapter` (not the `IntrinsicAdapter` shim) drives the full
    `_generate_from_intrinsic` path — name/config resolution, the composed
    `_generate_composed_local_file_with_adapter_scope` dispatch, and normal
    post-processing — end to end (Epic #929, issue #1144).
    """
    from mellea.backends.adapters._core import (
        Adapter as _AdapterCore,
        Identity,
        LocalFileBinding,
    )
    from mellea.backends.adapters.catalog import AdapterType
    from mellea.backends.adapters.io_contracts import get_io_contract

    backend = _make_intrinsic_backend_stub(stub_backend)
    backend.processing = AsyncMock(return_value=None)
    binding = LocalFileBinding(
        name="answerability",
        adapter_type=AdapterType.ALORA,
        repo_id="ibm-granite/granitelib-rag-r1.0",
        revision="abc123",
    )
    composed = _AdapterCore(
        identity=Identity(
            name="answerability", adapter_type="alora", capability="answerability"
        ),
        io_contract=get_io_contract("answerability"),
        weights=binding,
    )
    backend._added_adapters = {}
    backend._composed_adapters = {"answerability_alora": composed}
    backend.base_model_name = "granite-4.1-3b"

    def fake_transformers_inputs(request, tokenizer, model, ll_tokenizer=None):
        return {"input_tokens": object()}, {}

    def fake_generate_with_transformers(tokenizer, model, generate_input, other_input):
        return _FakeChatCompletionResponseWithContent('{"result": "ok"}')

    class _PassthroughResultProcessor:
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
            _PassthroughResultProcessor,
        ),
        patch(
            "mellea.formatters.granite.base.util.chat_completion_request_to_transformers_inputs",
            side_effect=fake_transformers_inputs,
        ),
        patch(
            "mellea.formatters.granite.base.util.generate_with_transformers",
            side_effect=fake_generate_with_transformers,
        ),
        patch(
            "mellea.formatters.granite.intrinsics.obtain_io_yaml",
            return_value="/fake/adapter.yaml",
        ),
        patch("builtins.open", mock_open(read_data="key: value")),
        patch("yaml.safe_load", return_value={"parameters": {}}),
    ):
        output = await LocalHFBackend._generate_from_intrinsic(
            backend,
            Intrinsic("answerability"),
            ChatContext().add(Message("user", "Is the sky blue?")),
            model_options={},
        )
        assert output._gen.generate is not None
        await output._gen.generate

        assert output._gen.process is not None
        processed = False
        while not output._gen.queue.empty():
            item = output._gen.queue.get_nowait()
            if item is not None:
                await output._gen.process(output, item)
                processed = True
        assert processed, "the composed local-file path must produce a response"


def test_composed_local_file_config_is_cached_after_first_derivation():
    """`_intrinsic_adapter_name_and_config` must not re-derive (re-download,
    re-parse) a composed LocalFile adapter's io.yaml on every call.

    Regression: the first version of this helper called
    `intrinsics.obtain_io_yaml` (a Hugging Face Hub round trip) and re-parsed
    the file on *every* invocation, unlike the shim it replaces (which loads
    once in `IntrinsicAdapter.__init__`). Every composed-LocalFile intrinsic
    call — the default path via `resolve_adapter` — paid that cost.
    """
    from mellea.backends.adapters._core import (
        Adapter as _AdapterCore,
        Identity,
        LocalFileBinding,
    )
    from mellea.backends.adapters.catalog import AdapterType
    from mellea.backends.adapters.io_contracts import get_io_contract

    backend = _make_backend()
    backend._composed_adapter_configs = {}
    binding = LocalFileBinding(
        name="answerability",
        adapter_type=AdapterType.ALORA,
        repo_id="ibm-granite/granitelib-rag-r1.0",
        revision="abc123",
    )
    composed = _AdapterCore(
        identity=Identity(
            name="answerability", adapter_type="alora", capability="answerability"
        ),
        io_contract=get_io_contract("answerability"),
        weights=binding,
    )

    with (
        patch(
            "mellea.formatters.granite.intrinsics.obtain_io_yaml",
            return_value="/fake/adapter.yaml",
        ) as mock_obtain,
        patch("builtins.open", mock_open(read_data="key: value")),
        patch("yaml.safe_load", return_value={"parameters": {}}),
    ):
        name1, config1 = backend._intrinsic_adapter_name_and_config(composed)
        name2, config2 = backend._intrinsic_adapter_name_and_config(composed)

    assert mock_obtain.call_count == 1, (
        "obtain_io_yaml must only run once; the second call should hit the cache"
    )
    assert name1 == name2 == "answerability"
    assert config1 is config2


def test_generate_embedded_with_generation_lock_deactivates_peft_state():
    """Embedded generation clears stale PEFT state before running the checkpoint."""
    backend = _make_backend()
    backend._model.active_adapters.return_value = []  # type: ignore[union-attr]

    with patch.object(backend, "deactivate_peft_adapter") as mock_deactivate:
        assert (
            backend._generate_embedded_with_generation_lock(lambda: "output")
            == "output"
        )

    mock_deactivate.assert_called_once_with("")


def test_add_embedded_adapter_rejects_mutated_weights_without_binding_backend():
    """A malformed shim must not retain this backend after registration fails."""
    backend = _make_backend()
    adapter = _make_embedded_adapter_stub()
    adapter.weights = ServerMediatedBinding()

    with pytest.raises(TypeError, match="must be an EmbeddedBinding"):
        backend.add_adapter(adapter)

    assert adapter.backend is None
    assert adapter.qualified_name not in backend._added_adapters


def test_load_peft_adapter_rejects_embedded_adapter():
    """Embedded adapters are selected by the chat template, not loaded by PEFT."""
    backend = _make_backend()
    adapter = _make_embedded_adapter_stub()
    backend.add_adapter(adapter)

    with pytest.raises(TypeError, match="through PEFT"):
        backend.load_peft_adapter(adapter.qualified_name)

    backend._model.load_adapter.assert_not_called()


def test_generate_with_adapter_lock_deactivates_and_calls_generate_func():
    """_generate_with_adapter_lock delegates deactivation and runs the model call.

    Standard (non-intrinsic) generation runs without adapters: the method
    delegates to `deactivate_peft_adapter("")` (rather than calling
    `_model.set_adapter` directly, Epic #929 Phase 2 / issue #1141), never
    touches the activation verbs or `load_peft_adapter`, and forwards to
    `generate_func` (its return value is the method's). Since #1465 routed
    intrinsic generation through `_generate_intrinsic_with_adapter_scope`, no
    production caller passes it an adapter to activate — which is why the
    method takes no adapter name at all. The method's deactivate-then-generate
    ordering is fixed by its body, not observable from these patched verbs.
    """
    backend = _make_backend()
    backend._model.active_adapters.return_value = []  # type: ignore[union-attr]

    with (
        patch.object(backend, "load_peft_adapter") as mock_load,
        patch.object(backend, "activate_peft_adapter") as mock_activate,
        patch.object(backend, "deactivate_peft_adapter") as mock_deactivate,
    ):
        out = backend._generate_with_adapter_lock(lambda: "output")

    assert out == "output"
    mock_deactivate.assert_called_once_with("")
    mock_activate.assert_not_called()
    mock_load.assert_not_called()


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


def test_generation_lock_is_reentrant():
    """`_generation_lock` must be a `threading.RLock`, not a plain `threading.Lock`.

    `_generate_intrinsic_with_adapter_scope` (issue #1465) holds `_generation_lock`
    for the whole prepare -> activate -> generate -> deactivate critical section
    on one thread, and the binding's verb calls (prepare/activate/deactivate)
    re-acquire the same lock (via `_adapter_activation_lock()`) from inside that
    section. A plain `Lock` can't tell that the second acquisition is
    same-thread and refuses it; only an `RLock` allows it.
    """
    backend = _make_backend()
    lock = backend._generation_lock

    assert lock.acquire(blocking=False)
    reentrant = lock.acquire(blocking=False)
    if reentrant:
        lock.release()
    lock.release()

    assert reentrant, "expected _generation_lock to be reentrant (threading.RLock)"


def test_generation_lock_reentrant_activation_does_not_deadlock():
    """Regression test for #1465's known lock-reentrancy deadlock.

    `activate()`/`deactivate()` (driven by `adapter_scope()`) acquire
    `_adapter_activation_lock()`, which is `_generation_lock`. Intrinsic
    generation holds `_generation_lock` for its whole critical section (see
    `_generate_intrinsic_with_adapter_scope`), so that inner acquisition happens
    on the same thread while the outer one is still held. Runs on a background
    thread with a bounded join so a regression to a non-reentrant lock fails
    fast instead of hanging the suite.
    """
    backend = _make_backend()
    completed = threading.Event()

    def nested_acquire():
        with backend._generation_lock:
            with backend._adapter_activation_lock():
                completed.set()

    t = threading.Thread(target=nested_acquire, daemon=True)
    t.start()
    t.join(timeout=5)

    assert completed.is_set(), (
        "re-acquiring _adapter_activation_lock() while already holding "
        "_generation_lock deadlocked; _generation_lock must be reentrant"
    )


def _make_fake_intrinsic_adapter(
    qualified_name: str, revision: str = "fake0000000000000000000000000000000000000"
) -> IntrinsicAdapter:
    """Builds a minimal `IntrinsicAdapter` stand-in, bypassing `__init__` (which
    downloads the adapter's `io.yaml`), exposing the attribute set
    `_generate_intrinsic_with_adapter_scope` reads — `.identity` (reused
    directly, not rebuilt, so this must already be the real `Identity`
    `IntrinsicAdapter.__init__` would have built), `.qualified_name`, and
    `.intrinsic_metadata.revision` — plus `.name`/`.adapter_type` for realism
    (the method reaches those via `identity`, not the stand-in's own fields).
    The stand-in pins `revision` to a catalogue SHA, mirroring the real
    `IntrinsicsCatalogEntry.revision` (a required, non-optional `str`) rather
    than allowing `None`.
    """
    name, _, adapter_type_str = qualified_name.rpartition("_")
    adapter_type = (
        AdapterType.ALORA if adapter_type_str == "alora" else AdapterType.LORA
    )
    adapter = IntrinsicAdapter.__new__(IntrinsicAdapter)
    adapter.name = name
    adapter.qualified_name = qualified_name
    adapter.adapter_type = adapter_type
    adapter.intrinsic_metadata = IntrinsicsCatalogEntry(
        name=name, repo_id="fake/repo", revision=revision
    )
    object.__setattr__(
        adapter,
        "identity",
        Identity(name=name, adapter_type=adapter_type.value, capability=name),
    )
    return adapter


def _register_fake_adapter(
    backend: LocalHFBackend, qualified_name: str, path: str
) -> None:
    """Registers a minimal `IntrinsicAdapter` stand-in under `_added_adapters`,
    satisfying what `load_peft_adapter` reads (`.path`, `.qualified_name`).
    """
    fake = IntrinsicAdapter.__new__(IntrinsicAdapter)
    fake.path = path
    fake.qualified_name = qualified_name
    backend._added_adapters[qualified_name] = fake


def _wire_fake_peft_model(backend: LocalHFBackend) -> None:
    """Makes `backend._model.set_adapter`/`.active_adapters` track real state.

    Without this, both are unconfigured `MagicMock`s: `set_adapter` records
    calls but doesn't affect what `active_adapters()` returns, so activation
    couldn't be observed from the generate callback.
    """
    active: list[str] = []

    def fake_set_adapter(name_or_names):
        if isinstance(name_or_names, list):
            active.clear()
        else:
            active[:] = [name_or_names]

    backend._model.set_adapter.side_effect = fake_set_adapter  # type: ignore[attr-defined]
    backend._model.active_adapters.side_effect = lambda: list(active)  # type: ignore[attr-defined]


def test_generate_composed_local_file_with_adapter_scope_activates_during_generation():
    """Composed-adapter counterpart of the test above, exercising the real
    `_generate_composed_local_file_with_adapter_scope` method directly.

    Coverage gap noted in review: the only other test reaching this method
    goes through `stub_backend`, whose fixture replaces it with a
    pass-through bypassing the lock hold, `adapter_scope` drive, and both
    `_assert_correct_adapters` calls entirely — so the method that owns that
    logic had no direct coverage of its own, unlike its shim sibling above.
    """
    from mellea.backends.adapters._core import (
        Adapter as _AdapterCore,
        LocalFileBinding as _LocalFileBinding,
    )
    from mellea.backends.adapters.io_contracts import get_io_contract

    backend = _make_backend()
    _wire_fake_peft_model(backend)

    binding = _LocalFileBinding(
        name="answerability",
        adapter_type=AdapterType.ALORA,
        repo_id="fake/repo",
        revision="fake0000000000000000000000000000000000000",
    )
    binding.backend = backend
    binding.path = "/fake/path"
    binding._loaded = True
    composed = _AdapterCore(
        identity=Identity(
            name="answerability", adapter_type="alora", capability="answerability"
        ),
        io_contract=get_io_contract("answerability"),
        weights=binding,
    )

    seen_during_generation = []

    def fake_generate():
        seen_during_generation.append(backend._model.active_adapters())
        return "output"

    out = backend._generate_composed_local_file_with_adapter_scope(
        composed, fake_generate
    )

    assert out == "output"
    assert seen_during_generation == [[binding.qualified_name]]
    assert backend._model.active_adapters() == []
    backend._model.set_adapter.assert_any_call([])  # type: ignore[attr-defined]


def test_generate_intrinsic_with_adapter_scope_activates_during_generation():
    """Generation demonstrably runs with the adapter active — asserted from
    inside the generate callback via the real (mocked) PEFT model state, not
    smoke-tested by checking generation merely succeeds.
    """
    backend = _make_backend()
    _wire_fake_peft_model(backend)
    adapter = _make_fake_intrinsic_adapter("answerability_alora")
    _register_fake_adapter(backend, adapter.qualified_name, "/fake/path")

    seen_during_generation = []

    def fake_generate():
        seen_during_generation.append(backend._model.active_adapters())
        return "output"

    out = backend._generate_intrinsic_with_adapter_scope(adapter, fake_generate)

    assert out == "output"
    assert seen_during_generation == [[adapter.qualified_name]]
    assert backend._model.active_adapters() == []
    # Asserts the real verb ran deactivation, not just that _wire_fake_peft_model's
    # `active_adapters()` mock still happens to read back empty.
    backend._model.set_adapter.assert_any_call([])  # type: ignore[attr-defined]


def test_generate_intrinsic_with_adapter_scope_fires_hooks_with_correct_payload():
    """The hooks `_generate_intrinsic_with_adapter_scope`'s docstring claims to
    enable must actually fire, with a payload that matches reality — not just
    smoke-tested by checking that *some* hooks fire.

    Regression coverage for two bugs caught in review: `revision` reported as
    `None` (mislabelled "unpinned") despite the adapter being pinned, and
    `binding_type` set to an invented `"intrinsic_legacy"` value instead of the
    `"local_file"` reality this binding actually is.
    """
    backend = _make_backend()
    _wire_fake_peft_model(backend)
    adapter = _make_fake_intrinsic_adapter(
        "answerability_alora", revision="deadbeef00000000000000000000000000000000"
    )
    _register_fake_adapter(backend, adapter.qualified_name, "/fake/path")

    with capture_adapter_hooks() as mock_invoke:
        out = backend._generate_intrinsic_with_adapter_scope(adapter, lambda: "output")

    assert out == "output"
    payloads = hook_payloads(mock_invoke)

    phases = [p.phase for p in payloads if hasattr(p, "phase")]
    assert phases == ["activate", "deactivate"]

    invocations = [p for p in payloads if hasattr(p, "outcome")]
    assert len(invocations) == 1
    invocation = invocations[0]
    assert invocation.outcome == "success"
    assert invocation.name == "answerability"
    assert invocation.adapter_type == "alora"
    assert invocation.binding_type == "local_file"
    assert invocation.revision == "deadbeef00000000000000000000000000000000"


def test_generate_intrinsic_with_adapter_scope_deactivates_on_error():
    """`adapter_scope()`'s deactivate-in-finally guarantee holds on the intrinsic path."""
    backend = _make_backend()
    _wire_fake_peft_model(backend)
    adapter = _make_fake_intrinsic_adapter("answerability_alora")
    _register_fake_adapter(backend, adapter.qualified_name, "/fake/path")

    def failing_generate():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        backend._generate_intrinsic_with_adapter_scope(adapter, failing_generate)

    assert backend._model.active_adapters() == []
    backend._model.set_adapter.assert_any_call([])  # type: ignore[attr-defined]


def test_generate_intrinsic_with_adapter_scope_reports_error_outcome_in_hook_payload():
    """A failed intrinsic generation reports `outcome="error"` with both phase events.

    The `AdapterMixin.adapter_scope` contract for a failing body is pinned
    generically (against a `LocalFileBinding`) in `test_adapter_scope.py`; this
    pins it with the intrinsic payload — `name`, `adapter_type`, `binding_type`
    and the pinned `revision` — so a regression can't silently mislabel
    intrinsic failures while the generic coverage keeps passing.
    """
    backend = _make_backend()
    _wire_fake_peft_model(backend)
    adapter = _make_fake_intrinsic_adapter(
        "answerability_alora", revision="deadbeef00000000000000000000000000000000"
    )
    _register_fake_adapter(backend, adapter.qualified_name, "/fake/path")

    def failing_generate():
        raise RuntimeError("boom")

    with capture_adapter_hooks() as mock_invoke:
        with pytest.raises(RuntimeError, match="boom"):
            backend._generate_intrinsic_with_adapter_scope(adapter, failing_generate)

    payloads = hook_payloads(mock_invoke)
    phases = [p.phase for p in payloads if hasattr(p, "phase")]
    assert phases == ["activate", "deactivate"]

    invocations = [p for p in payloads if hasattr(p, "outcome")]
    assert len(invocations) == 1
    invocation = invocations[0]
    assert invocation.outcome == "error"
    assert isinstance(invocation.error, RuntimeError)
    assert invocation.name == "answerability"
    assert invocation.adapter_type == "alora"
    assert invocation.binding_type == "local_file"
    assert invocation.revision == "deadbeef00000000000000000000000000000000"

    assert backend._model.active_adapters() == []


def test_generate_intrinsic_with_adapter_scope_fires_no_hooks_on_prepare_failure():
    """A `load_peft_adapter` failure happens before `adapter_scope()` is entered.

    So no adapter hooks fire at all, activation state is unchanged, and the
    next call still succeeds — a failed load must not poison later calls.
    """
    backend = _make_backend()
    _wire_fake_peft_model(backend)
    adapter = _make_fake_intrinsic_adapter("answerability_alora")
    _register_fake_adapter(backend, adapter.qualified_name, "/fake/path")

    with (
        capture_adapter_hooks() as mock_invoke,
        patch.object(
            backend, "load_peft_adapter", side_effect=RuntimeError("load failed")
        ),
    ):
        with pytest.raises(RuntimeError, match="load failed"):
            backend._generate_intrinsic_with_adapter_scope(adapter, lambda: "output")

    assert hook_payloads(mock_invoke) == []
    assert backend._model.active_adapters() == []

    out = backend._generate_intrinsic_with_adapter_scope(adapter, lambda: "output")
    assert out == "output"
    assert backend._model.active_adapters() == []


def test_concurrent_intrinsic_calls_cannot_observe_each_others_adapter():
    """Two concurrent intrinsic generate calls must never see each other's adapter active.

    Regression coverage for the intra-scope atomicity gap `adapter_scope()`'s
    docstring describes: if `_generation_lock` only guarded the verb calls
    themselves (rather than the whole prepare -> activate -> generate ->
    deactivate section its driver now holds, per #1465), one thread's
    `activate()` could interleave during another thread's generate call. The
    `time.sleep` below widens that window so a regression would be caught
    reliably rather than by luck.
    """
    backend = _make_backend()
    _wire_fake_peft_model(backend)
    _register_fake_adapter(backend, "answerability_lora", "/fake/a")
    _register_fake_adapter(backend, "uncertainty_lora", "/fake/b")

    mismatches: list[tuple[str, list[str]]] = []
    errors: list[BaseException] = []

    def run(qualified_name: str):
        adapter = _make_fake_intrinsic_adapter(qualified_name)

        def fake_generate():
            time.sleep(0.05)
            current = backend._model.active_adapters()
            if current != [qualified_name]:
                mismatches.append((qualified_name, current))
            return "ok"

        try:
            backend._generate_intrinsic_with_adapter_scope(adapter, fake_generate)
        except Exception as exc:  # surfaced via `errors`, not swallowed
            errors.append(exc)

    threads = [
        threading.Thread(target=run, args=("answerability_lora",)),
        threading.Thread(target=run, args=("uncertainty_lora",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert not any(t.is_alive() for t in threads)
    # A `Thread` swallows exceptions from its target by default, so without this
    # check a totally broken `_generate_intrinsic_with_adapter_scope` (e.g. an
    # AttributeError before `fake_generate` ever runs) would leave `mismatches`
    # empty and this test would pass vacuously.
    assert errors == []
    assert mismatches == []


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


def test_remove_adapter_removes_from_added_adapters():
    """remove_adapter() is the inverse of add_adapter() (#1528)."""
    backend = _make_backend()
    adapter = _make_intrinsic_adapter_stub()
    adapter.backend = None
    adapter.get_local_hf_path = lambda base_model_name: "/fake/path"
    backend.add_adapter(adapter)
    assert adapter.qualified_name in backend.list_adapters()

    backend.remove_adapter(adapter.qualified_name)

    assert adapter.qualified_name not in backend.list_adapters()
    assert adapter.qualified_name not in backend._added_adapters


def test_remove_adapter_unregistered_name_is_noop():
    """remove_adapter() on a name that was never added must not raise."""
    backend = _make_backend()
    backend.remove_adapter("never_registered_lora")  # must not raise


def test_remove_adapter_deregisters_composed_embedded_adapter():
    """remove_adapter() must deregister a composed Embedded adapter too.

    Regression: a composed Embedded adapter lives only in _composed_adapters
    (see add_adapter) — it never has a bare-binding entry in _added_adapters
    to mutate .backend/.path on. remove_adapter's original early return on an
    _added_adapters miss meant this case never reached the
    _composed_adapters/_composed_adapter_configs cleanup below it.
    """
    from mellea.backends.adapters._core import (
        Adapter as _AdapterCore,
        EmbeddedBinding,
        Identity,
    )
    from mellea.backends.adapters.io_contracts import get_io_contract

    backend = _make_backend()
    composed = _AdapterCore(
        identity=Identity(
            name="answerability", adapter_type="alora", capability="answerability"
        ),
        io_contract=get_io_contract("answerability"),
        weights=EmbeddedBinding(),
    )
    backend.add_adapter(composed)
    key = "answerability_alora"
    assert key in backend.list_adapters()
    backend._composed_adapter_configs[key] = {"parameters": {}}

    backend.remove_adapter(key)

    assert key not in backend.list_adapters()
    assert key not in backend._composed_adapters
    assert key not in backend._composed_adapter_configs


def test_load_peft_adapter_on_composed_embedded_adapter_raises_type_error():
    """load_peft_adapter() must reject a composed Embedded adapter with TypeError.

    Regression: a composed Embedded adapter lives only in _composed_adapters,
    never _added_adapters (see add_adapter's composed-Adapter branch). Without
    a check there, load_peft_adapter's `_added_adapters.get(...)` miss raised
    ValueError("was not previously added") instead of the documented
    TypeError explaining it's activated by the chat template, not PEFT —
    misleading, since the adapter genuinely was added.
    """
    from mellea.backends.adapters._core import (
        Adapter as _AdapterCore,
        EmbeddedBinding,
        Identity,
    )
    from mellea.backends.adapters.io_contracts import get_io_contract

    backend = _make_backend()
    composed = _AdapterCore(
        identity=Identity(
            name="answerability", adapter_type="alora", capability="answerability"
        ),
        io_contract=get_io_contract("answerability"),
        weights=EmbeddedBinding(),
    )
    backend.add_adapter(composed)

    with pytest.raises(TypeError, match="cannot load embedded adapter"):
        backend.load_peft_adapter("answerability_alora")


def test_add_adapter_allows_composed_wrapper_of_a_standalone_registered_binding():
    """A composed Adapter wrapping an already-standalone-registered LocalFileBinding
    must register into _composed_adapters, not be silently refused.

    Regression: the duplicate-key guard on the composed branch checked
    `key in self._added_adapters` unconditionally — a LocalFileBinding
    registered standalone (bare `add_adapter(binding)`) lands there under the
    same key, so a subsequent `add_adapter(composed_wrapping_that_binding)`
    hit the guard and was silently dropped: the binding's weights stayed
    loaded and functional, but the composed Adapter carrying `identity`/
    `io_contract` never made it into `_composed_adapters`, so
    `_find_adapter(capability)` could never find this capability.
    """
    from mellea.backends.adapters._core import (
        Adapter as _AdapterCore,
        LocalFileBinding as _LocalFileBinding,
    )
    from mellea.backends.adapters.io_contracts import get_io_contract

    backend = _make_backend()
    binding = _LocalFileBinding(
        name="answerability", adapter_type=AdapterType.ALORA, repo_id="fake/repo"
    )
    binding.get_local_hf_path = lambda base_model_name: "/fake/path"
    backend.add_adapter(binding)
    assert backend._added_adapters["answerability_alora"] is binding

    composed = _AdapterCore(
        identity=Identity(
            name="answerability", adapter_type="alora", capability="answerability"
        ),
        io_contract=get_io_contract("answerability"),
        weights=binding,
    )
    backend.add_adapter(composed)

    assert backend._composed_adapters["answerability_alora"] is composed
    assert backend._find_adapter("answerability") is composed


def test_add_adapter_shim_refuses_name_already_claimed_by_composed_adapter():
    """The legacy/shim add_adapter path must see composed-adapter registrations too.

    Regression: the duplicate-name guard on the shim path only checked
    `_added_adapters`, not `_composed_adapters` — a composed Adapter already
    registered under a qualified name did not stop a later shim
    `EmbeddedIntrinsicAdapter` (or bare `LocalFileBinding`) from silently
    claiming the same name, defeating the "adapter loading is not
    idempotent" invariant the warning message itself describes.
    """
    from mellea.backends.adapters._core import (
        Adapter as _AdapterCore,
        EmbeddedBinding as _EmbeddedBinding,
        Identity as _Identity,
    )
    from mellea.backends.adapters.io_contracts import get_io_contract

    backend = _make_backend()
    composed = _AdapterCore(
        identity=_Identity(
            name="answerability", adapter_type="alora", capability="answerability"
        ),
        io_contract=get_io_contract("answerability"),
        weights=_EmbeddedBinding(),
    )
    backend.add_adapter(composed)
    assert "answerability_alora" in backend._composed_adapters

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        shim = EmbeddedIntrinsicAdapter("answerability", config={}, technology="alora")

    backend.add_adapter(shim)

    assert shim.backend is None
    assert "answerability_alora" not in backend._added_adapters


def test_register_embedded_adapter_model_refused_duplicate_does_not_clobber_cached_config():
    """register_embedded_adapter_model() must not overwrite a live adapter's cached
    config, or falsely report it as (re-)registered, when add_adapter() refuses
    a duplicate name.

    Regression: add_adapter() silently refuses (logs a warning, returns) rather
    than raising for an already-registered qualified name. The loop here used to
    write `discovered`'s config into `_composed_adapter_configs` and append the
    name unconditionally, so a second discovery call for the same adapter name
    would overwrite the first, still-registered adapter's config with whatever
    the second (possibly different) discovery produced, and falsely list it as
    registered in the returned names.
    """
    from mellea.backends.adapters._core import (
        Adapter as _AdapterCore,
        EmbeddedBinding as _EmbeddedBinding,
        Identity as _Identity,
    )
    from mellea.backends.adapters.io_contracts import get_io_contract

    backend = _make_backend()

    def _make_composed():
        return _AdapterCore(
            identity=_Identity(
                name="answerability", adapter_type="alora", capability="answerability"
            ),
            io_contract=get_io_contract("answerability"),
            weights=_EmbeddedBinding(),
        )

    first_config = {"version": "first"}
    with patch(
        "mellea.backends.huggingface._discover_embedded_adapters",
        return_value=[(_make_composed(), first_config)],
    ):
        names = backend.register_embedded_adapter_model(
            "some/repo", intrinsic_name="answerability"
        )
    assert names == ["answerability"]
    assert backend._composed_adapter_configs["answerability_alora"] is first_config

    second_config = {"version": "second"}
    with patch(
        "mellea.backends.huggingface._discover_embedded_adapters",
        return_value=[(_make_composed(), second_config)],
    ):
        names = backend.register_embedded_adapter_model(
            "some/other-repo", intrinsic_name="answerability"
        )

    assert names == []
    assert backend._composed_adapter_configs["answerability_alora"] is first_config


def test_remove_adapter_deregisters_composed_local_file_adapter():
    """remove_adapter() must clean up a composed LocalFile adapter's entries too.

    Coverage gap noted in review: only the composed-Embedded path (above) had
    a regression test for this cleanup; the composed-LocalFile branch at the
    same call site (which additionally clears the bare binding's
    `.backend`/`.path`) had none.
    """
    from mellea.backends.adapters._core import (
        Adapter as _AdapterCore,
        LocalFileBinding as _LocalFileBinding,
    )
    from mellea.backends.adapters.io_contracts import get_io_contract

    backend = _make_backend()
    binding = _LocalFileBinding(
        name="answerability",
        adapter_type=AdapterType.ALORA,
        repo_id="fake/repo",
        revision="fake0000000000000000000000000000000000000",
    )
    binding.backend = backend
    binding.path = "/fake/path"
    binding._loaded = True
    composed = _AdapterCore(
        identity=Identity(
            name="answerability", adapter_type="alora", capability="answerability"
        ),
        io_contract=get_io_contract("answerability"),
        weights=binding,
    )
    key = "answerability_alora"
    # Simulates the post-add_adapter() state directly, bypassing the real
    # PEFT-loading side effects add_adapter() would otherwise trigger.
    backend._added_adapters[key] = binding
    backend._composed_adapters[key] = composed
    backend._composed_adapter_configs[key] = {"parameters": {}}

    backend.remove_adapter(key)

    assert key not in backend.list_adapters()
    assert key not in backend._added_adapters
    assert key not in backend._composed_adapters
    assert key not in backend._composed_adapter_configs
    assert binding.backend is None
    assert binding.path is None


def test_intrinsic_adapter_name_and_config_raises_for_uncached_composed_embedded():
    """`_intrinsic_adapter_name_and_config` must raise its documented ValueError
    when a composed Embedded adapter has no cached config.

    Coverage gap noted in review: the happy path (config present) was
    tested; this error branch — reachable if a composed Embedded adapter was
    registered via bare `add_adapter()` rather than
    `register_embedded_adapter_model()`/`resolve_adapter()`, which are the
    only call sites that populate `_composed_adapter_configs` for it — was not.
    """
    from mellea.backends.adapters._core import (
        Adapter as _AdapterCore,
        EmbeddedBinding as _EmbeddedBinding,
    )
    from mellea.backends.adapters.io_contracts import get_io_contract

    backend = _make_backend()
    composed = _AdapterCore(
        identity=Identity(
            name="answerability", adapter_type="alora", capability="answerability"
        ),
        io_contract=get_io_contract("answerability"),
        weights=_EmbeddedBinding(),
    )
    backend.add_adapter(composed)
    assert "answerability_alora" in backend._composed_adapters
    assert "answerability_alora" not in backend._composed_adapter_configs

    with pytest.raises(ValueError, match=r"No io\.yaml config cached"):
        backend._intrinsic_adapter_name_and_config(composed)


def test_add_adapter_after_remove_adapter_allows_a_fresh_registration():
    """#1528: removing an adapter frees its qualified_name for a different
    adapter object to register under — the name is no longer burned for the
    backend's lifetime.
    """
    backend = _make_backend()
    first = _make_intrinsic_adapter_stub()
    first.backend = None
    first.get_local_hf_path = lambda base_model_name: "/fake/path"
    backend.add_adapter(first)
    backend.remove_adapter(first.qualified_name)

    second = _make_intrinsic_adapter_stub()
    second.backend = None
    second.get_local_hf_path = lambda base_model_name: "/fake/path-2"
    backend.add_adapter(second)

    assert second.backend is backend
    assert backend._added_adapters[second.qualified_name] is second


def test_remove_adapter_clears_backend_and_path_references():
    """remove_adapter() must reverse ALL of add_adapter()'s mutations, not just
    the registry entry.

    Regression guard: `add_adapter()` sets `.path` and `.backend = self` in
    addition to inserting into `_added_adapters`. A `remove_adapter()` that
    only pops the dict entry leaves the removed object's `.backend` pointing at a
    backend that no longer knows about it — bricking the object for
    re-registration anywhere (see the next test).
    """
    backend = _make_backend()
    adapter = _make_intrinsic_adapter_stub()
    adapter.backend = None
    adapter.get_local_hf_path = lambda base_model_name: "/fake/path"
    backend.add_adapter(adapter)
    assert adapter.backend is backend
    assert adapter.path == "/fake/path"

    backend.remove_adapter(adapter.qualified_name)

    assert adapter.backend is None
    assert adapter.path is None


def test_add_adapter_after_remove_adapter_allows_reregistering_the_same_object():
    """A removed adapter object, not just a fresh one, must be re-addable.

    Before `remove_adapter()` cleared `.backend`, re-adding the *same* object
    hit the `adapter.backend is self` early-return in `add_adapter()` — a
    silent no-op, never re-registered, with no exception raised.
    """
    backend = _make_backend()
    adapter = _make_intrinsic_adapter_stub()
    adapter.backend = None
    adapter.get_local_hf_path = lambda base_model_name: "/fake/path"
    backend.add_adapter(adapter)
    backend.remove_adapter(adapter.qualified_name)

    backend.add_adapter(adapter)

    assert adapter.backend is backend
    assert backend._added_adapters[adapter.qualified_name] is adapter


def test_remove_adapter_raises_if_still_loaded():
    """remove_adapter() must refuse to free a name that is still loaded.

    `load_peft_adapter()` deliberately swallows PEFT's "Adapter with name X
    already exists." — safe only because a qualified_name, once claimed,
    could never be reclaimed. Freeing
    the name while it is still loaded lets a later `load_peft_adapter()` call
    for a *different* adapter object hit that swallow and silently keep
    running on the old weights. `unload_peft_adapter()` (which `release()`
    always calls first) must clear `_loaded_adapters` before `remove_adapter()`
    can succeed.
    """
    backend = _make_backend()
    adapter = _make_intrinsic_adapter_stub()
    adapter.backend = None
    adapter.get_local_hf_path = lambda base_model_name: "/fake/path"
    backend.add_adapter(adapter)
    backend.load_peft_adapter(adapter.qualified_name)
    assert adapter.qualified_name in backend._loaded_adapters

    with pytest.raises(ValueError, match="still loaded"):
        backend.remove_adapter(adapter.qualified_name)

    assert adapter.qualified_name in backend._added_adapters

    backend.unload_peft_adapter(adapter.qualified_name)
    backend.remove_adapter(adapter.qualified_name)  # now succeeds

    assert adapter.qualified_name not in backend._added_adapters


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
        assert rewritten["temperature"] == 0.0
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


@pytest.mark.asyncio
async def test_intrinsic_closure_cell_and_kv_cache_released_after_post_processing(
    stub_backend,
):
    """Holding only the MOT after post_processing must not pin intrinsic HF output.

    Two related retention paths are exercised:

    1. `raw_hf_output_cell` — the closure captured by `_gen.process` (a
       `functools.partial` that outlives the call). The cell must be cleared
       after its value is transferred to `mot.raw.response`, otherwise the
       held MOT retains the full GenerateDecoderOnlyOutput.

    2. `past_key_values` — the KV cache inside the HF output. On the no-cache
       path, post_processing must remove it from the GenerateDecoderOnlyOutput
       before clearing `mot.raw.response`.

    The test deliberately retains the MOT while dropping all independent test
    references to the HF output, DynamicCache, and KV tensors.
    """
    backend = _make_intrinsic_backend_stub(stub_backend)
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

    # Build a small KV cache with real tensors so weakrefs can verify that the
    # cache and its allocations are released.
    kv_cache = DynamicCache()
    kv_cache.update(
        key_states=torch.zeros(1, 1, 1, 4),
        value_states=torch.zeros(1, 1, 1, 4),
        layer_idx=0,
    )

    fake_scores = (torch.zeros(1, 32000),)
    fake_hf_output = GenerateDecoderOnlyOutput(
        sequences=torch.tensor([[1, 2]]),
        scores=fake_scores,
        logits=None,
        attentions=None,
        hidden_states=None,
        past_key_values=kv_cache,
    )

    # Take weakrefs before dropping all direct strong references.
    ref_container = weakref.ref(fake_hf_output)
    ref_kv_cache = weakref.ref(kv_cache)
    ref_kv_tensors = [
        weakref.ref(t)
        for layer in kv_cache.layers
        if isinstance(layer, CacheLayerMixin)
        for t in (layer.keys, layer.values)
        if t is not None
    ]

    adapter = _make_intrinsic_adapter_stub()
    backend._added_adapters = {adapter.qualified_name: adapter}

    class _FakeChatCompletionResponse:
        class _Choice:
            class _Message:
                content = "0.9"

            message = _Message()

        choices = [_Choice()]

    class _FakeResultProcessorPassthrough:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def transform(self, chunk: Any, rewritten: Any) -> Any:
            return chunk

    def fake_transformers_inputs(
        rewritten: Any, tokenizer: Any, model: Any, ll_tokenizer: Any = None
    ) -> tuple[dict, dict]:
        return {"input_tokens": torch.tensor([[1]])}, {}

    def fake_generate_with_transformers(
        tokenizer: Any, model: Any, generate_input: Any, other_input: Any
    ) -> Any:
        model.generate(inputs=generate_input["input_tokens"])
        return _FakeChatCompletionResponse()

    mock_model = MagicMock()
    mock_model.generate = MagicMock(return_value=fake_hf_output)
    backend._model = mock_model

    with (
        patch(
            "mellea.backends.huggingface.granite_formatters.IntrinsicsRewriter",
            _FakeRewriter,
        ),
        patch(
            "mellea.backends.huggingface.granite_formatters.IntrinsicsResultProcessor",
            _FakeResultProcessorPassthrough,
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
            model_options={ModelOption.LOGITS: True},
        )

        assert output._gen.generate is not None
        await output._gen.generate

        while not output._gen.queue.empty():
            item = output._gen.queue.get_nowait()
            if item is not None:
                await output._gen.process(output, item)

        output._computed = True

    # MagicMock artificially retains its return_value. A real HF model does not
    # retain the result of generate(), so remove this test-only retention path.
    mock_model.generate.return_value = None

    # Drop the test's direct strong references. From here onward, the retained
    # MOT should be the only object graph capable of keeping the HF output alive.
    del fake_hf_output
    del kv_cache
    del fake_scores

    await backend.post_processing(
        output, [], None, False, {}, None, torch.tensor([[1]])
    )

    gc.collect()
    gc.collect()

    assert output.raw.response is None, (
        "raw.response should be None on the no-caching path"
    )

    assert ref_container() is None, (
        "GenerateDecoderOnlyOutput is still alive while the MOT is held; "
        "the _gen.process closure or another MOT-owned path is pinning it"
    )

    assert ref_kv_cache() is None, (
        "DynamicCache is still alive while the MOT is held; "
        "past_key_values was not fully released on the no-cache path"
    )

    for i, ref in enumerate(ref_kv_tensors):
        assert ref() is None, (
            f"KV-cache tensor {i} is still alive while the MOT is held; "
            "past_key_values or another reference path is retaining it"
        )


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


def _make_raw_fake_setup(
    batch_size: int, vocab_size: int, n_tokens: int, prompt_len: int
):
    """Return (backend, fake_encoding, fake_input_ids) for generate_from_raw tests."""
    backend = _make_backend()
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
    decode_values = [f"result_{chr(ord('a') + i)}" for i in range(batch_size)]
    backend._tokenizer.batch_decode = MagicMock(return_value=decode_values)
    return backend, fake_encoding, fake_input_ids


@pytest.mark.asyncio
async def test_generate_from_raw_raw_response_set_per_mot():
    """Every MOT from generate_from_raw has raw.response set to a GenerateDecoderOnlyOutput.

    Asserts:
    - raw.response is not None for each MOT.
    - raw.response.sequences.shape == (1, full_seq_len).
    - raw.response.sequences must be a clone with distinct storage
    - raw.response.past_key_values is None.
    - raw.response.attentions is None.
    - raw.response.hidden_states is None.
    """
    batch_size = 2
    vocab_size = 32000
    n_tokens = 3
    prompt_len = 1
    full_seq_len = prompt_len + n_tokens

    backend, _fake_encoding, _fake_input_ids = _make_raw_fake_setup(
        batch_size, vocab_size, n_tokens, prompt_len
    )
    sequences = torch.zeros(batch_size, full_seq_len, dtype=torch.long)
    fake_outputs = GenerateDecoderOnlyOutput(
        sequences=sequences,
        scores=None,
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
            actions, MagicMock(), model_options={}
        )

    assert len(results) == batch_size
    for item_idx, result in enumerate(results):
        assert result.raw.response is not None, (
            f"item {item_idx}: raw.response must be set"
        )
        assert isinstance(result.raw.response, GenerateDecoderOnlyOutput), (
            f"item {item_idx}: raw.response must be GenerateDecoderOnlyOutput"
        )
        assert result.raw.response.sequences.shape == (1, full_seq_len), (
            f"item {item_idx}: sequences shape must be (1, {full_seq_len})"
        )
        # Clone - must NOT share storage with the original batch tensor.
        assert (
            result.raw.response.sequences.untyped_storage().data_ptr()
            != sequences.untyped_storage().data_ptr()
        ), f"item {item_idx}: sequences must be a clone, not a view"
        assert result.raw.response.past_key_values is None, (
            f"item {item_idx}: past_key_values must be None"
        )
        assert result.raw.response.attentions is None, (
            f"item {item_idx}: attentions must be None"
        )
        assert result.raw.response.hidden_states is None, (
            f"item {item_idx}: hidden_states must be None"
        )


@pytest.mark.asyncio
async def test_generate_from_raw_raw_response_scores_are_clones_when_logits_requested():
    """raw.response.scores is a tuple of clones when ModelOption.LOGITS is set.

    Each tensor in raw.response.scores must own compact per-row storage and must
    not share storage with the corresponding batch step tensor — consistent with
    generation.logits which also holds clones.
    """
    batch_size = 2
    vocab_size = 32000
    n_tokens = 3
    prompt_len = 1
    full_seq_len = prompt_len + n_tokens

    backend, _fake_encoding, _fake_input_ids = _make_raw_fake_setup(
        batch_size, vocab_size, n_tokens, prompt_len
    )
    sequences = torch.zeros(batch_size, full_seq_len, dtype=torch.long)
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

    for item_idx, result in enumerate(results):
        assert result.raw.response.scores is not None, (
            f"item {item_idx}: raw.response.scores must be set when LOGITS=True"
        )
        assert len(result.raw.response.scores) == n_tokens, (
            f"item {item_idx}: one scores tensor per generation step"
        )
        for tok_idx, t in enumerate(result.raw.response.scores):
            assert t.shape == (1, vocab_size), (
                f"item {item_idx} token {tok_idx}: shape must be (1, vocab_size)"
            )
            # Clone - must NOT share storage with the original batch step tensor.
            assert (
                t.untyped_storage().data_ptr()
                != fake_scores[tok_idx].untyped_storage().data_ptr()
            ), f"item {item_idx} token {tok_idx}: raw.response.scores must be a clone"


@pytest.mark.asyncio
async def test_generate_from_raw_raw_response_scores_none_when_logits_not_requested():
    """raw.response.scores is None when ModelOption.LOGITS is not set."""
    batch_size = 1
    vocab_size = 32000
    n_tokens = 2
    prompt_len = 1
    full_seq_len = prompt_len + n_tokens

    backend, _fake_encoding, _fake_input_ids = _make_raw_fake_setup(
        batch_size, vocab_size, n_tokens, prompt_len
    )
    # When LOGITS is not set, model.generate() is called without output_scores=True,
    # so outputs.scores will be None — simulate that here.
    sequences = torch.zeros(batch_size, full_seq_len, dtype=torch.long)
    fake_outputs = GenerateDecoderOnlyOutput(
        sequences=sequences,
        scores=None,
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

    assert results[0].raw.response.scores is None, (
        "raw.response.scores must be None when model.generate() returns no scores"
    )


@pytest.mark.asyncio
async def test_generate_from_raw_raw_response_none_for_non_tensor_sequences():
    """raw.response stays None when cached raw output sequences are not a tensor."""
    batch_size = 2
    vocab_size = 32000
    n_tokens = 3
    prompt_len = 1
    full_seq_len = prompt_len + n_tokens

    backend, _fake_encoding, _fake_input_ids = _make_raw_fake_setup(
        batch_size, vocab_size, n_tokens, prompt_len
    )
    backend._use_caches = True
    fake_outputs = cast(
        Any,
        SimpleNamespace(
            sequences=[[0] * full_seq_len for _ in range(batch_size)],
            scores=None,
            logits=None,
            attentions=None,
            hidden_states=None,
            past_key_values=None,
        ),
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
            cast(Any, actions), MagicMock(), model_options={}
        )

    assert len(results) == batch_size
    assert "raw_batch_non_tensor_sequences_unsupported" in backend._warned_about
    for item_idx, result in enumerate(results):
        assert result.raw.response is None, (
            f"item {item_idx}: raw.response must stay None for non-tensor sequences"
        )


@pytest.mark.asyncio
async def test_generate_from_raw_raw_response_none_for_beam_outputs():
    """raw.response stays None for cached beam-search outputs."""
    batch_size = 2
    vocab_size = 32000
    n_tokens = 3
    prompt_len = 1
    full_seq_len = prompt_len + n_tokens

    backend, _fake_encoding, _fake_input_ids = _make_raw_fake_setup(
        batch_size, vocab_size, n_tokens, prompt_len
    )
    backend._use_caches = True
    fake_outputs = GenerateBeamDecoderOnlyOutput(
        sequences=torch.zeros(batch_size, full_seq_len, dtype=torch.long),
        sequences_scores=None,
        scores=None,
        logits=None,
        beam_indices=torch.zeros(batch_size, full_seq_len, dtype=torch.long),
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
            cast(Any, actions), MagicMock(), model_options={}
        )

    assert len(results) == batch_size
    assert "raw_batch_beam_search_unsupported" in backend._warned_about
    for item_idx, result in enumerate(results):
        assert result.raw.response is None, (
            f"item {item_idx}: raw.response must stay None for beam-search outputs"
        )
