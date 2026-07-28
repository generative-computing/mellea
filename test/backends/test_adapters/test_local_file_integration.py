# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test: LocalFileBinding's full lifecycle through a real LocalHFBackend.

Real OTel span capture and a real `LocalHFBackend` instance are used; only the
Hugging Face download (`intrinsics.obtain_lora`) and the underlying PEFT model
(`_model`) are mocked, per test/README.md's definition of `integration` — real
framework/library objects wired together, external network and model weights
mocked at the outer boundary.
"""

from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch", reason="torch not installed — install mellea[hf]")
pytest.importorskip(
    "transformers", reason="transformers not installed — install mellea[hf]"
)
pytest.importorskip(
    "llguidance", reason="llguidance not installed — install mellea[hf]"
)

from mellea.backends.adapters._core import (
    Adapter,
    Identity,
    IOContract,
    LocalFileBinding,
)
from mellea.backends.adapters.catalog import fetch_intrinsic_metadata
from mellea.backends.huggingface import LocalHFBackend
from mellea.core import Component
from test.telemetry.conftest import reset_tracing_state

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not installed"),
]


class _Contract(IOContract):
    def build_prompt(self, **kwargs: object) -> Component:
        raise NotImplementedError

    def parse(self, raw: str) -> dict[str, object]:
        return {}


@pytest.fixture(scope="module", autouse=True)
def setup_telemetry():
    mp = pytest.MonkeyPatch()
    mp.setenv("MELLEA_TRACES_ENABLED", "true")
    reset_tracing_state()

    yield

    mp.undo()
    reset_tracing_state()


@pytest.fixture
def span_exporter():
    from mellea.telemetry import tracing

    tracing.get_application_tracer()
    provider = tracing._tracer_provider
    if provider is None:
        pytest.skip("Telemetry not initialized")

    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    yield exporter

    exporter.clear()


def _make_backend() -> LocalHFBackend:
    mock_tok = MagicMock(eos_token_id=0, vocab_size=32000)
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


def _make_binding() -> LocalFileBinding:
    metadata = fetch_intrinsic_metadata("answerability")
    return LocalFileBinding(
        name="answerability",
        adapter_type=metadata.adapter_types[0],
        repo_id=metadata.repo_id,
        revision=metadata.revision,
    )


def _make_adapter(binding: LocalFileBinding) -> Adapter:
    identity = Identity(
        name="answerability", adapter_type="alora", capability="answerability"
    )
    return Adapter(identity=identity, io_contract=_Contract(), weights=binding)


def test_prepare_activate_deactivate_release_full_lifecycle(span_exporter):
    backend = _make_backend()
    binding = _make_binding()
    adapter = _make_adapter(binding)

    with patch(
        "mellea.formatters.granite.intrinsics.obtain_lora",
        return_value="/fake/local/adapter/path",
    ) as mock_obtain_lora:
        binding.bind_backend(backend)
        binding.prepare()

        assert binding.backend is backend
        assert binding.qualified_name in backend.list_adapters()
        mock_obtain_lora.assert_called_once()
        assert mock_obtain_lora.call_args.kwargs["revision"] == binding.revision

        with backend.adapter_scope(adapter):
            backend._model.set_adapter.assert_called_with(binding.qualified_name)  # type: ignore[union-attr]

        backend._model.set_adapter.assert_called_with([])  # type: ignore[union-attr]

        binding.release()

    backend._model.delete_adapter.assert_called_once_with(binding.qualified_name)  # type: ignore[union-attr]
    assert binding.backend is None

    trace.get_tracer_provider().force_flush()
    span_names = [s.name for s in span_exporter.get_finished_spans()]
    assert "adapter_function" in span_names
    assert "adapter_function.activate" in span_names
    assert "adapter_function.deactivate" in span_names


def test_deactivate_runs_even_when_generation_body_raises(span_exporter):
    backend = _make_backend()
    binding = _make_binding()
    adapter = _make_adapter(binding)

    with patch(
        "mellea.formatters.granite.intrinsics.obtain_lora",
        return_value="/fake/local/adapter/path",
    ):
        binding.bind_backend(backend)
        binding.prepare()

        with pytest.raises(RuntimeError, match="generation failed"):
            with backend.adapter_scope(adapter):
                raise RuntimeError("generation failed")

        backend._model.set_adapter.assert_called_with([])  # type: ignore[union-attr]
        binding.release()

    trace.get_tracer_provider().force_flush()
    spans = {s.name: s for s in span_exporter.get_finished_spans()}
    assert spans["adapter_function"].status.status_code == trace.StatusCode.ERROR


if __name__ == "__main__":
    pytest.main([__file__])
