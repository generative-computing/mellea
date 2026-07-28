# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the adapter-function span helpers (Epic #929 Phase 2, issue #1141).

See docs/dev/adapter_observability.md for the span/attribute schema.
"""

import pytest

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

pytestmark = pytest.mark.skipif(
    not OTEL_AVAILABLE, reason="OpenTelemetry not installed"
)


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


@pytest.mark.integration
def test_adapter_function_span_success_records_attributes(span_exporter):
    from mellea.telemetry.tracing import (
        finish_adapter_function_span_success,
        start_adapter_function_span,
    )

    start_adapter_function_span(
        "call-1",
        name="answerability",
        revision="abc123",
        binding_type="local_file",
        adapter_type="lora",
    )
    finish_adapter_function_span_success("call-1", outcome="success")

    trace.get_tracer_provider().force_flush()
    spans = span_exporter.get_finished_spans()
    span = next(s for s in spans if s.name == "adapter_function")

    assert span.attributes["mellea.adapter_function.name"] == "answerability"
    assert span.attributes["mellea.adapter_function.revision"] == "abc123"
    assert span.attributes["mellea.adapter_function.binding_type"] == "local_file"
    assert span.attributes["mellea.adapter_function.adapter_type"] == "lora"
    assert span.attributes["mellea.adapter_function.outcome"] == "success"
    assert span.status.status_code == trace.StatusCode.UNSET


@pytest.mark.integration
def test_adapter_function_span_error_records_exception(span_exporter):
    from mellea.telemetry.tracing import (
        finish_adapter_function_span_error,
        start_adapter_function_span,
    )

    start_adapter_function_span(
        "call-2",
        name="answerability",
        revision=None,
        binding_type="local_file",
        adapter_type="lora",
    )
    exc = RuntimeError("boom")
    finish_adapter_function_span_error("call-2", outcome="error", exception=exc)

    trace.get_tracer_provider().force_flush()
    spans = span_exporter.get_finished_spans()
    span = next(s for s in spans if s.name == "adapter_function")

    assert span.attributes["mellea.adapter_function.outcome"] == "error"
    assert span.status.status_code == trace.StatusCode.ERROR
    assert span.attributes.get("mellea.adapter_function.revision") is None


@pytest.mark.integration
def test_adapter_function_phase_span_success(span_exporter):
    from mellea.telemetry.tracing import (
        finish_adapter_function_phase_span,
        start_adapter_function_phase_span,
    )

    start_adapter_function_phase_span("call-3", "activate")
    finish_adapter_function_phase_span("call-3", "activate")

    trace.get_tracer_provider().force_flush()
    spans = span_exporter.get_finished_spans()
    span = next(s for s in spans if s.name == "adapter_function.activate")

    assert span.attributes["mellea.adapter_function.phase"] == "activate"
    assert span.status.status_code == trace.StatusCode.UNSET


@pytest.mark.integration
def test_adapter_function_phase_span_error(span_exporter):
    from mellea.telemetry.tracing import (
        finish_adapter_function_phase_span,
        start_adapter_function_phase_span,
    )

    start_adapter_function_phase_span("call-4", "deactivate")
    finish_adapter_function_phase_span(
        "call-4", "deactivate", exception=ValueError("nope")
    )

    trace.get_tracer_provider().force_flush()
    spans = span_exporter.get_finished_spans()
    span = next(s for s in spans if s.name == "adapter_function.deactivate")

    assert span.status.status_code == trace.StatusCode.ERROR


@pytest.mark.integration
def test_parent_and_phase_spans_use_independent_keys(span_exporter):
    """A phase name matching the call_id's own key must not collide with the parent span."""
    from mellea.telemetry.tracing import (
        finish_adapter_function_phase_span,
        finish_adapter_function_span_success,
        start_adapter_function_phase_span,
        start_adapter_function_span,
    )

    start_adapter_function_span(
        "call-5",
        name="answerability",
        revision=None,
        binding_type="local_file",
        adapter_type="lora",
    )
    start_adapter_function_phase_span("call-5", "activate")
    finish_adapter_function_phase_span("call-5", "activate")
    finish_adapter_function_span_success("call-5", outcome="success")

    trace.get_tracer_provider().force_flush()
    spans = span_exporter.get_finished_spans()
    names = [s.name for s in spans if s.name.startswith("adapter_function")]
    assert names.count("adapter_function") == 1
    assert names.count("adapter_function.activate") == 1


if __name__ == "__main__":
    pytest.main([__file__])
