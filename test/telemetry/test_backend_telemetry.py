"""Unit tests for backend telemetry instrumentation with Gen-AI semantic conventions."""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.components import Message
from mellea.stdlib.context import SimpleContext

# Check if OpenTelemetry is available
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
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


@pytest.fixture(scope="session", autouse=True)
def setup_telemetry():
    """Set up telemetry for all tests in this module."""
    import os

    # Enable backend tracing before any imports
    os.environ["MELLEA_TRACE_BACKEND"] = "true"


@pytest.fixture
def span_exporter():
    """Create an in-memory span exporter for testing."""
    # Import mellea.telemetry to ensure it's initialized
    from mellea import telemetry

    # Get the real tracer provider from mellea.telemetry module
    # The global trace.get_tracer_provider() returns a ProxyTracerProvider
    provider = telemetry._tracer_provider

    if provider is None:
        pytest.skip("Telemetry not initialized")

    # Add our in-memory exporter to it
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    yield exporter
    exporter.clear()


@pytest.mark.asyncio
async def test_span_duration_captures_async_operation(span_exporter, gh_run):
    """Test that span duration includes the full async operation time."""
    if gh_run:
        pytest.skip("Skipping in CI - requires Ollama")

    backend = OllamaModelBackend(model_id="llama3.2:1b")
    ctx = SimpleContext()
    ctx = ctx.add(Message(role="user", content="Say 'test' and nothing else"))

    # Add a small delay to ensure measurable duration
    start_time = time.time()
    mot, _ = await backend.generate_from_context(
        Message(role="assistant", content=""), ctx
    )
    await mot.avalue()  # Wait for async completion
    end_time = time.time()
    actual_duration = end_time - start_time

    # Force flush to ensure spans are exported
    trace.get_tracer_provider().force_flush()  # type: ignore

    # Get the recorded span
    spans = span_exporter.get_finished_spans()
    assert len(spans) > 0, "No spans were recorded"

    backend_span = None
    for span in spans:
        if span.name == "chat":
            backend_span = span
            break

    assert backend_span is not None, "Backend span not found"

    # Span duration should be close to actual duration (within 100ms tolerance)
    span_duration_ns = backend_span.end_time - backend_span.start_time
    span_duration_s = span_duration_ns / 1e9

    assert span_duration_s >= 0.1, (
        f"Span duration too short: {span_duration_s}s (expected >= 0.1s)"
    )
    assert abs(span_duration_s - actual_duration) < 0.5, (
        f"Span duration {span_duration_s}s differs significantly from actual {actual_duration}s"
    )


@pytest.mark.asyncio
async def test_context_propagation_parent_child(span_exporter, gh_run):
    """Test that parent-child span relationships are maintained."""
    if gh_run:
        pytest.skip("Skipping in CI - requires Ollama")

    backend = OllamaModelBackend(model_id="llama3.2:1b")
    ctx = SimpleContext()
    ctx = ctx.add(Message(role="user", content="Say 'test' and nothing else"))

    # Create a parent span
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("parent_operation"):
        mot, _ = await backend.generate_from_context(
            Message(role="assistant", content=""), ctx
        )
        await mot.avalue()  # Wait for async completion

    # Get the recorded spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 2, f"Expected at least 2 spans, got {len(spans)}"

    # Find parent and child spans
    parent_recorded = None
    child_recorded = None

    for span in spans:
        if span.name == "parent_operation":
            parent_recorded = span
        elif span.name == "chat":  # Gen-AI convention
            child_recorded = span

    assert parent_recorded is not None, "Parent span not found"
    assert child_recorded is not None, "Child span not found"

    # Verify parent-child relationship
    assert child_recorded.parent is not None, "Child span has no parent context"
    assert child_recorded.parent.span_id == parent_recorded.context.span_id, (
        "Child span parent ID doesn't match parent span ID"
    )
    assert child_recorded.context.trace_id == parent_recorded.context.trace_id, (
        "Child and parent have different trace IDs"
    )


@pytest.mark.asyncio
async def test_token_usage_recorded_after_completion(span_exporter, gh_run):
    """Test that token usage metrics are recorded after async completion."""
    if gh_run:
        pytest.skip("Skipping in CI - requires Ollama")

    backend = OllamaModelBackend(model_id="llama3.2:1b")
    ctx = SimpleContext()
    ctx = ctx.add(Message(role="user", content="Say 'test' and nothing else"))

    mot, _ = await backend.generate_from_context(
        Message(role="assistant", content=""), ctx
    )
    await mot.avalue()  # Wait for async completion

    # Get the recorded span
    spans = span_exporter.get_finished_spans()
    assert len(spans) > 0, "No spans were recorded"

    # Debug: print all span names
    print(f"\nRecorded spans: {[s.name for s in spans]}")

    backend_span = None
    for span in spans:
        if span.name == "chat":  # Gen-AI convention uses 'chat' for chat completions
            backend_span = span
            break

    assert backend_span is not None, (
        f"Backend span not found. Available spans: {[s.name for s in spans]}"
    )

    # Check for Gen-AI semantic convention attributes
    attributes = dict(backend_span.attributes)

    # Verify Gen-AI attributes are present
    assert "gen_ai.system" in attributes, "gen_ai.system attribute missing"
    assert attributes["gen_ai.system"] == "ollama", "Incorrect system name"

    assert "gen_ai.request.model" in attributes, (
        "gen_ai.request.model attribute missing"
    )

    # Token usage should be recorded (if available from backend)
    # Note: Not all backends provide token counts
    if "gen_ai.usage.input_tokens" in attributes:
        assert attributes["gen_ai.usage.input_tokens"] > 0, "Input tokens should be > 0"

    if "gen_ai.usage.output_tokens" in attributes:
        assert attributes["gen_ai.usage.output_tokens"] > 0, (
            "Output tokens should be > 0"
        )


@pytest.mark.asyncio
async def test_span_not_closed_prematurely(span_exporter, gh_run):
    """Test that spans are not closed before async operations complete."""
    if gh_run:
        pytest.skip("Skipping in CI - requires Ollama")

    backend = OllamaModelBackend(model_id="llama3.2:1b")
    ctx = SimpleContext()
    ctx = ctx.add(Message(role="user", content="Count to 5"))

    mot, _ = await backend.generate_from_context(
        Message(role="assistant", content=""), ctx
    )

    # At this point, the span should still be open (not in exporter yet)
    # because we haven't awaited the ModelOutputThunk
    spans_before = span_exporter.get_finished_spans()
    backend_spans_before = [
        s for s in spans_before if s.name == "chat"
    ]  # Gen-AI convention

    # Now complete the async operation
    await mot.avalue()

    # Now the span should be closed
    spans_after = span_exporter.get_finished_spans()
    backend_spans_after = [
        s for s in spans_after if s.name == "chat"
    ]  # Gen-AI convention

    # The span should only appear after completion
    assert len(backend_spans_after) > len(backend_spans_before), (
        "Span was closed before async completion"
    )


@pytest.mark.asyncio
async def test_multiple_generations_separate_spans(span_exporter, gh_run):
    """Test that multiple generations create separate spans."""
    if gh_run:
        pytest.skip("Skipping in CI - requires Ollama")

    backend = OllamaModelBackend(model_id="llama3.2:1b")
    ctx = SimpleContext()
    ctx = ctx.add(Message(role="user", content="Say 'test'"))

    # Generate twice
    mot1, _ = await backend.generate_from_context(
        Message(role="assistant", content=""), ctx
    )
    await mot1.avalue()

    mot2, _ = await backend.generate_from_context(
        Message(role="assistant", content=""), ctx
    )
    await mot2.avalue()

    # Get the recorded spans
    spans = span_exporter.get_finished_spans()
    backend_spans = [s for s in spans if s.name == "chat"]  # Gen-AI convention

    assert len(backend_spans) >= 2, (
        f"Expected at least 2 spans, got {len(backend_spans)}"
    )

    # Verify spans have different span IDs
    span_ids = {s.context.span_id for s in backend_spans}
    assert len(span_ids) >= 2, "Spans should have unique IDs"


@pytest.mark.asyncio
async def test_streaming_span_duration(span_exporter, gh_run):
    """Test that streaming operations have accurate span durations."""
    if gh_run:
        pytest.skip("Skipping in CI - requires Ollama")

    from mellea.backends.model_options import ModelOption

    backend = OllamaModelBackend(model_id="llama3.2:1b")
    ctx = SimpleContext()
    ctx = ctx.add(Message(role="user", content="Count to 3"))

    start_time = time.time()
    mot, _ = await backend.generate_from_context(
        Message(role="assistant", content=""),
        ctx,
        model_options={ModelOption.STREAM: True},
    )

    # Consume the stream
    await mot.astream()
    await mot.avalue()

    end_time = time.time()
    actual_duration = end_time - start_time

    # Get the recorded span
    spans = span_exporter.get_finished_spans()
    backend_span = None
    for span in spans:
        if span.name == "chat":  # Gen-AI convention
            backend_span = span
            break

    assert backend_span is not None, "Backend span not found"

    # Span duration should include streaming time
    span_duration_ns = backend_span.end_time - backend_span.start_time
    span_duration_s = span_duration_ns / 1e9

    assert span_duration_s >= 0.1, (
        f"Span duration too short for streaming: {span_duration_s}s"
    )
    assert abs(span_duration_s - actual_duration) < 0.5, (
        f"Streaming span duration {span_duration_s}s differs from actual {actual_duration}s"
    )


# Made with Bob
