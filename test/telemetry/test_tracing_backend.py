"""End-to-end tests for backend tracing with Gen-AI semantic conventions."""

import pytest

from mellea.backends.model_ids import IBM_GRANITE_4_1_3B
from mellea.backends.ollama import OllamaModelBackend
from mellea.plugins.manager import (
    disable_background_collection,
    discard_background_tasks,
    drain_background_tasks,
    enable_background_collection,
)
from mellea.stdlib.components import Message
from mellea.stdlib.context import SimpleContext

# Check if OpenTelemetry is available
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
    pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not installed"),
    pytest.mark.e2e,
    pytest.mark.ollama,
]


def _reset_tracing_state() -> None:
    import mellea.telemetry.tracing as tracing_mod

    tracing_mod._tracer_provider = None
    tracing_mod._application_tracer = None
    tracing_mod._backend_tracer = None
    tracing_mod._in_flight_spans.clear()
    tracing_mod._setup_tracing()


@pytest.fixture(scope="module", autouse=True)
def setup_telemetry():
    """Enable tracing for all tests in this module."""
    mp = pytest.MonkeyPatch()
    mp.setenv("MELLEA_TRACES_ENABLED", "true")
    _reset_tracing_state()

    yield

    mp.undo()
    _reset_tracing_state()


@pytest.fixture
def span_exporter():
    """Create an in-memory span exporter attached to the tracing module's provider.

    The plugin's post_call/error hooks run in FIRE_AND_FORGET mode, so each
    test must drain background tasks before asserting on the exporter. We
    enable collection here and provide a wrapper that drains + flushes on
    every read.
    """
    from mellea.telemetry import tracing

    # Trigger lazy init so the provider exists.
    tracing.get_backend_tracer()
    provider = tracing._tracer_provider

    if provider is None:
        pytest.skip("Telemetry not initialized")

    enable_background_collection()
    discard_background_tasks()

    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    yield exporter

    exporter.clear()
    disable_background_collection()


@pytest.mark.asyncio
async def test_span_duration_captures_async_operation(span_exporter):
    """Test that span duration includes the full async operation time."""

    backend = OllamaModelBackend(model_id=IBM_GRANITE_4_1_3B.ollama_name)  # type: ignore
    ctx = SimpleContext()
    ctx = ctx.add(Message(role="user", content="Say 'test' and nothing else"))

    mot, _ = await backend.generate_from_context(
        Message(role="assistant", content=""), ctx
    )
    await mot.avalue()  # Wait for async completion
    await drain_background_tasks()

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

    span_duration_ns = backend_span.end_time - backend_span.start_time
    span_duration_s = span_duration_ns / 1e9

    assert span_duration_s >= 0.1, (
        f"Span duration too short: {span_duration_s}s (expected >= 0.1s)"
    )


@pytest.mark.asyncio
async def test_context_propagation_parent_child(span_exporter):
    """Test that parent-child span relationships are maintained."""

    backend = OllamaModelBackend(model_id=IBM_GRANITE_4_1_3B.ollama_name)  # type: ignore
    ctx = SimpleContext()
    ctx = ctx.add(Message(role="user", content="Say 'test' and nothing else"))

    # Create a parent span using the module's own tracer provider
    # (not the global one, which may be pinned to a different provider
    # due to OTel's set-once semantics for set_tracer_provider)
    from mellea.telemetry import tracing

    tracer = tracing._tracer_provider.get_tracer(__name__)
    with tracer.start_as_current_span("parent_operation"):
        mot, _ = await backend.generate_from_context(
            Message(role="assistant", content=""), ctx
        )
        await mot.avalue()  # Wait for async completion
        await drain_background_tasks()

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
async def test_token_usage_recorded_after_completion(span_exporter):
    """Test that token usage metrics are recorded after async completion."""

    backend = OllamaModelBackend(model_id=IBM_GRANITE_4_1_3B.ollama_name)  # type: ignore
    ctx = SimpleContext()
    ctx = ctx.add(Message(role="user", content="Say 'test' and nothing else"))

    mot, _ = await backend.generate_from_context(
        Message(role="assistant", content=""), ctx
    )
    await mot.avalue()  # Wait for async completion
    await drain_background_tasks()

    # Get the recorded span
    spans = span_exporter.get_finished_spans()
    assert len(spans) > 0, "No spans were recorded"

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
    assert attributes.get("gen_ai.provider.name") == "ollama", "Incorrect provider name"
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
async def test_span_not_closed_prematurely(span_exporter):
    """Test that spans are not closed before async operations complete."""

    backend = OllamaModelBackend(model_id=IBM_GRANITE_4_1_3B.ollama_name)  # type: ignore
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
    await drain_background_tasks()

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
async def test_multiple_generations_separate_spans(span_exporter):
    """Test that multiple generations create separate spans."""

    backend = OllamaModelBackend(model_id=IBM_GRANITE_4_1_3B.ollama_name)  # type: ignore
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
    await drain_background_tasks()

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
async def test_streaming_span_duration(span_exporter):
    """Test that streaming operations have accurate span durations."""

    from mellea.backends.model_options import ModelOption

    backend = OllamaModelBackend(model_id=IBM_GRANITE_4_1_3B.ollama_name)  # type: ignore
    ctx = SimpleContext()
    ctx = ctx.add(Message(role="user", content="Count to 3"))

    mot, _ = await backend.generate_from_context(
        Message(role="assistant", content=""),
        ctx,
        model_options={ModelOption.STREAM: True},
    )

    # Consume the stream
    await mot.astream()
    await mot.avalue()
    await drain_background_tasks()

    # Get the recorded span
    spans = span_exporter.get_finished_spans()
    backend_span = None
    for span in spans:
        if span.name == "chat":  # Gen-AI convention
            backend_span = span
            break

    assert backend_span is not None, "Backend span not found"

    span_duration_ns = backend_span.end_time - backend_span.start_time
    span_duration_s = span_duration_ns / 1e9

    assert span_duration_s >= 0.1, (
        f"Span duration too short for streaming: {span_duration_s}s"
    )
