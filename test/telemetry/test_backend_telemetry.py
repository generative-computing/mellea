"""Unit tests for backend telemetry instrumentation with Gen-AI semantic conventions."""

import asyncio
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.components import Message
from mellea.stdlib.context import SimpleContext

# Check if OpenTelemetry is available
try:
    from opentelemetry import metrics, trace
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
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


@pytest.fixture(scope="module")
def setup_telemetry():
    """Set up telemetry for all tests in this module."""
    import importlib

    mp = pytest.MonkeyPatch()
    mp.setenv("MELLEA_TRACE_BACKEND", "true")
    mp.setenv("MELLEA_METRICS_ENABLED", "true")

    import mellea.telemetry.tracing

    importlib.reload(mellea.telemetry.tracing)

    yield

    mp.undo()
    importlib.reload(mellea.telemetry.tracing)


@pytest.fixture
def span_exporter(setup_telemetry):
    """Create an in-memory span exporter for testing."""
    # Import mellea.telemetry.tracing to ensure it's initialized
    from mellea.telemetry import tracing

    # Get the real tracer provider from mellea.telemetry.tracing module
    # The global trace.get_tracer_provider() returns a ProxyTracerProvider
    provider = tracing._tracer_provider

    if provider is None:
        pytest.skip("Telemetry not initialized")

    # Add our in-memory exporter to it
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    yield exporter
    exporter.clear()


@pytest.fixture
def metric_reader():
    """Create an in-memory metric reader for testing."""
    reader = InMemoryMetricReader()
    yield reader


@pytest.fixture
def enable_metrics(monkeypatch):
    """Enable metrics for tests."""
    monkeypatch.setenv("MELLEA_METRICS_ENABLED", "true")
    # Force reload of metrics module to pick up env vars
    import importlib

    import mellea.telemetry.metrics

    importlib.reload(mellea.telemetry.metrics)
    yield
    # Reset after test
    monkeypatch.setenv("MELLEA_METRICS_ENABLED", "false")
    importlib.reload(mellea.telemetry.metrics)


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


# ============================================================================
# Token Metrics Integration Tests
# ============================================================================


def get_metric_value(metrics_data, metric_name, attributes=None):
    """Helper to extract metric value from metrics data.

    Args:
        metrics_data: Metrics data from reader (may be None)
        metric_name: Name of the metric to find
        attributes: Optional dict of attributes to match

    Returns:
        The metric value or None if not found
    """
    if metrics_data is None:
        return None

    for resource_metrics in metrics_data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name == metric_name:
                    for data_point in metric.data.data_points:
                        if attributes is None:
                            return data_point.value
                        # Check if attributes match
                        point_attrs = dict(data_point.attributes)
                        if all(point_attrs.get(k) == v for k, v in attributes.items()):
                            return data_point.value
    return None


@pytest.mark.asyncio
@pytest.mark.llm
@pytest.mark.ollama
@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
async def test_ollama_token_metrics_integration(
    enable_metrics, metric_reader, gh_run, stream
):
    """Test that Ollama backend records token metrics correctly."""
    if gh_run:
        pytest.skip("Skipping in CI - requires Ollama")

    from mellea.backends.model_options import ModelOption
    from mellea.backends.ollama import OllamaModelBackend
    from mellea.telemetry import metrics as metrics_module

    provider = MeterProvider(metric_readers=[metric_reader])
    metrics_module._meter_provider = provider
    metrics_module._meter = provider.get_meter("mellea")
    metrics_module._input_token_counter = None
    metrics_module._output_token_counter = None

    backend = OllamaModelBackend(model_id="llama3.2:1b")
    ctx = SimpleContext()
    ctx = ctx.add(Message(role="user", content="Say 'hello' and nothing else"))

    model_options = {ModelOption.STREAM: True} if stream else {}
    mot, _ = await backend.generate_from_context(
        Message(role="assistant", content=""), ctx, model_options=model_options
    )

    # For streaming, consume the stream fully before checking metrics
    if stream:
        await mot.astream()
    await mot.avalue()

    # Force metrics export and collection
    provider.force_flush()
    metrics_data = metric_reader.get_metrics_data()

    # Verify input token counter
    input_tokens = get_metric_value(
        metrics_data, "mellea.llm.tokens.input", {"gen_ai.system": "ollama"}
    )

    # Verify output token counter
    output_tokens = get_metric_value(
        metrics_data, "mellea.llm.tokens.output", {"gen_ai.system": "ollama"}
    )

    # Ollama should always return token counts
    assert input_tokens is not None, "Input tokens should not be None"
    assert input_tokens > 0, f"Input tokens should be > 0, got {input_tokens}"

    assert output_tokens is not None, "Output tokens should not be None"
    assert output_tokens > 0, f"Output tokens should be > 0, got {output_tokens}"


@pytest.mark.asyncio
@pytest.mark.llm
@pytest.mark.ollama
@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
async def test_openai_token_metrics_integration(
    enable_metrics, metric_reader, gh_run, stream
):
    """Test that OpenAI backend records token metrics correctly using Ollama's OpenAI-compatible endpoint."""
    if gh_run:
        pytest.skip("Skipping in CI - requires Ollama")

    from mellea.backends.model_options import ModelOption
    from mellea.backends.openai import OpenAIBackend
    from mellea.telemetry import metrics as metrics_module

    provider = MeterProvider(metric_readers=[metric_reader])
    metrics_module._meter_provider = provider
    metrics_module._meter = provider.get_meter("mellea")
    metrics_module._input_token_counter = None
    metrics_module._output_token_counter = None

    # Use Ollama's OpenAI-compatible endpoint
    backend = OpenAIBackend(
        model_id="llama3.2:1b",
        base_url=f"http://{os.environ.get('OLLAMA_HOST', 'localhost:11434')}/v1",
        api_key="ollama",
    )
    ctx = SimpleContext()
    ctx = ctx.add(Message(role="user", content="Say 'hello' and nothing else"))

    model_options = {ModelOption.STREAM: True} if stream else {}
    mot, _ = await backend.generate_from_context(
        Message(role="assistant", content=""), ctx, model_options=model_options
    )

    # For streaming, consume the stream fully before checking metrics
    if stream:
        await mot.astream()
    await mot.avalue()

    provider.force_flush()
    metrics_data = metric_reader.get_metrics_data()

    # OpenAI always provides token counts
    input_tokens = get_metric_value(
        metrics_data, "mellea.llm.tokens.input", {"gen_ai.system": "openai"}
    )

    output_tokens = get_metric_value(
        metrics_data, "mellea.llm.tokens.output", {"gen_ai.system": "openai"}
    )

    assert input_tokens is not None, "Input tokens should be recorded"
    assert input_tokens > 0, f"Input tokens should be > 0, got {input_tokens}"

    assert output_tokens is not None, "Output tokens should be recorded"
    assert output_tokens > 0, f"Output tokens should be > 0, got {output_tokens}"


@pytest.mark.asyncio
@pytest.mark.llm
@pytest.mark.watsonx
@pytest.mark.requires_api_key
async def test_watsonx_token_metrics_integration(enable_metrics, metric_reader, gh_run):
    """Test that WatsonX backend records token metrics correctly."""
    if gh_run:
        pytest.skip("Skipping in CI - requires WatsonX credentials")

    if not os.getenv("WATSONX_API_KEY"):
        pytest.skip("WATSONX_API_KEY not set")

    from mellea.backends.watsonx import WatsonxAIBackend
    from mellea.telemetry import metrics as metrics_module

    provider = MeterProvider(metric_readers=[metric_reader])
    metrics_module._meter_provider = provider
    metrics_module._meter = provider.get_meter("mellea")
    metrics_module._input_token_counter = None
    metrics_module._output_token_counter = None

    backend = WatsonxAIBackend(
        model_id="ibm/granite-4-h-small",
        project_id=os.getenv("WATSONX_PROJECT_ID", "test-project"),
    )
    ctx = SimpleContext()
    ctx = ctx.add(Message(role="user", content="Say 'hello' and nothing else"))

    mot, _ = await backend.generate_from_context(
        Message(role="assistant", content=""), ctx
    )
    await mot.avalue()

    provider.force_flush()
    metrics_data = metric_reader.get_metrics_data()

    input_tokens = get_metric_value(
        metrics_data, "mellea.llm.tokens.input", {"gen_ai.system": "watsonx"}
    )

    output_tokens = get_metric_value(
        metrics_data, "mellea.llm.tokens.output", {"gen_ai.system": "watsonx"}
    )

    assert input_tokens is not None, "Input tokens should be recorded"
    assert input_tokens > 0, f"Input tokens should be > 0, got {input_tokens}"

    assert output_tokens is not None, "Output tokens should be recorded"
    assert output_tokens > 0, f"Output tokens should be > 0, got {output_tokens}"


@pytest.mark.asyncio
@pytest.mark.llm
@pytest.mark.litellm
@pytest.mark.ollama
@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
async def test_litellm_token_metrics_integration(
    enable_metrics, metric_reader, gh_run, monkeypatch, stream
):
    """Test that LiteLLM backend records token metrics correctly using OpenAI-compatible endpoint."""
    if gh_run:
        pytest.skip("Skipping in CI - requires Ollama")

    from mellea.backends.litellm import LiteLLMBackend
    from mellea.backends.model_options import ModelOption
    from mellea.telemetry import metrics as metrics_module

    # Set environment variables for LiteLLM to use Ollama's OpenAI-compatible endpoint
    ollama_url = f"http://{os.environ.get('OLLAMA_HOST', 'localhost:11434')}/v1"
    monkeypatch.setenv("OPENAI_API_KEY", "ollama")
    monkeypatch.setenv("OPENAI_BASE_URL", ollama_url)

    provider = MeterProvider(metric_readers=[metric_reader])
    metrics_module._meter_provider = provider
    metrics_module._meter = provider.get_meter("mellea")
    metrics_module._input_token_counter = None
    metrics_module._output_token_counter = None

    # Use LiteLLM with openai/ prefix - it will use the OPENAI_BASE_URL env var
    # This tests LiteLLM with a provider that properly returns token usage
    backend = LiteLLMBackend(model_id="openai/llama3.2:1b")
    ctx = SimpleContext()
    ctx = ctx.add(Message(role="user", content="Say 'hello' and nothing else"))

    model_options = {ModelOption.STREAM: True} if stream else {}
    mot, _ = await backend.generate_from_context(
        Message(role="assistant", content=""), ctx, model_options=model_options
    )

    # For streaming, consume the stream fully before checking metrics
    if stream:
        await mot.astream()
    await mot.avalue()

    provider.force_flush()
    metrics_data = metric_reader.get_metrics_data()

    input_tokens = get_metric_value(
        metrics_data, "mellea.llm.tokens.input", {"gen_ai.system": "litellm"}
    )

    output_tokens = get_metric_value(
        metrics_data, "mellea.llm.tokens.output", {"gen_ai.system": "litellm"}
    )

    # LiteLLM with Ollama backend should always provide token counts
    assert input_tokens is not None, "Input tokens should be recorded"
    assert input_tokens > 0, f"Input tokens should be > 0, got {input_tokens}"

    assert output_tokens is not None, "Output tokens should be recorded"
    assert output_tokens > 0, f"Output tokens should be > 0, got {output_tokens}"


@pytest.mark.asyncio
@pytest.mark.llm
@pytest.mark.huggingface
@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
async def test_huggingface_token_metrics_integration(
    enable_metrics, metric_reader, gh_run, stream
):
    """Test that HuggingFace backend records token metrics correctly."""
    if gh_run:
        pytest.skip("Skipping in CI - requires model download")

    from mellea.backends.huggingface import LocalHFBackend
    from mellea.backends.model_options import ModelOption
    from mellea.telemetry import metrics as metrics_module

    provider = MeterProvider(metric_readers=[metric_reader])
    metrics_module._meter_provider = provider
    metrics_module._meter = provider.get_meter("mellea")
    metrics_module._input_token_counter = None
    metrics_module._output_token_counter = None

    from mellea.backends.cache import SimpleLRUCache

    backend = LocalHFBackend(
        model_id="ibm-granite/granite-4.0-micro", cache=SimpleLRUCache(5)
    )
    ctx = SimpleContext()
    ctx = ctx.add(Message(role="user", content="Say 'hello' and nothing else"))

    model_options = {ModelOption.STREAM: True} if stream else {}
    mot, _ = await backend.generate_from_context(
        Message(role="assistant", content=""), ctx, model_options=model_options
    )

    # For streaming, consume the stream fully before checking metrics
    if stream:
        await mot.astream()
    await mot.avalue()

    provider.force_flush()
    metrics_data = metric_reader.get_metrics_data()

    # HuggingFace computes token counts locally
    input_tokens = get_metric_value(
        metrics_data, "mellea.llm.tokens.input", {"gen_ai.system": "huggingface"}
    )

    output_tokens = get_metric_value(
        metrics_data, "mellea.llm.tokens.output", {"gen_ai.system": "huggingface"}
    )

    assert input_tokens is not None, "Input tokens should be recorded"
    assert input_tokens > 0, f"Input tokens should be > 0, got {input_tokens}"

    assert output_tokens is not None, "Output tokens should be recorded"
    assert output_tokens > 0, f"Output tokens should be > 0, got {output_tokens}"
