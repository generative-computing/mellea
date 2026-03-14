"""Backend integration tests for token usage metrics.

Tests that backends correctly record token metrics through the telemetry system.
"""

import os

import pytest

from mellea.backends.model_ids import (
    IBM_GRANITE_4_HYBRID_MICRO,
    IBM_GRANITE_4_HYBRID_SMALL,
)
from mellea.stdlib.components import Message
from mellea.stdlib.context import SimpleContext

# Check if OpenTelemetry is available
try:
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not OTEL_AVAILABLE, reason="OpenTelemetry not installed"
)


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


@pytest.fixture(scope="module")
def hf_metrics_backend(gh_run):
    """Shared HuggingFace backend for telemetry metrics tests.

    Uses module scope to load the model once and reuse it across all tests,
    preventing memory exhaustion from loading multiple model instances.
    """
    if gh_run:
        pytest.skip("Skipping HuggingFace backend creation in CI")

    from mellea.backends.cache import SimpleLRUCache
    from mellea.backends.huggingface import LocalHFBackend

    backend = LocalHFBackend(
        model_id=IBM_GRANITE_4_HYBRID_MICRO.hf_model_name,  # type: ignore
        cache=SimpleLRUCache(5),
    )

    yield backend

    # Cleanup
    import gc

    del backend
    gc.collect()


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
async def test_ollama_token_metrics_integration(enable_metrics, metric_reader, stream):
    """Test that Ollama backend records token metrics correctly."""
    from mellea.backends.model_options import ModelOption
    from mellea.backends.ollama import OllamaModelBackend
    from mellea.telemetry import metrics as metrics_module

    provider = MeterProvider(metric_readers=[metric_reader])
    metrics_module._meter_provider = provider
    metrics_module._meter = provider.get_meter("mellea")
    metrics_module._input_token_counter = None
    metrics_module._output_token_counter = None

    backend = OllamaModelBackend(model_id=IBM_GRANITE_4_HYBRID_MICRO.ollama_name)  # type: ignore
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
        metrics_data, "mellea.llm.tokens.input", {"gen_ai.provider.name": "ollama"}
    )

    # Verify output token counter
    output_tokens = get_metric_value(
        metrics_data, "mellea.llm.tokens.output", {"gen_ai.provider.name": "ollama"}
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
async def test_openai_token_metrics_integration(enable_metrics, metric_reader, stream):
    """Test that OpenAI backend records token metrics correctly using Ollama's OpenAI-compatible endpoint."""
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
        model_id=IBM_GRANITE_4_HYBRID_MICRO.ollama_name,  # type: ignore
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
        metrics_data, "mellea.llm.tokens.input", {"gen_ai.provider.name": "openai"}
    )

    output_tokens = get_metric_value(
        metrics_data, "mellea.llm.tokens.output", {"gen_ai.provider.name": "openai"}
    )

    assert input_tokens is not None, "Input tokens should be recorded"
    assert input_tokens > 0, f"Input tokens should be > 0, got {input_tokens}"

    assert output_tokens is not None, "Output tokens should be recorded"
    assert output_tokens > 0, f"Output tokens should be > 0, got {output_tokens}"


@pytest.mark.asyncio
@pytest.mark.llm
@pytest.mark.watsonx
@pytest.mark.requires_api_key
async def test_watsonx_token_metrics_integration(enable_metrics, metric_reader):
    """Test that WatsonX backend records token metrics correctly."""
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
        model_id=IBM_GRANITE_4_HYBRID_SMALL.watsonx_name,  # type: ignore
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
        metrics_data, "mellea.llm.tokens.input", {"gen_ai.provider.name": "watsonx"}
    )

    output_tokens = get_metric_value(
        metrics_data, "mellea.llm.tokens.output", {"gen_ai.provider.name": "watsonx"}
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
    enable_metrics, metric_reader, monkeypatch, stream
):
    """Test that LiteLLM backend records token metrics correctly using OpenAI-compatible endpoint."""
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
    backend = LiteLLMBackend(
        model_id=f"openai/{IBM_GRANITE_4_HYBRID_MICRO.ollama_name}"
    )  # type: ignore
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
        metrics_data, "mellea.llm.tokens.input", {"gen_ai.provider.name": "litellm"}
    )

    output_tokens = get_metric_value(
        metrics_data, "mellea.llm.tokens.output", {"gen_ai.provider.name": "litellm"}
    )

    # LiteLLM with Ollama backend should always provide token counts
    assert input_tokens is not None, "Input tokens should be recorded"
    assert input_tokens > 0, f"Input tokens should be > 0, got {input_tokens}"

    assert output_tokens is not None, "Output tokens should be recorded"
    assert output_tokens > 0, f"Output tokens should be > 0, got {output_tokens}"


@pytest.mark.asyncio
@pytest.mark.llm
@pytest.mark.huggingface
@pytest.mark.requires_heavy_ram
@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
async def test_huggingface_token_metrics_integration(
    enable_metrics, metric_reader, stream, hf_metrics_backend
):
    """Test that HuggingFace backend records token metrics correctly."""
    from mellea.backends.model_options import ModelOption
    from mellea.telemetry import metrics as metrics_module

    provider = MeterProvider(metric_readers=[metric_reader])
    metrics_module._meter_provider = provider
    metrics_module._meter = provider.get_meter("mellea")
    metrics_module._input_token_counter = None
    metrics_module._output_token_counter = None

    ctx = SimpleContext()
    ctx = ctx.add(Message(role="user", content="Say 'hello' and nothing else"))

    model_options = {ModelOption.STREAM: True} if stream else {}
    mot, _ = await hf_metrics_backend.generate_from_context(
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
        metrics_data, "mellea.llm.tokens.input", {"gen_ai.provider.name": "huggingface"}
    )

    output_tokens = get_metric_value(
        metrics_data,
        "mellea.llm.tokens.output",
        {"gen_ai.provider.name": "huggingface"},
    )

    assert input_tokens is not None, "Input tokens should be recorded"
    assert input_tokens > 0, f"Input tokens should be > 0, got {input_tokens}"

    assert output_tokens is not None, "Output tokens should be recorded"
    assert output_tokens > 0, f"Output tokens should be > 0, got {output_tokens}"
