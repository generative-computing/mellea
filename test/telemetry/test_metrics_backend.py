# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend integration tests for token usage metrics.

Tests that backends correctly record token metrics through the telemetry system.
"""

import asyncio
import os

import pytest

from mellea.backends.model_ids import IBM_GRANITE_4_2_3B, IBM_GRANITE_4_HYBRID_SMALL
from mellea.plugins.manager import (
    disable_background_collection,
    discard_background_tasks,
    drain_background_tasks,
    enable_background_collection,
)
from mellea.stdlib.components import Message
from mellea.stdlib.context import SimpleContext
from test.conftest import hf_skip
from test.predicates import require_api_key, require_gpu
from test.telemetry.conftest import reset_metrics_state

# Check if OpenTelemetry is available
try:
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

pytestmark = [
    pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not installed"),
    pytest.mark.e2e,
]

TEST_CONTEXT_WINDOW = 2048


@pytest.fixture
def metric_reader():
    """Create an in-memory metric reader for testing."""
    reader = InMemoryMetricReader()
    yield reader


@pytest.fixture
def enable_metrics(monkeypatch):
    """Enable metrics for tests."""
    enable_background_collection()
    discard_background_tasks()
    monkeypatch.setenv("MELLEA_METRICS_ENABLED", "true")
    reset_metrics_state()
    yield
    monkeypatch.setenv("MELLEA_METRICS_ENABLED", "false")
    reset_metrics_state()
    disable_background_collection()


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

    with hf_skip():
        backend = LocalHFBackend(
            model_id=IBM_GRANITE_4_2_3B.hf_model_name,  # type: ignore
            cache=SimpleLRUCache(5),
        )

    yield backend

    from test.conftest import cleanup_gpu_backend

    cleanup_gpu_backend(backend, "hf-metrics")


def _setup_metrics_provider(metrics_module, metric_reader):
    """Wire an InMemoryMetricReader into the metrics module globals."""
    provider = MeterProvider(metric_readers=[metric_reader])
    metrics_module._meter_provider = provider
    metrics_module._meter = provider.get_meter("mellea")
    metrics_module._token_usage_histogram = None
    metrics_module._duration_histogram = None
    metrics_module._ttfb_histogram = None
    metrics_module._time_per_output_chunk_histogram = None
    metrics_module._error_counter = None
    metrics_module._sampling_attempts_counter = None
    metrics_module._sampling_successes_counter = None
    metrics_module._sampling_failures_counter = None
    metrics_module._requirement_checks_counter = None
    metrics_module._requirement_failures_counter = None
    metrics_module._tool_calls_counter = None
    metrics_module._cost_counter = None
    return provider


def _find_histogram_data_point(metrics_data, metric_name, attributes=None):
    """Return the first histogram data point matching the given attributes."""
    if metrics_data is None:
        return None
    for rm in metrics_data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == metric_name:
                    for dp in metric.data.data_points:
                        if attributes is None:
                            return dp
                        point_attrs = dict(dp.attributes)
                        if all(point_attrs.get(k) == v for k, v in attributes.items()):
                            return dp
    return None


def _token_sum(metrics_data, provider, token_type):
    """Return the recorded token count for a provider/token type from the usage histogram."""
    dp = _find_histogram_data_point(
        metrics_data,
        "gen_ai.client.token.usage",
        {"gen_ai.provider.name": provider, "gen_ai.token.type": token_type},
    )
    return dp.sum if dp is not None else None


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
@pytest.mark.ollama
@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
async def test_ollama_token_metrics_integration(
    enable_metrics, metric_reader, monkeypatch, stream
):
    """Test that Ollama backend records token metrics correctly."""
    from mellea.backends.model_options import ModelOption
    from mellea.backends.ollama import OllamaModelBackend
    from mellea.telemetry import metrics as metrics_module

    monkeypatch.setenv("MELLEA_GENERATION_CHUNK_EVENTS", "true")
    provider = _setup_metrics_provider(metrics_module, metric_reader)

    backend = OllamaModelBackend(  # type: ignore
        model_id=IBM_GRANITE_4_2_3B.ollama_name,
        model_options={
            ModelOption.CONTEXT_WINDOW: TEST_CONTEXT_WINDOW,
            ModelOption.THINKING: False,
            # Bound the worst case: this "say hello" prompt has produced
            # 1800+ token generations on CI, and at CI's ~9 t/s CPU decode
            # that is a 3-15 minute single request that can eat the test's
            # entire pytest-timeout budget. 64 tokens is far more than a
            # "hello" answer needs.
            ModelOption.MAX_NEW_TOKENS: 64,
        },
    )
    ctx = SimpleContext()
    ctx = ctx.add(Message(role="user", content="Say 'hello' and nothing else"))

    model_options = {ModelOption.STREAM: True} if stream else {}
    mot, _ = await backend.generate_from_context(
        Message(role="assistant", content=""), ctx, model_options=model_options
    )

    # For streaming, consume the stream fully before checking metrics
    if stream:
        # The 120 s per-chunk stream guard does not bound total request time:
        # a stalled or queued stream that keeps the HTTP connection alive
        # (observed on CI: a 900 s /v1 stream, run 33015176815) keeps the
        # per-chunk guard re-armed and rides to the 900 s pytest watchdog,
        # killing the job. Bound the stream to the same 300 s the
        # non-streaming paths use so a genuine stall surfaces as a bounded
        # TimeoutError the flaky marker retries, matching non-streaming
        # behaviour.
        await asyncio.wait_for(mot.astream(), timeout=300.0)
    # astream() returns as soon as its queue drains, so a long stream is
    # finished by avalue(); keep that inside the same 300 s budget (run
    # 33048969379: a 15 m stream escaped the astream bound above).
    await asyncio.wait_for(mot.avalue(), timeout=300.0)

    # Force metrics export and collection
    await drain_background_tasks()
    provider.force_flush()
    metrics_data = metric_reader.get_metrics_data()

    # Verify input token counter
    input_tokens = _token_sum(metrics_data, "ollama", "input")

    # Verify output tokens
    output_tokens = _token_sum(metrics_data, "ollama", "output")

    # Ollama should always return token counts
    assert input_tokens is not None, "Input tokens should not be None"
    assert input_tokens > 0, f"Input tokens should be > 0, got {input_tokens}"

    assert output_tokens is not None, "Output tokens should not be None"
    assert output_tokens > 0, f"Output tokens should be > 0, got {output_tokens}"

    # Verify latency metrics
    duration_dp = _find_histogram_data_point(
        metrics_data,
        "gen_ai.client.operation.duration",
        {"gen_ai.request.stream": stream},
    )
    assert duration_dp is not None, "Request duration should be recorded"
    assert duration_dp.sum > 0, "Request duration should be > 0"

    if stream:
        ttfb_dp = _find_histogram_data_point(
            metrics_data, "gen_ai.client.operation.time_to_first_chunk"
        )
        assert ttfb_dp is not None, "TTFB should be recorded for streaming requests"
        assert ttfb_dp.sum > 0, "TTFB should be > 0"

        # With MELLEA_GENERATION_CHUNK_EVENTS enabled, each chunk after the
        # first records an inter-chunk interval.
        tpoc_dp = _find_histogram_data_point(
            metrics_data, "gen_ai.client.operation.time_per_output_chunk"
        )
        assert tpoc_dp is not None, (
            "time_per_output_chunk should be recorded when chunk events are enabled"
        )
        assert tpoc_dp.count >= 1, (
            "at least one inter-chunk interval should be recorded"
        )


@pytest.mark.asyncio
@pytest.mark.openai
@pytest.mark.ollama
@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
async def test_openai_token_metrics_integration(enable_metrics, metric_reader, stream):
    """Test that OpenAI backend records token metrics correctly using Ollama's OpenAI-compatible endpoint."""
    from mellea.backends.model_options import ModelOption
    from mellea.backends.openai import OpenAIBackend
    from mellea.telemetry import metrics as metrics_module

    provider = _setup_metrics_provider(metrics_module, metric_reader)

    # Use Ollama's OpenAI-compatible endpoint
    backend = OpenAIBackend(
        model_id=IBM_GRANITE_4_2_3B.ollama_name,  # type: ignore
        base_url=f"http://{os.environ.get('OLLAMA_HOST', 'localhost:11434')}/v1",
        api_key="ollama",
        # granite4.2 thinks by default and Ollama's /v1 endpoint exposes no
        # template-kwargs control for it: THINKING: False sends
        # reasoning_effort="none", which the CI-pinned Ollama 0.33.1 maps to
        # think=false. Without it a 64-token generation is ~45 thinking
        # tokens — minutes on a 4-vCPU runner, blowing the 300 s request cap
        # whenever the runner is 2-3x slower than nominal (runs
        # 33093698028, 33104419835). The vLLM-style
        # chat_template_kwargs.enable_thinking workaround was ignored by
        # Ollama, which is why every /v1 CI run wedged in the openai/litellm
        # phase.
        # Same output bound as the other live "say hello" tests: without it a
        # non-compliant generation (observed 1800+ tokens on CI) can run for
        # minutes on a ~9 t/s CPU runner and eat the test's entire budget.
        # Sent as the raw "max_tokens" key for consistency with the litellm
        # test (Ollama 0.33.1 /v1 accepts both max_tokens and
        # max_completion_tokens; 0.32.2 accepted only max_tokens, which is
        # why the raw key was introduced).
        model_options={ModelOption.THINKING: False, "max_tokens": 64},
        # Disable the OpenAI SDK's automatic retries and bound each request to
        # the 300 s the other live paths use. With the SDK defaults (2 retries,
        # 600 s read timeout) a stalled server multiplies into a
        # 600 s + 600 s chain inside one attempt (observed in run
        # 33039206380: a request timed out at 10m0s, the SDK retried, and the
        # second attempt was killed by the 900 s pytest watchdog mid-flight).
        max_retries=0,
        timeout=300.0,
    )
    ctx = SimpleContext()
    ctx = ctx.add(Message(role="user", content="Say 'hello' and nothing else"))

    model_options = {ModelOption.STREAM: True} if stream else {}
    mot, _ = await backend.generate_from_context(
        Message(role="assistant", content=""), ctx, model_options=model_options
    )

    # For streaming, consume the stream fully before checking metrics
    if stream:
        # The 120 s per-chunk stream guard does not bound total request time:
        # a stalled or queued stream that keeps the HTTP connection alive
        # (observed on CI: a 900 s /v1 stream, run 33015176815) keeps the
        # per-chunk guard re-armed and rides to the 900 s pytest watchdog,
        # killing the job. Bound the stream to the same 300 s the
        # non-streaming paths use so a genuine stall surfaces as a bounded
        # TimeoutError the flaky marker retries, matching non-streaming
        # behaviour.
        await asyncio.wait_for(mot.astream(), timeout=300.0)
    # astream() returns as soon as its queue drains, so a long stream is
    # finished by avalue(); keep that inside the same 300 s budget (run
    # 33048969379: a 15 m stream escaped the astream bound above).
    await asyncio.wait_for(mot.avalue(), timeout=300.0)

    await drain_background_tasks()
    provider.force_flush()
    metrics_data = metric_reader.get_metrics_data()

    # OpenAI always provides token counts
    input_tokens = _token_sum(metrics_data, "openai", "input")

    output_tokens = _token_sum(metrics_data, "openai", "output")

    assert input_tokens is not None, "Input tokens should be recorded"
    assert input_tokens > 0, f"Input tokens should be > 0, got {input_tokens}"

    assert output_tokens is not None, "Output tokens should be recorded"
    assert output_tokens > 0, f"Output tokens should be > 0, got {output_tokens}"

    # Verify latency metrics
    duration_dp = _find_histogram_data_point(
        metrics_data,
        "gen_ai.client.operation.duration",
        {"gen_ai.request.stream": stream},
    )
    assert duration_dp is not None, "Request duration should be recorded"
    assert duration_dp.sum > 0, "Request duration should be > 0"

    if stream:
        ttfb_dp = _find_histogram_data_point(
            metrics_data, "gen_ai.client.operation.time_to_first_chunk"
        )
        assert ttfb_dp is not None, "TTFB should be recorded for streaming requests"
        assert ttfb_dp.sum > 0, "TTFB should be > 0"


@pytest.mark.asyncio
@pytest.mark.watsonx
@require_api_key("WATSONX_API_KEY", "WATSONX_URL", "WATSONX_PROJECT_ID")
async def test_watsonx_token_metrics_integration(enable_metrics, metric_reader):
    """Test that WatsonX backend records token metrics correctly."""
    from mellea.backends.watsonx import WatsonxAIBackend
    from mellea.telemetry import metrics as metrics_module

    provider = _setup_metrics_provider(metrics_module, metric_reader)

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

    await drain_background_tasks()
    provider.force_flush()
    metrics_data = metric_reader.get_metrics_data()

    input_tokens = _token_sum(metrics_data, "ibm.watsonx.ai", "input")

    output_tokens = _token_sum(metrics_data, "ibm.watsonx.ai", "output")

    assert input_tokens is not None, "Input tokens should be recorded"
    assert input_tokens > 0, f"Input tokens should be > 0, got {input_tokens}"

    assert output_tokens is not None, "Output tokens should be recorded"
    assert output_tokens > 0, f"Output tokens should be > 0, got {output_tokens}"

    # Verify cost metric — litellm has pricing for ibm/granite-4-h-small on watsonx
    cost = get_metric_value(
        metrics_data, "mellea.llm.cost.usd", {"gen_ai.provider.name": "ibm.watsonx.ai"}
    )
    assert cost is not None, "Cost should be recorded for a known watsonx model"
    assert cost > 0, f"Cost should be > 0, got {cost}"

    # Verify latency metrics (watsonx is non-streaming only)
    duration_dp = _find_histogram_data_point(
        metrics_data,
        "gen_ai.client.operation.duration",
        {"gen_ai.request.stream": False},
    )
    assert duration_dp is not None, "Request duration should be recorded"
    assert duration_dp.sum > 0, "Request duration should be > 0"


@pytest.mark.asyncio
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

    provider = _setup_metrics_provider(metrics_module, metric_reader)

    # Use LiteLLM with openai/ prefix - it will use the OPENAI_BASE_URL env var
    # This tests LiteLLM with a provider that properly returns token usage.
    #
    # "timeout" bounds each attempt to the same 300 s the native
    # OllamaModelBackend uses, and "num_retries": 0 stops the OpenAI SDK from
    # silently retrying a stalled attempt: without both, a single stalled
    # request can consume the whole 900 s pytest-timeout budget (litellm falls
    # back to a 600 s per-attempt timeout and the SDK retries twice), so the
    # test dies to the watchdog with no retryable error. With the bound in
    # place, a stall raises litellm.Timeout, whose APITimeoutError message the
    # conftest flaky marker (OLLAMA_TIMEOUT_RERUN_PATTERNS) retries like a
    # native ReadTimeout.
    backend = LiteLLMBackend(  # type: ignore
        model_id=f"openai/{IBM_GRANITE_4_2_3B.ollama_name}",
        # "max_tokens": same output bound as the other live "say hello"
        # tests (see test_ollama_token_metrics_integration). Sent as the raw
        # key for consistency with the openai test (Ollama 0.33.1 /v1
        # accepts both max_tokens and max_completion_tokens; 0.32.2
        # accepted only max_tokens, which is why the raw key was
        # introduced). THINKING: False is required: granite4.2 thinks by
        # default and a 64-token generation is ~45 thinking tokens —
        # minutes on a 4-vCPU CI runner, blowing the 300 s request cap
        # (runs 33093698028, 33104419835); mellea maps it to
        # reasoning_effort="none" for the /v1 endpoint.
        model_options={
            ModelOption.THINKING: False,
            "timeout": 300.0,
            "num_retries": 0,
            "max_tokens": 64,
        },
    )
    ctx = SimpleContext()
    ctx = ctx.add(Message(role="user", content="Say 'hello' and nothing else"))

    model_options = {ModelOption.STREAM: True} if stream else {}
    mot, _ = await backend.generate_from_context(
        Message(role="assistant", content=""), ctx, model_options=model_options
    )

    # For streaming, consume the stream fully before checking metrics
    if stream:
        # The 120 s per-chunk stream guard does not bound total request time:
        # a stalled or queued stream that keeps the HTTP connection alive
        # (observed on CI: a 900 s /v1 stream, run 33015176815) keeps the
        # per-chunk guard re-armed and rides to the 900 s pytest watchdog,
        # killing the job. Bound the stream to the same 300 s the
        # non-streaming paths use so a genuine stall surfaces as a bounded
        # TimeoutError the flaky marker retries, matching non-streaming
        # behaviour.
        await asyncio.wait_for(mot.astream(), timeout=300.0)
    # astream() returns as soon as its queue drains, so a long stream is
    # finished by avalue(); keep that inside the same 300 s budget (run
    # 33048969379: a 15 m stream escaped the astream bound above).
    await asyncio.wait_for(mot.avalue(), timeout=300.0)

    await drain_background_tasks()
    provider.force_flush()
    metrics_data = metric_reader.get_metrics_data()

    input_tokens = _token_sum(metrics_data, "litellm", "input")

    output_tokens = _token_sum(metrics_data, "litellm", "output")

    # LiteLLM with Ollama backend should always provide token counts
    assert input_tokens is not None, "Input tokens should be recorded"
    assert input_tokens > 0, f"Input tokens should be > 0, got {input_tokens}"

    assert output_tokens is not None, "Output tokens should be recorded"
    assert output_tokens > 0, f"Output tokens should be > 0, got {output_tokens}"

    # Verify latency metrics
    duration_dp = _find_histogram_data_point(
        metrics_data,
        "gen_ai.client.operation.duration",
        {"gen_ai.request.stream": stream},
    )
    assert duration_dp is not None, "Request duration should be recorded"
    assert duration_dp.sum > 0, "Request duration should be > 0"

    if stream:
        ttfb_dp = _find_histogram_data_point(
            metrics_data, "gen_ai.client.operation.time_to_first_chunk"
        )
        assert ttfb_dp is not None, "TTFB should be recorded for streaming requests"
        assert ttfb_dp.sum > 0, "TTFB should be > 0"


@pytest.mark.asyncio
@pytest.mark.huggingface
@require_gpu(min_vram_gb=8)
@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
async def test_huggingface_token_metrics_integration(
    enable_metrics, metric_reader, stream, hf_metrics_backend
):
    """Test that HuggingFace backend records token metrics correctly."""
    from mellea.backends.model_options import ModelOption
    from mellea.telemetry import metrics as metrics_module

    provider = _setup_metrics_provider(metrics_module, metric_reader)

    ctx = SimpleContext()
    ctx = ctx.add(Message(role="user", content="Say 'hello' and nothing else"))

    model_options = {ModelOption.STREAM: True} if stream else {}
    mot, _ = await hf_metrics_backend.generate_from_context(
        Message(role="assistant", content=""), ctx, model_options=model_options
    )

    # For streaming, consume the stream fully before checking metrics
    if stream:
        # The 120 s per-chunk stream guard does not bound total request time:
        # a stalled or queued stream that keeps the HTTP connection alive
        # (observed on CI: a 900 s /v1 stream, run 33015176815) keeps the
        # per-chunk guard re-armed and rides to the 900 s pytest watchdog,
        # killing the job. Bound the stream to the same 300 s the
        # non-streaming paths use so a genuine stall surfaces as a bounded
        # TimeoutError the flaky marker retries, matching non-streaming
        # behaviour.
        await asyncio.wait_for(mot.astream(), timeout=300.0)
    # astream() returns as soon as its queue drains, so a long stream is
    # finished by avalue(); keep that inside the same 300 s budget (run
    # 33048969379: a 15 m stream escaped the astream bound above).
    await asyncio.wait_for(mot.avalue(), timeout=300.0)

    await drain_background_tasks()
    provider.force_flush()
    metrics_data = metric_reader.get_metrics_data()

    # HuggingFace computes token counts locally
    input_tokens = _token_sum(metrics_data, "huggingface", "input")

    output_tokens = _token_sum(metrics_data, "huggingface", "output")

    assert input_tokens is not None, "Input tokens should be recorded"
    assert input_tokens > 0, f"Input tokens should be > 0, got {input_tokens}"

    assert output_tokens is not None, "Output tokens should be recorded"
    assert output_tokens > 0, f"Output tokens should be > 0, got {output_tokens}"


@pytest.mark.asyncio
@pytest.mark.openai
@pytest.mark.ollama
async def test_error_metrics_on_backend_failure(enable_metrics, metric_reader):
    """Test that error metrics are recorded when a backend call fails.

    Uses OpenAI backend pointed at Ollama with a non-existent model so the
    error fires during generation (through base.py:astream), which is where
    GENERATION_ERROR is triggered.  Also verifies that model/provider are
    correctly populated in the error counter attributes (proving the early
    output.model/provider set in generate_from_context works).
    """
    from mellea.backends.openai import OpenAIBackend
    from mellea.telemetry import metrics as metrics_module

    provider = _setup_metrics_provider(metrics_module, metric_reader)

    backend = OpenAIBackend(
        model_id="nonexistent-model-xyz",
        base_url=f"http://{os.environ.get('OLLAMA_HOST', 'localhost:11434')}/v1",
        api_key="dummy",
    )
    ctx = SimpleContext()
    ctx = ctx.add(Message(role="user", content="Say hello"))

    mot, _ = await backend.generate_from_context(
        Message(role="assistant", content=""), ctx, model_options={}
    )

    # avalue() drives astream(), where the backend call fails, GENERATION_ERROR
    # fires, and the exception is re-raised. pytest.raises catches that re-raise.
    with pytest.raises(Exception):
        await mot.avalue()

    await drain_background_tasks()
    provider.force_flush()
    metrics_data = metric_reader.get_metrics_data()

    error_count = get_metric_value(
        metrics_data,
        "mellea.llm.errors",
        {
            "gen_ai.provider.name": "openai",
            "gen_ai.request.model": "nonexistent-model-xyz",
        },
    )

    assert error_count is not None, "Error counter should have been recorded"
    assert error_count == 1, f"Expected 1 error, got {error_count}"


@pytest.mark.asyncio
@pytest.mark.ollama
async def test_ollama_sampling_metrics_integration(enable_metrics, metric_reader):
    """Test that sampling metrics are recorded through a full RejectionSamplingStrategy loop."""
    from mellea.backends.model_options import ModelOption
    from mellea.backends.ollama import OllamaModelBackend
    from mellea.stdlib.components import Instruction
    from mellea.stdlib.context import SimpleContext
    from mellea.stdlib.sampling import RejectionSamplingStrategy
    from mellea.telemetry import metrics as metrics_module

    provider = _setup_metrics_provider(metrics_module, metric_reader)

    backend = OllamaModelBackend(  # type: ignore
        model_id=IBM_GRANITE_4_2_3B.ollama_name,
        # Same output bound as the other live "say hello" tests.
        model_options={ModelOption.THINKING: False, ModelOption.MAX_NEW_TOKENS: 64},
    )
    strategy = RejectionSamplingStrategy(loop_budget=1)
    ctx = SimpleContext()

    result = await strategy.sample(
        action=Instruction("Say hello"), context=ctx, backend=backend, requirements=None
    )

    await drain_background_tasks()
    provider.force_flush()
    metrics_data = metric_reader.get_metrics_data()

    attempts = get_metric_value(
        metrics_data,
        "mellea.sampling.attempts",
        {"strategy": "RejectionSamplingStrategy"},
    )
    assert attempts is not None, "Sampling attempts should be recorded"
    assert attempts >= 1, f"Expected at least 1 attempt, got {attempts}"

    # With no requirements and loop_budget=1 the loop always succeeds
    successes = get_metric_value(
        metrics_data,
        "mellea.sampling.successes",
        {"strategy": "RejectionSamplingStrategy"},
    )
    assert result.success
    assert successes == 1, f"Expected 1 success, got {successes}"


@pytest.mark.asyncio
@pytest.mark.ollama
async def test_ollama_generate_from_raw_metrics_integration(
    enable_metrics, metric_reader
):
    """Token and latency metrics are recorded for `generate_from_raw` calls."""
    from mellea.backends.model_options import ModelOption
    from mellea.backends.ollama import OllamaModelBackend
    from mellea.core import CBlock
    from mellea.stdlib.context import SimpleContext
    from mellea.telemetry import metrics as metrics_module

    provider = _setup_metrics_provider(metrics_module, metric_reader)

    backend = OllamaModelBackend(  # type: ignore
        model_id=IBM_GRANITE_4_2_3B.ollama_name,
        # Same output bound as the other live "say hello" tests.
        model_options={ModelOption.THINKING: False, ModelOption.MAX_NEW_TOKENS: 64},
    )
    ctx = SimpleContext()
    actions = [CBlock("Say 'hi' and nothing else."), CBlock("Say 'hello'.")]

    results = await backend.generate_from_raw(actions, ctx=ctx)
    assert len(results) == len(actions)

    await drain_background_tasks()
    provider.force_flush()
    metrics_data = metric_reader.get_metrics_data()

    input_tokens = _token_sum(metrics_data, "ollama", "input")
    output_tokens = _token_sum(metrics_data, "ollama", "output")
    assert input_tokens is not None and input_tokens > 0
    assert output_tokens is not None and output_tokens > 0

    duration_dp = _find_histogram_data_point(
        metrics_data,
        "gen_ai.client.operation.duration",
        {"gen_ai.provider.name": "ollama", "gen_ai.request.stream": False},
    )
    assert duration_dp is not None, "Request duration should be recorded for batch"
    assert duration_dp.sum > 0


@pytest.mark.asyncio
@pytest.mark.openai
@pytest.mark.ollama
async def test_generate_from_raw_error_metrics_integration(
    enable_metrics, metric_reader
):
    """Test that error metrics are recorded when `generate_from_raw` fails."""
    from mellea.backends.openai import OpenAIBackend
    from mellea.core import CBlock
    from mellea.stdlib.context import SimpleContext
    from mellea.telemetry import metrics as metrics_module

    provider = _setup_metrics_provider(metrics_module, metric_reader)

    backend = OpenAIBackend(
        model_id="nonexistent-model-xyz",
        base_url=f"http://{os.environ.get('OLLAMA_HOST', 'localhost:11434')}/v1",
        api_key="dummy",
    )
    ctx = SimpleContext()
    actions = [CBlock("Say 'hi'.")]

    with pytest.raises(Exception):
        await backend.generate_from_raw(actions, ctx=ctx)

    await drain_background_tasks()
    provider.force_flush()
    metrics_data = metric_reader.get_metrics_data()

    error_count = get_metric_value(
        metrics_data,
        "mellea.llm.errors",
        {
            "gen_ai.provider.name": "openai",
            "gen_ai.request.model": "nonexistent-model-xyz",
        },
    )
    assert error_count is not None, "Error counter should have been recorded"
    assert error_count == 1, f"Expected 1 error, got {error_count}"
