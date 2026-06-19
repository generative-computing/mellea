"""Integration tests for streaming span tracing with a mocked Ollama client.

Verifies that the TracingPlugin correctly opens and closes a span during
streaming without requiring a live Ollama server.  This complements
``test_tracing_backend.py::test_streaming_span_duration``, which tests
real-model streaming but is marked ``slow`` and excluded from standard CI.
"""

import asyncio
from unittest.mock import MagicMock, patch

import ollama
import pytest

from mellea.backends.model_options import ModelOption
from mellea.backends.ollama import OllamaModelBackend
from mellea.plugins.manager import (
    disable_background_collection,
    discard_background_tasks,
    drain_background_tasks,
    enable_background_collection,
)
from mellea.stdlib.components import Message
from mellea.stdlib.context import SimpleContext
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
    pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not installed"),
    pytest.mark.integration,
]


@pytest.fixture(scope="module", autouse=True)
def setup_telemetry():
    """Enable tracing for all tests in this module."""
    mp = pytest.MonkeyPatch()
    mp.setenv("MELLEA_TRACES_ENABLED", "true")
    reset_tracing_state()

    yield

    mp.undo()
    reset_tracing_state()


@pytest.fixture
def span_exporter():
    """Create an in-memory span exporter attached to the tracing module's provider."""
    from mellea.telemetry import tracing

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
async def test_streaming_span_creates_and_closes_span(span_exporter):
    """Streaming backend call creates a chat span that closes after the stream completes.

    Uses a mocked Ollama client so no server is needed.  Verifies the core
    TracingPlugin invariant: the span must remain open for the full duration of
    streaming and close only once all chunks are consumed.
    """

    async def fake_chat_stream(*args, **kwargs):
        for content in ["1", " 2", " 3"]:
            await asyncio.sleep(0.05)
            yield ollama.ChatResponse(
                model="test-model",
                created_at=None,
                message=ollama.Message(role="assistant", content=content),
                done=False,
            )
        yield ollama.ChatResponse(
            model="test-model",
            created_at=None,
            message=ollama.Message(role="assistant", content=""),
            done=True,
            eval_count=10,
            prompt_eval_count=5,
        )

    with (
        patch.object(OllamaModelBackend, "_check_ollama_server", return_value=True),
        patch.object(OllamaModelBackend, "_pull_ollama_model", return_value=True),
        patch("mellea.backends.ollama.ollama.Client"),
        patch("mellea.backends.ollama.ollama.AsyncClient") as mock_async_client_cls,
    ):
        mock_async_instance = MagicMock()
        mock_async_instance.chat.side_effect = fake_chat_stream
        mock_async_client_cls.return_value = mock_async_instance

        backend = OllamaModelBackend(model_id="test-model")
        ctx = SimpleContext().add(Message(role="user", content="Count to 3"))

        mot, _ = await backend.generate_from_context(
            Message(role="assistant", content=""),
            ctx,
            model_options={ModelOption.STREAM: True},
        )

        await mot.astream()
        await mot.avalue()
        await drain_background_tasks()

    trace.get_tracer_provider().force_flush()

    spans = span_exporter.get_finished_spans()
    backend_span = next((s for s in spans if s.name == "chat"), None)

    assert backend_span is not None, "Backend span not found"
    assert backend_span.end_time > backend_span.start_time, (
        "Span must have nonzero duration"
    )

    span_duration_s = (backend_span.end_time - backend_span.start_time) / 1e9
    # fake stream is 3 chunks x 50 ms ~= 150 ms; >= 0.1 confirms span survived past first chunk
    assert span_duration_s >= 0.1, (
        f"Span closed too early — duration {span_duration_s:.3f}s is shorter than "
        "the streaming delay, suggesting the span did not stay open for the full stream"
    )
