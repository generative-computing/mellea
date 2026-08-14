# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `stream` tracing span."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from mellea.telemetry import tracing
from mellea.telemetry.tracing import finish_streaming_span, start_streaming_span
from mellea.telemetry.tracing_plugins import _CONTEXT_ATTACH_SUPPORTED
from test.telemetry.conftest import reset_tracing_state

try:
    import opentelemetry

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not OTEL_AVAILABLE, reason="OpenTelemetry not installed"
)


@pytest.fixture
def enabled_tracing(monkeypatch):
    monkeypatch.setenv("MELLEA_TRACES_ENABLED", "true")
    reset_tracing_state()
    yield
    reset_tracing_state()


@pytest.fixture
def disabled_tracing(monkeypatch):
    monkeypatch.delenv("MELLEA_TRACES_ENABLED", raising=False)
    reset_tracing_state()
    yield
    reset_tracing_state()


def _attrs(span: MagicMock) -> dict:
    return {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}


def _patch_app_tracer() -> tuple[MagicMock, MagicMock]:
    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span
    return fake_span, fake_tracer


def test_start_streaming_span_stamps_attrs_and_stashes_under_id(enabled_tracing):
    fake_span, fake_tracer = _patch_app_tracer()
    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        start_streaming_span(
            "sid-1",
            has_requirements=True,
            requirement_count=2,
            chunking_strategy="SentenceChunking",
        )

    fake_tracer.start_span.assert_called_once_with("stream")
    assert "sid-1" in tracing._in_flight_spans
    attrs = _attrs(fake_span)
    assert attrs["mellea.streaming.has_requirements"] is True
    assert attrs["mellea.streaming.requirement_count"] == 2
    assert attrs["mellea.streaming.chunking_strategy"] == "SentenceChunking"
    # The correlation id is the in-flight key, not a span attribute.
    assert "mellea.streaming_id" not in attrs


def test_finish_streaming_span_success_records_completed_attrs(enabled_tracing):
    fake_span, fake_tracer = _patch_app_tracer()
    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        start_streaming_span(
            "sid-2", has_requirements=False, requirement_count=0, chunking_strategy="x"
        )
        finish_streaming_span(
            "sid-2",
            success=True,
            model="gpt-4o",
            provider="openai",
            full_text_length=11,
        )

    fake_span.end.assert_called_once()
    attrs = _attrs(fake_span)
    assert attrs["mellea.streaming.full_text_length"] == 11
    assert attrs["gen_ai.request.model"] == "gpt-4o"
    assert attrs["gen_ai.provider.name"] == "openai"
    fake_span.record_exception.assert_not_called()
    assert "sid-2" not in tracing._in_flight_spans


def test_finish_streaming_span_validation_fail_marks_error_without_exception(
    enabled_tracing,
):
    fake_span, fake_tracer = _patch_app_tracer()
    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        start_streaming_span(
            "sid-3", has_requirements=True, requirement_count=1, chunking_strategy="x"
        )
        finish_streaming_span("sid-3", success=False, failure_reason="too short")

    fake_span.end.assert_called_once()
    fake_span.set_status.assert_called_once()
    fake_span.record_exception.assert_not_called()


def test_finish_streaming_span_exception_records_exception(enabled_tracing):
    fake_span, fake_tracer = _patch_app_tracer()
    exc = ValueError("boom")
    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        start_streaming_span(
            "sid-4", has_requirements=False, requirement_count=0, chunking_strategy="x"
        )
        finish_streaming_span("sid-4", success=False, exception=exc)

    fake_span.end.assert_called_once()
    fake_span.record_exception.assert_called_once_with(exc)
    assert "sid-4" not in tracing._in_flight_spans


def test_streaming_span_helpers_silent_when_tracing_disabled(disabled_tracing):
    assert (
        start_streaming_span(
            "sid-d", has_requirements=False, requirement_count=0, chunking_strategy="x"
        )
        is None
    )
    finish_streaming_span("sid-d", success=True)  # should not raise


@pytest.fixture
def span_exporter(enabled_tracing):
    """Attach an in-memory span exporter to the active tracer provider."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    if tracing._tracer_provider is None:
        pytest.skip("Telemetry not initialized")
    exporter = InMemorySpanExporter()
    tracing._tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield exporter
    exporter.clear()


def _finished_spans(exporter):
    tracing._tracer_provider.force_flush()
    return exporter.get_finished_spans()


def _streaming_backend(chunks, *, judge_reply="yes"):
    """Build an OllamaModelBackend whose AsyncClient is mocked to stream `chunks`.

    The mocked `chat` serves both call shapes `stream()` triggers:
    a streaming generation (`stream=True` → async iterator of deltas) and a
    non-streaming LLM-as-a-judge `validate()` call (`stream=False` → a single
    awaited `ChatResponse` carrying `judge_reply`).
    """
    import ollama

    from mellea.backends.ollama import OllamaModelBackend

    async def stream_response():
        for content in chunks:
            await asyncio.sleep(0.01)
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

    async def judge_response():
        return ollama.ChatResponse(
            model="test-model",
            created_at=None,
            message=ollama.Message(role="assistant", content=judge_reply),
            done=True,
            eval_count=1,
            prompt_eval_count=1,
        )

    def chat(*args, stream=False, **kwargs):
        return stream_response() if stream else judge_response()

    with (
        patch.object(OllamaModelBackend, "_check_ollama_server", return_value=True),
        patch.object(OllamaModelBackend, "_pull_ollama_model", return_value=True),
        patch("mellea.backends.ollama.ollama.Client"),
        patch("mellea.backends.ollama.ollama.AsyncClient") as mock_async_client_cls,
    ):
        mock_async_instance = MagicMock()
        mock_async_instance.chat.side_effect = chat
        mock_async_client_cls.return_value = mock_async_instance
        yield OllamaModelBackend(model_id="test-model")


async def _run_streaming(backend, *, requirements=None):
    from mellea.stdlib.components import Message
    from mellea.stdlib.context import SimpleContext
    from mellea.stdlib.streaming import stream

    ctx = SimpleContext().add(Message(role="user", content="Count to three."))
    async with await stream(
        Message(role="assistant", content=""), backend, ctx, requirements=requirements
    ) as streamer:
        async for _ in streamer:
            pass
    return streamer


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stream_emits_span_with_lifecycle_events(span_exporter):
    """A `stream` call emits one span carrying its events."""
    gen = _streaming_backend(["One.", " Two.", " Three."])
    backend = next(gen)
    try:
        await _run_streaming(backend)
    finally:
        gen.close()

    spans = _finished_spans(span_exporter)
    streaming_span = next((s for s in spans if s.name == "stream"), None)
    assert streaming_span is not None, "stream span not emitted"

    event_names = [e.name for e in streaming_span.events]
    assert "chunk" in event_names
    assert "streaming_done" in event_names
    assert "completed" in event_names


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stream_chat_span_nests_under_streaming_span(span_exporter):
    """The backend `chat` span nests under the `stream` span."""
    gen = _streaming_backend(["One.", " Two."])
    backend = next(gen)
    try:
        await _run_streaming(backend)
    finally:
        gen.close()

    spans = _finished_spans(span_exporter)
    streaming_span = next((s for s in spans if s.name == "stream"), None)
    assert streaming_span is not None, "stream span not emitted"
    chat_span = next((s for s in spans if s.name == "chat"), None)
    assert chat_span is not None, "chat span not emitted"

    assert streaming_span.parent is None, "streaming span should be a root"
    if _CONTEXT_ATTACH_SUPPORTED:
        assert chat_span.parent is not None
        assert chat_span.parent.span_id == streaming_span.context.span_id, (
            "chat span should nest under stream"
        )
    else:
        assert chat_span.parent is None, "chat span should be flat on Python <=3.11"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stream_validation_chat_span_is_sibling_of_generation(span_exporter):
    """Both `chat` spans parent under `stream`, which owns the events."""
    from mellea.core.requirement import Requirement

    gen = _streaming_backend(["A full sentence here."])
    backend = next(gen)
    try:
        await _run_streaming(backend, requirements=[Requirement("Be friendly.")])
    finally:
        gen.close()

    spans = _finished_spans(span_exporter)
    streaming_span = next((s for s in spans if s.name == "stream"), None)
    assert streaming_span is not None, "stream span not emitted"
    chat_spans = [s for s in spans if s.name == "chat"]
    assert len(chat_spans) == 2, f"expected 2 chat spans, got {len(chat_spans)}"

    streaming_id = streaming_span.context.span_id
    if _CONTEXT_ATTACH_SUPPORTED:
        assert all(
            s.parent is not None and s.parent.span_id == streaming_id
            for s in chat_spans
        ), "both chat spans should nest directly under stream"
    else:
        assert all(s.parent is None for s in chat_spans), (
            "chat spans should be flat on Python <=3.11"
        )

    # Lifecycle events attach to the streaming span, not the chat spans.
    event_names = {e.name for e in streaming_span.events}
    assert event_names >= {
        "quick_check",
        "chunk",
        "streaming_done",
        "full_validation",
        "completed",
    }, f"missing streaming events: {event_names}"
    assert all(not s.events for s in chat_spans), (
        "chat spans should carry no streaming events"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stream_span_ends_error_on_early_exit(span_exporter):
    """A mid-stream validation fail still closes the streaming span (ERROR)."""
    from opentelemetry.trace import StatusCode

    from mellea.core.requirement import PartialValidationResult, Requirement

    class _FailingReq(Requirement):
        async def stream_validate(self, chunk, *, backend, ctx):
            return PartialValidationResult("fail", reason="nope")

    gen = _streaming_backend(["A full sentence here."])
    backend = next(gen)
    try:
        await _run_streaming(backend, requirements=[_FailingReq()])
    finally:
        gen.close()

    streaming_span = next(
        (s for s in _finished_spans(span_exporter) if s.name == "stream"), None
    )
    assert streaming_span is not None, "stream span not emitted"
    assert streaming_span.status.status_code == StatusCode.ERROR
