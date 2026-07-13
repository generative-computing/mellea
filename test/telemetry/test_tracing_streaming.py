"""Tests for the `stream_with_chunking` tracing span."""

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
            chunking_strategy="SentenceChunker",
        )

    fake_tracer.start_span.assert_called_once_with("stream_with_chunking")
    assert "sid-1" in tracing._in_flight_spans
    attrs = _attrs(fake_span)
    assert attrs["mellea.has_requirements"] is True
    assert attrs["mellea.requirement_count"] == 2
    assert attrs["mellea.chunking_strategy"] == "SentenceChunker"
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
    assert attrs["mellea.full_text_length"] == 11
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


def test_reattach_span_attaches_and_releases(enabled_tracing):
    from mellea.telemetry.tracing import reattach_span, release_reattached_span

    _, fake_tracer = _patch_app_tracer()
    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        start_streaming_span(
            "sid-r", has_requirements=False, requirement_count=0, chunking_strategy="x"
        )
        reattach_span("sid-r")
        # A token is held while the span is re-attached.
        assert "sid-r" in tracing._reattached_tokens
        release_reattached_span("sid-r")
        # The token is gone after release.
        assert "sid-r" not in tracing._reattached_tokens


def test_reattach_span_noop_when_not_in_flight(enabled_tracing):
    from mellea.telemetry.tracing import reattach_span, release_reattached_span

    # No matching in-flight span — both calls must be silent no-ops.
    reattach_span("missing")
    release_reattached_span("missing")
    assert not tracing._reattached_tokens


@pytest.mark.asyncio
async def test_cross_task_detach_outside_reattached_scope_warns_and_runs(
    enabled_tracing, caplog
):
    """A cross-task detach with no reattached scope warns and still runs the detach.

    Without a reattach scope on the current task the mismatch is unexpected, so
    the detach is left to run (OTel logs its own ERROR) after a mellea warning.
    """
    import logging

    from mellea.telemetry.tracing import finish_backend_span_success, start_backend_span

    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span

    with (
        patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer),
        patch("mellea.telemetry.tracing.otel_context.detach") as detach,
    ):
        # Attach on this (caller) task.
        start_backend_span("chat", "gid-x", model="m", provider="p")

        async def _finish_on_other_task() -> None:
            finish_backend_span_success(
                "gid-x", operation="chat", usage=None, mot=None, gen=None
            )

        with caplog.at_level(logging.WARNING, logger="mellea"):
            await asyncio.create_task(_finish_on_other_task())

    detach.assert_called_once()
    fake_span.end.assert_called_once()
    assert "gid-x" not in tracing._in_flight_spans
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("across asyncio tasks" in r.getMessage() for r in warnings), (
        "expected a warning naming the cross-task detach"
    )


@pytest.mark.asyncio
async def test_cross_task_detach_inside_reattached_scope_is_debug(
    enabled_tracing, caplog
):
    """A cross-task detach inside a reattached scope is skipped quietly at debug.

    The `stream_with_chunking` case: the orchestration task re-attaches the
    streaming span, then finishes the caller-attached `chat` span. The chat
    token's doomed detach is skipped (only the reattach token is detached, by
    release) and no warning is logged.
    """
    import logging

    from mellea.telemetry.tracing import (
        finish_backend_span_success,
        reattach_span,
        release_reattached_span,
        start_backend_span,
        start_streaming_span,
    )

    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span

    with (
        patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer),
        patch(
            "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
        ),
    ):
        start_streaming_span(
            "sid-x", has_requirements=False, requirement_count=0, chunking_strategy="x"
        )
        # Attach the chat span on this (caller) task; capture its doomed token.
        start_backend_span("chat", "gid-x", model="m", provider="p")
        chat_token = tracing._in_flight_spans["gid-x"][1]

        async def _finish_on_other_task() -> None:
            reattach_span("sid-x")
            try:
                with patch("mellea.telemetry.tracing.otel_context.detach") as detach:
                    finish_backend_span_success(
                        "gid-x", operation="chat", usage=None, mot=None, gen=None
                    )
                # The chat token's cross-task detach was skipped entirely.
                assert chat_token not in [c.args[0] for c in detach.call_args_list]
            finally:
                release_reattached_span("sid-x")

        with caplog.at_level(logging.DEBUG, logger="mellea"):
            await asyncio.create_task(_finish_on_other_task())

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert not warnings, f"unexpected warning(s): {[r.getMessage() for r in warnings]}"
    debug = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("reattached-span scope" in r.getMessage() for r in debug), (
        "expected a debug line for the expected cross-task detach"
    )


@pytest.mark.asyncio
async def test_cross_task_detach_warns_when_scope_belongs_to_another_task(
    enabled_tracing, caplog
):
    """A reattach scope on a different task does not mark this task's detach expected.

    Guards the per-task classifier: a concurrent run's open scope (task A) must
    not silence an unrelated cross-task detach finishing on task B.
    """
    import logging

    from mellea.telemetry.tracing import (
        finish_backend_span_success,
        reattach_span,
        release_reattached_span,
        start_backend_span,
        start_streaming_span,
    )

    _, fake_tracer = _patch_app_tracer()
    release_a = asyncio.Event()

    with (
        patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer),
        patch(
            "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
        ),
    ):
        start_streaming_span(
            "sid-a", has_requirements=False, requirement_count=0, chunking_strategy="x"
        )
        start_backend_span("chat", "gid-b", model="m", provider="p")

        async def _task_a_holds_scope() -> None:
            reattach_span("sid-a")
            try:
                await release_a.wait()
            finally:
                release_reattached_span("sid-a")

        async def _task_b_finishes_span() -> None:
            finish_backend_span_success(
                "gid-b", operation="chat", usage=None, mot=None, gen=None
            )

        task_a = asyncio.create_task(_task_a_holds_scope())
        await asyncio.sleep(0)  # let task A open its scope
        with caplog.at_level(logging.WARNING, logger="mellea"):
            await asyncio.create_task(_task_b_finishes_span())
        release_a.set()
        await task_a

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("across asyncio tasks" in r.getMessage() for r in warnings), (
        "task B's detach should warn despite task A holding a scope"
    )


def test_safe_detach_runs_when_no_attach_task(enabled_tracing):
    """With no recorded attach task the detach runs unconditionally (no task check)."""
    from mellea.telemetry.tracing import _safe_detach

    with patch("mellea.telemetry.tracing.otel_context.detach") as detach:
        _safe_detach(MagicMock(), None)

    detach.assert_called_once()


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

    The mocked `chat` serves both call shapes `stream_with_chunking` triggers:
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
    from mellea.stdlib.streaming import stream_with_chunking

    ctx = SimpleContext().add(Message(role="user", content="Count to three."))
    result = await stream_with_chunking(
        Message(role="assistant", content=""), backend, ctx, requirements=requirements
    )
    async for _ in result.astream():
        pass
    await result.acomplete()
    return result


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stream_with_chunking_emits_span_with_lifecycle_events(span_exporter):
    """A `stream_with_chunking` call emits one span carrying its events."""
    gen = _streaming_backend(["One.", " Two.", " Three."])
    backend = next(gen)
    try:
        await _run_streaming(backend)
    finally:
        gen.close()

    spans = _finished_spans(span_exporter)
    streaming_span = next((s for s in spans if s.name == "stream_with_chunking"), None)
    assert streaming_span is not None, "stream_with_chunking span not emitted"

    event_names = [e.name for e in streaming_span.events]
    assert "chunk" in event_names
    assert "streaming_done" in event_names
    assert "completed" in event_names


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stream_with_chunking_chat_span_nests_under_streaming_span(span_exporter):
    """The backend `chat` span nests under the `stream_with_chunking` span."""
    gen = _streaming_backend(["One.", " Two."])
    backend = next(gen)
    try:
        await _run_streaming(backend)
    finally:
        gen.close()

    spans = _finished_spans(span_exporter)
    streaming_span = next((s for s in spans if s.name == "stream_with_chunking"), None)
    assert streaming_span is not None, "stream_with_chunking span not emitted"
    chat_span = next((s for s in spans if s.name == "chat"), None)
    assert chat_span is not None, "chat span not emitted"

    assert streaming_span.parent is None, "streaming span should be a root"
    if _CONTEXT_ATTACH_SUPPORTED:
        assert chat_span.parent is not None
        assert chat_span.parent.span_id == streaming_span.context.span_id, (
            "chat span should nest under stream_with_chunking"
        )
    else:
        assert chat_span.parent is None, "chat span should be flat on Python <=3.11"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stream_with_chunking_validation_chat_span_is_sibling_of_generation(
    span_exporter,
):
    """Both `chat` spans parent under `stream_with_chunking`, which owns the events."""
    from mellea.core.requirement import Requirement

    gen = _streaming_backend(["A full sentence here."])
    backend = next(gen)
    try:
        await _run_streaming(backend, requirements=[Requirement("Be friendly.")])
    finally:
        gen.close()

    spans = _finished_spans(span_exporter)
    streaming_span = next((s for s in spans if s.name == "stream_with_chunking"), None)
    assert streaming_span is not None, "stream_with_chunking span not emitted"
    chat_spans = [s for s in spans if s.name == "chat"]
    assert len(chat_spans) == 2, f"expected 2 chat spans, got {len(chat_spans)}"

    streaming_id = streaming_span.context.span_id
    if _CONTEXT_ATTACH_SUPPORTED:
        assert all(
            s.parent is not None and s.parent.span_id == streaming_id
            for s in chat_spans
        ), "both chat spans should nest directly under stream_with_chunking"
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
async def test_stream_with_chunking_span_ends_error_on_early_exit(
    span_exporter, caplog
):
    """A mid-stream validation fail still closes the streaming span (ERROR), quietly."""
    import logging

    from opentelemetry.trace import StatusCode

    from mellea.core.requirement import PartialValidationResult, Requirement

    class _FailingReq(Requirement):
        async def stream_validate(self, chunk, *, backend, ctx):
            return PartialValidationResult("fail", reason="nope")

    gen = _streaming_backend(["A full sentence here."])
    backend = next(gen)
    try:
        with caplog.at_level(logging.WARNING, logger="mellea"):
            await _run_streaming(backend, requirements=[_FailingReq()])
    finally:
        gen.close()

    streaming_span = next(
        (s for s in _finished_spans(span_exporter) if s.name == "stream_with_chunking"),
        None,
    )
    assert streaming_span is not None, "stream_with_chunking span not emitted"
    assert streaming_span.status.status_code == StatusCode.ERROR
    cross_task = [r for r in caplog.records if "across asyncio tasks" in r.getMessage()]
    assert not cross_task, "early exit should not warn about cross-task detach"
