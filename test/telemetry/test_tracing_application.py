"""Tests for application-level tracing — session and action spans.

This file covers the application-tracing surface in three layers:

1. Helper-level unit tests (mock tracer): pin attribute shapes, key
   derivation, and content-gating in the typed helpers in `tracing.py`.
2. Plugin-disabled / no-op contracts: confirm helpers stay quiet when
   tracing is off.
3. Integration tests through real call sites (real OTel SDK, in-memory
   exporter, mock backend): verify the full nesting works end-to-end via
   `MelleaSession.__enter__/__exit__` and `m.act(...)`.
"""

import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip(
    "opentelemetry", reason="opentelemetry not installed — install mellea[telemetry]"
)
pytest.importorskip("cpex", reason="cpex not installed — install mellea[hooks]")

from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from mellea.core.backend import Backend
from mellea.core.base import GenerateLog, ModelOutputThunk, _CallInfo, _GenerationState
from mellea.stdlib.components import Message
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.session import MelleaSession, start_session
from mellea.telemetry import tracing
from mellea.telemetry.tracing import (
    finish_action_span_error,
    finish_action_span_success,
    finish_session_span,
    finish_session_startup_span,
    start_action_span,
    start_session_span,
    start_session_startup_span,
)
from test.telemetry.conftest import reset_tracing_state

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


@pytest.fixture
def span_exporter(enabled_tracing):
    """Attach an in-memory span exporter to the active tracer provider."""
    if tracing._tracer_provider is None:
        pytest.skip("Telemetry not initialized")
    exporter = InMemorySpanExporter()
    tracing._tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield exporter
    exporter.clear()


def _patch_app_tracer() -> tuple[MagicMock, MagicMock]:
    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span
    return fake_span, fake_tracer


def _attrs(span: MagicMock) -> dict:
    return {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}


def _spans_by_name(exporter: InMemorySpanExporter) -> dict[str, Any]:
    tracing._tracer_provider.force_flush()  # type: ignore[union-attr]
    return {s.name: s for s in exporter.get_finished_spans()}


# ---------------------------------------------------------------------------
# Mock backend (no LLM calls)
# ---------------------------------------------------------------------------


class _MockBackend(Backend):
    """Minimal backend that returns a faked ModelOutputThunk — no LLM API calls."""

    model_id = "mock-model"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Discard constructor args; real backends need model_id etc.
        self._model_id: str = "mock-model"
        self._provider: str = "mock-provider"

    async def _generate_from_context(self, action: Any, ctx: Any, **kwargs: Any):
        mot = MagicMock(spec=ModelOutputThunk)
        mot._gen = _GenerationState()
        mot._call = _CallInfo()
        glog = GenerateLog()
        glog.prompt = "mocked formatted prompt"
        mot._generate_log = glog
        mot.parsed_repr = None
        mot._gen.start = datetime.datetime.now()

        async def _avalue() -> str:
            return "mocked output"

        mot.avalue = _avalue
        mot.value = "mocked output"
        return mot, SimpleContext()

    async def _generate_from_raw(self, actions: Any, ctx: Any, **kwargs: Any):
        return [], None


# ---------------------------------------------------------------------------
# Helper-level unit tests (mock tracer)
# ---------------------------------------------------------------------------


def test_start_session_span_stamps_attrs_and_stashes_under_session_id(enabled_tracing):
    fake_span, fake_tracer = _patch_app_tracer()
    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        start_session_span("sess-1", context_type="SimpleContext")

    fake_tracer.start_span.assert_called_once_with("session")
    assert "sess-1" in tracing._in_flight_spans
    attrs = _attrs(fake_span)
    assert attrs["mellea.session_id"] == "sess-1"
    assert attrs["mellea.context_type"] == "SimpleContext"
    # `backend` not passed → no `mellea.backend` attribute.
    assert "mellea.backend" not in attrs


def test_start_session_span_stamps_backend_when_provided(enabled_tracing):
    fake_span, fake_tracer = _patch_app_tracer()
    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        start_session_span("sess-b", context_type="SimpleContext", backend="ollama")

    attrs = _attrs(fake_span)
    assert attrs["mellea.backend"] == "ollama"


def test_start_session_startup_span_stashes_under_suffixed_key(enabled_tracing):
    fake_span, fake_tracer = _patch_app_tracer()
    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        start_session_startup_span(
            "sess-2",
            backend="ollama",
            model_id="granite4.1:3b",
            context_type="SimpleContext",
        )

    fake_tracer.start_span.assert_called_once_with("start_session")
    assert "sess-2" not in tracing._in_flight_spans
    assert "sess-2:startup" in tracing._in_flight_spans
    attrs = _attrs(fake_span)
    assert attrs["mellea.session_id"] == "sess-2"
    assert attrs["mellea.backend"] == "ollama"
    assert attrs["mellea.model_id"] == "granite4.1:3b"
    assert attrs["mellea.context_type"] == "SimpleContext"


def test_finish_session_startup_span_returns_true_when_in_flight(enabled_tracing):
    fake_span, fake_tracer = _patch_app_tracer()
    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        start_session_startup_span(
            "sess-3", backend="ollama", model_id="m", context_type="SimpleContext"
        )
        result = finish_session_startup_span("sess-3")

    assert result is True
    fake_span.end.assert_called_once()
    assert "sess-3:startup" not in tracing._in_flight_spans


def test_finish_session_startup_span_returns_false_when_not_in_flight(enabled_tracing):
    result = finish_session_startup_span("never-opened")
    assert result is False


def test_finish_session_span_marks_error_with_exception(enabled_tracing):
    fake_span, fake_tracer = _patch_app_tracer()
    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        start_session_span("sess-err", context_type="SimpleContext")
        finish_session_span("sess-err", exception=RuntimeError("boom"))

    fake_span.record_exception.assert_called_once()
    fake_span.set_status.assert_called_once()
    fake_span.end.assert_called_once()
    attrs = _attrs(fake_span)
    assert attrs["error.type"] == "RuntimeError"


def test_finish_session_span_no_op_when_not_in_flight(enabled_tracing):
    # Contract: missing key → silent no-op.
    finish_session_span("never-opened")  # should not raise
    assert "never-opened" not in tracing._in_flight_spans


def test_start_action_span_stamps_request_attrs(enabled_tracing):
    fake_span, fake_tracer = _patch_app_tracer()
    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        start_action_span(
            "cid-1",
            action_class_name="Instruction",
            has_requirements=True,
            has_strategy=True,
            strategy_type="RejectionSamplingStrategy",
            has_format=False,
            tool_calls=False,
        )

    fake_tracer.start_span.assert_called_once_with("action")
    assert "cid-1" in tracing._in_flight_spans
    attrs = _attrs(fake_span)
    assert attrs["mellea.action_type"] == "Instruction"
    assert attrs["mellea.has_requirements"] is True
    assert attrs["mellea.has_strategy"] is True
    assert attrs["mellea.strategy_type"] == "RejectionSamplingStrategy"
    assert attrs["mellea.has_format"] is False
    assert attrs["mellea.tool_calls"] is False


def test_finish_action_span_success_records_length_always(enabled_tracing, monkeypatch):
    monkeypatch.delenv("MELLEA_TRACES_CONTENT", raising=False)
    monkeypatch.delenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False
    )
    fake_span, fake_tracer = _patch_app_tracer()
    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        start_action_span(
            "cid-2",
            action_class_name="X",
            has_requirements=False,
            has_strategy=False,
            strategy_type=None,
            has_format=False,
            tool_calls=False,
        )
        finish_action_span_success(
            "cid-2", response_text="hello world", response_length=11
        )

    attrs = _attrs(fake_span)
    assert attrs["mellea.response_length"] == 11
    assert "mellea.response" not in attrs


def test_finish_action_span_success_records_response_when_content_enabled(
    enabled_tracing, monkeypatch
):
    monkeypatch.setenv("MELLEA_TRACES_CONTENT", "true")
    fake_span, fake_tracer = _patch_app_tracer()
    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        start_action_span(
            "cid-3",
            action_class_name="X",
            has_requirements=False,
            has_strategy=False,
            strategy_type=None,
            has_format=False,
            tool_calls=False,
        )
        finish_action_span_success(
            "cid-3", response_text="captured text", response_length=13
        )

    attrs = _attrs(fake_span)
    assert attrs["mellea.response"] == "captured text"
    assert attrs["mellea.response_length"] == 13


def test_finish_action_span_success_truncates_long_response(
    enabled_tracing, monkeypatch
):
    monkeypatch.setenv("MELLEA_TRACES_CONTENT", "true")
    fake_span, fake_tracer = _patch_app_tracer()
    long_text = "a" * 800
    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        start_action_span(
            "cid-4",
            action_class_name="X",
            has_requirements=False,
            has_strategy=False,
            strategy_type=None,
            has_format=False,
            tool_calls=False,
        )
        finish_action_span_success(
            "cid-4", response_text=long_text, response_length=800
        )

    attrs = _attrs(fake_span)
    assert attrs["mellea.response"].endswith("...")
    assert len(attrs["mellea.response"]) == 503  # 500 chars + "..."
    assert attrs["mellea.response_length"] == 800


def test_finish_action_span_error_marks_and_ends(enabled_tracing):
    fake_span, fake_tracer = _patch_app_tracer()
    err = ValueError("nope")
    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        start_action_span(
            "cid-err",
            action_class_name="X",
            has_requirements=False,
            has_strategy=False,
            strategy_type=None,
            has_format=False,
            tool_calls=False,
        )
        finish_action_span_error("cid-err", exception=err)

    fake_span.record_exception.assert_called_once_with(err)
    fake_span.set_status.assert_called_once()
    fake_span.end.assert_called_once()
    attrs = _attrs(fake_span)
    assert attrs["error.type"] == "ValueError"
    assert "cid-err" not in tracing._in_flight_spans


# ---------------------------------------------------------------------------
# No-op contract: stdlib helpers stay quiet when tracing is disabled
# ---------------------------------------------------------------------------


def test_helpers_are_silent_when_tracing_disabled(disabled_tracing):
    """Direct calls to span helpers no-op cleanly when tracing is off."""
    # No tracer is set up; helpers must return None / not raise.
    assert start_session_span("sess-d", context_type="SimpleContext") is None
    assert (
        start_session_startup_span(
            "sess-d", backend="x", model_id="m", context_type="C"
        )
        is None
    )
    assert (
        start_action_span(
            "cid-d",
            action_class_name=None,
            has_requirements=None,
            has_strategy=None,
            strategy_type=None,
            has_format=None,
            tool_calls=None,
        )
        is None
    )
    finish_session_startup_span("sess-d")  # should not raise
    finish_session_span("sess-d")  # should not raise
    finish_action_span_success("cid-d")  # should not raise
    finish_action_span_error("cid-d", exception=RuntimeError("x"))  # should not raise
    assert tracing._in_flight_spans == {}


# ---------------------------------------------------------------------------
# Integration: real call sites + real OTel SDK + mock backend
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_session_with_block_emits_session_action_nesting(span_exporter):
    """`with start_session(...) as m: m.act(...)` emits a properly nested span tree.

    Verifies the load-bearing nesting fix: action span (created inside
    `_run_async_in_thread` Task) nests under session span (created on the
    user thread). Backend (chat) span emission is covered against a real
    LLM in `test_tracing_backend.py`; this mock backend's custom `avalue`
    starts the chat span but doesn't fire `generation_post_call`, so the
    chat span stays open and isn't visible in the exporter.

    `start_session` is a sibling root (it lives inside `start_session()`'s
    body, before `__enter__` opens the long-lived session span — same
    relationship as pre-this-PR).
    """
    with patch(
        "mellea.stdlib.session.backend_name_to_class", return_value=_MockBackend
    ):
        with start_session("ollama") as m:
            m.act(Message(role="user", content="hi"), strategy=None)

    by_name = _spans_by_name(span_exporter)

    assert "session" in by_name
    assert "start_session" in by_name
    assert "action" in by_name

    session_span = by_name["session"]
    startup_span = by_name["start_session"]
    action_span = by_name["action"]

    # Session is a root.
    assert session_span.parent is None
    # start_session is a separate root — opened by start_session() before __enter__.
    assert startup_span.parent is None
    # Action nests under session — the load-bearing assertion that validates
    # one-way contextvar propagation in `_run_async_in_thread`.
    assert action_span.parent is not None
    assert action_span.parent.span_id == session_span.context.span_id
    # mellea.backend on the session span comes from `backend._provider`.
    assert session_span.attributes is not None
    assert session_span.attributes.get("mellea.backend") == "mock-provider"


@pytest.mark.integration
def test_bare_construction_emits_no_session_span(span_exporter):
    """Bare `MelleaSession(...)` (no `with`) doesn't open a session span — pre-this-PR shape preserved."""
    backend = _MockBackend()
    m = MelleaSession(backend, ctx=SimpleContext())
    try:
        m.act(Message(role="user", content="hi"), strategy=None)
    finally:
        m.cleanup()

    by_name = _spans_by_name(span_exporter)

    # No session span (no __enter__ → no start_session_span).
    assert "session" not in by_name
    assert "start_session" not in by_name
    # action still emits (plugin-driven from inside aact).
    assert "action" in by_name


@pytest.mark.integration
def test_session_span_marks_error_on_with_block_exception(span_exporter):
    """Exception inside the `with` block propagates to session span ERROR status."""
    with patch(
        "mellea.stdlib.session.backend_name_to_class", return_value=_MockBackend
    ):
        with pytest.raises(RuntimeError):
            with start_session("ollama"):
                raise RuntimeError("user code failed")

    by_name = _spans_by_name(span_exporter)
    session_span = by_name["session"]

    from opentelemetry.trace import StatusCode

    assert session_span.status.status_code == StatusCode.ERROR
    event_names = [e.name for e in session_span.events]
    assert "exception" in event_names


@pytest.mark.integration
def test_start_session_span_marks_error_on_backend_construction_failure(span_exporter):
    """Backend construction failure during `start_session()` marks the startup span ERROR."""

    class _BoomBackend:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("backend init failed")

    with patch(
        "mellea.stdlib.session.backend_name_to_class", return_value=_BoomBackend
    ):
        with pytest.raises(RuntimeError):
            start_session("ollama")

    by_name = _spans_by_name(span_exporter)
    startup_span = by_name["start_session"]

    from opentelemetry.trace import StatusCode

    assert startup_span.status.status_code == StatusCode.ERROR
    event_names = [e.name for e in startup_span.events]
    assert "exception" in event_names
    assert startup_span.attributes is not None
    assert startup_span.attributes.get("error.type") == "RuntimeError"
    # No long-lived `session` span — __enter__ never ran.
    assert "session" not in by_name
