# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for sampling and validation tracing spans."""

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
from mellea.core.requirement import Requirement, ValidationResult
from mellea.stdlib.components import Instruction
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.sampling.base import RejectionSamplingStrategy
from mellea.stdlib.session import start_session
from mellea.telemetry import tracing
from mellea.telemetry.tracing import (
    add_span_event,
    finish_sampling_span,
    finish_validation_span,
    start_sampling_span,
    start_validation_span,
)
from mellea.telemetry.tracing_plugins import _CONTEXT_ATTACH_SUPPORTED
from test.telemetry.conftest import reset_tracing_state

# ---------------------------------------------------------------------------
# Fixtures / helpers
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


class _MockBackend(Backend):
    """Minimal backend that returns a faked ModelOutputThunk — no LLM API calls."""

    model_id = "mock-model"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._model_id = "mock-model"
        self._provider = "mock-provider"

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


def test_start_sampling_span_stamps_attrs_and_stashes(enabled_tracing):
    fake_span, fake_tracer = _patch_app_tracer()
    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        start_sampling_span(
            "sid-1",
            strategy_type="RejectionSamplingStrategy",
            loop_budget=4,
            requirement_count=2,
        )

    fake_tracer.start_span.assert_called_once_with("sampling")
    assert "sid-1" in tracing._in_flight_spans
    attrs = _attrs(fake_span)
    assert attrs["mellea.strategy_type"] == "RejectionSamplingStrategy"
    assert attrs["mellea.loop_budget"] == 4
    assert attrs["mellea.requirement_count"] == 2


def test_add_span_event_records_event_on_in_flight_span(enabled_tracing):
    fake_span, fake_tracer = _patch_app_tracer()
    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        start_sampling_span(
            "sid-ev", strategy_type="S", loop_budget=1, requirement_count=1
        )
        add_span_event(
            "sid-ev",
            event_name="iteration",
            attributes={"iteration": 1, "all_validations_passed": True, "skip": None},
        )

    fake_span.add_event.assert_called_once()
    name, attrs = fake_span.add_event.call_args.args
    assert name == "iteration"
    # None-valued attributes are filtered out.
    assert attrs == {"iteration": 1, "all_validations_passed": True}


def test_add_span_event_no_op_when_key_missing(enabled_tracing):
    # Contract: unknown key → silent no-op (does not raise).
    add_span_event("never-opened", event_name="iteration", attributes={"x": 1})
    assert "never-opened" not in tracing._in_flight_spans


def test_finish_sampling_span_success_stamps_outcome(enabled_tracing):
    fake_span, fake_tracer = _patch_app_tracer()
    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        start_sampling_span(
            "sid-ok", strategy_type="S", loop_budget=1, requirement_count=1
        )
        finish_sampling_span("sid-ok", success=True, iterations_used=3)

    fake_span.end.assert_called_once()
    assert "sid-ok" not in tracing._in_flight_spans
    attrs = _attrs(fake_span)
    assert attrs["mellea.sampling_success"] is True
    assert attrs["mellea.iterations_used"] == 3


def test_finish_sampling_span_failure_is_not_an_error(enabled_tracing):
    # A budget-exhausted loop (success=False, no exception) is a routine
    # outcome: the span closes with default status, not ERROR.
    fake_span, fake_tracer = _patch_app_tracer()
    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        start_sampling_span(
            "sid-fail", strategy_type="S", loop_budget=2, requirement_count=1
        )
        finish_sampling_span(
            "sid-fail",
            success=False,
            iterations_used=2,
            failure_reason="Budget exhausted after 2 iterations",
        )

    fake_span.record_exception.assert_not_called()
    fake_span.set_status.assert_not_called()
    fake_span.end.assert_called_once()
    attrs = _attrs(fake_span)
    assert attrs["mellea.sampling_success"] is False
    assert attrs["mellea.failure_reason"] == "Budget exhausted after 2 iterations"


def test_finish_sampling_span_exception_sets_error(enabled_tracing):
    # Only a raised loop marks the span ERROR.
    fake_span, fake_tracer = _patch_app_tracer()
    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        start_sampling_span(
            "sid-exc", strategy_type="S", loop_budget=1, requirement_count=1
        )
        finish_sampling_span("sid-exc", success=False, exception=RuntimeError("boom"))

    fake_span.record_exception.assert_called_once()
    fake_span.set_status.assert_called_once()
    fake_span.end.assert_called_once()
    assert "sid-exc" not in tracing._in_flight_spans
    assert _attrs(fake_span)["error.type"] == "RuntimeError"


def test_start_validation_span_stamps_requirement_count(enabled_tracing):
    fake_span, fake_tracer = _patch_app_tracer()
    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        start_validation_span("vid-1", requirement_count=3)

    fake_tracer.start_span.assert_called_once_with("validation")
    assert "vid-1" in tracing._in_flight_spans
    assert _attrs(fake_span)["mellea.requirement_count"] == 3


def test_finish_validation_span_records_counts_always(enabled_tracing, monkeypatch):
    monkeypatch.delenv("MELLEA_TRACES_CONTENT", raising=False)
    monkeypatch.delenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False
    )
    fake_span, fake_tracer = _patch_app_tracer()
    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        start_validation_span("vid-c", requirement_count=2)
        finish_validation_span(
            "vid-c",
            all_validations_passed=False,
            passed_count=1,
            failed_count=1,
            failure_reasons=["constraint not met"],
        )

    fake_span.end.assert_called_once()
    attrs = _attrs(fake_span)
    assert attrs["mellea.validation_passed"] is False
    assert attrs["mellea.passed_count"] == 1
    assert attrs["mellea.failed_count"] == 1
    # Content capture disabled → failure reasons omitted.
    assert "mellea.failure_reasons" not in attrs


def test_finish_validation_span_records_reasons_when_content_enabled(
    enabled_tracing, monkeypatch
):
    monkeypatch.setenv("MELLEA_TRACES_CONTENT", "true")
    fake_span, fake_tracer = _patch_app_tracer()
    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        start_validation_span("vid-r", requirement_count=1)
        finish_validation_span(
            "vid-r",
            all_validations_passed=False,
            passed_count=0,
            failed_count=1,
            failure_reasons=["output too short"],
        )

    assert _attrs(fake_span)["mellea.failure_reasons"] == ["output too short"]


def test_finish_validation_span_records_reasons_as_list(enabled_tracing, monkeypatch):
    # Multiple failing requirements are recorded as a list, one entry each.
    monkeypatch.setenv("MELLEA_TRACES_CONTENT", "true")
    fake_span, fake_tracer = _patch_app_tracer()
    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        start_validation_span("vid-multi", requirement_count=2)
        finish_validation_span(
            "vid-multi",
            all_validations_passed=False,
            passed_count=0,
            failed_count=2,
            failure_reasons=["too short", "wrong tone"],
        )

    assert _attrs(fake_span)["mellea.failure_reasons"] == ["too short", "wrong tone"]


def test_finish_validation_span_exception_marks_error(enabled_tracing):
    fake_span, fake_tracer = _patch_app_tracer()
    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        start_validation_span("vid-exc", requirement_count=1)
        finish_validation_span("vid-exc", exception=RuntimeError("boom"))

    fake_span.record_exception.assert_called_once()
    fake_span.set_status.assert_called_once()
    fake_span.end.assert_called_once()
    assert "vid-exc" not in tracing._in_flight_spans
    assert _attrs(fake_span)["error.type"] == "RuntimeError"


def test_helpers_silent_when_tracing_disabled(disabled_tracing):
    # No tracer → helpers return None / no-op and stash nothing.
    assert (
        start_sampling_span("x", strategy_type="S", loop_budget=1, requirement_count=1)
        is None
    )
    assert start_validation_span("y", requirement_count=1) is None
    finish_sampling_span("x", success=True)
    finish_validation_span("y")
    add_span_event("x", event_name="iteration", attributes={"a": 1})
    assert "x" not in tracing._in_flight_spans
    assert "y" not in tracing._in_flight_spans


# ---------------------------------------------------------------------------
# Integration test (real hooks, mock backend)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_act_with_strategy_emits_sampling_and_validate_nesting(span_exporter):
    """`m.act(strategy=...)` with a requirement emits `action > sampling > validate`.

    Uses a mocked-transport `OllamaModelBackend` so the full generation
    lifecycle fires — the `chat` span opens *and* closes before validation runs,
    so the `validation` span nests directly under `sampling`. Parent-chain asserts
    are gated on `_CONTEXT_ATTACH_SUPPORTED`.
    """
    import ollama

    from mellea.backends.ollama import OllamaModelBackend

    async def fake_chat(*args, **kwargs):
        return ollama.ChatResponse(
            model="test-model",
            created_at=None,
            message=ollama.Message(role="assistant", content="hi there"),
            done=True,
            eval_count=3,
            prompt_eval_count=2,
        )

    mock_client = MagicMock()
    mock_client.chat.side_effect = fake_chat

    passing_req = Requirement(
        "always ok", validation_fn=lambda _ctx: ValidationResult(True)
    )

    with (
        patch.object(OllamaModelBackend, "_check_ollama_server", return_value=True),
        patch.object(OllamaModelBackend, "_pull_ollama_model", return_value=True),
        patch("mellea.backends.ollama.ollama.Client"),
        patch("mellea.backends.ollama.ollama.AsyncClient", return_value=mock_client),
        patch(
            "mellea.stdlib.session.backend_name_to_class",
            return_value=OllamaModelBackend,
        ),
    ):
        with start_session("ollama", model_id="test-model") as m:
            m.act(
                Instruction(description="say hi"),
                requirements=[passing_req],
                strategy=RejectionSamplingStrategy(loop_budget=1),
            )

    by_name = _spans_by_name(span_exporter)
    assert "sampling" in by_name, "sampling span not emitted"
    assert "validation" in by_name, "validation span not emitted"
    assert "action" in by_name

    sampling_span = by_name["sampling"]
    validate_span = by_name["validation"]
    action_span = by_name["action"]

    assert sampling_span.attributes is not None
    assert (
        sampling_span.attributes.get("mellea.strategy_type")
        == "RejectionSamplingStrategy"
    )
    assert sampling_span.attributes.get("mellea.sampling_success") is True

    if _CONTEXT_ATTACH_SUPPORTED:
        assert sampling_span.parent is not None
        assert sampling_span.parent.span_id == action_span.context.span_id, (
            "sampling should nest under action"
        )
        assert validate_span.parent is not None
        assert validate_span.parent.span_id == sampling_span.context.span_id, (
            "validate should nest under sampling"
        )
    else:
        session_span = by_name["session"]
        assert sampling_span.parent is not None
        assert sampling_span.parent.span_id == session_span.context.span_id, (
            "sampling should collapse to session on Python <=3.11"
        )
        assert validate_span.parent is not None
        assert validate_span.parent.span_id == session_span.context.span_id, (
            "validate should collapse to session on Python <=3.11"
        )


@pytest.mark.integration
def test_avalidate_raises_closes_validate_span_error(span_exporter):
    """A validator that raises still closes the `validation` span (ERROR, no leak)."""
    import asyncio

    from mellea.stdlib.functional import avalidate

    def _boom(_ctx):
        raise RuntimeError("validator boom")

    backend = _MockBackend()
    req = Requirement("explodes", validation_fn=_boom)

    with pytest.raises(RuntimeError, match="validator boom"):
        asyncio.run(
            avalidate(req, SimpleContext(), backend, output=ModelOutputThunk(value="x"))
        )

    # Span closed, not leaked, and marked ERROR.
    assert not tracing._in_flight_spans
    by_name = _spans_by_name(span_exporter)
    assert "validation" in by_name
    assert by_name["validation"].status.status_code.name == "ERROR"


@pytest.mark.integration
def test_sample_raises_closes_sampling_span_error(span_exporter):
    """A backend that raises during sampling still closes the `sampling` span."""
    import asyncio

    class _RaisingBackend(_MockBackend):
        async def _generate_from_context(self, action, ctx, **kwargs):
            raise RuntimeError("backend boom")

    with pytest.raises(RuntimeError, match="backend boom"):
        asyncio.run(
            RejectionSamplingStrategy(loop_budget=1).sample(
                Instruction(description="hi"),
                context=SimpleContext(),
                backend=_RaisingBackend(),
                requirements=[],
            )
        )

    assert not tracing._in_flight_spans
    by_name = _spans_by_name(span_exporter)
    assert "sampling" in by_name
    assert by_name["sampling"].status.status_code.name == "ERROR"
