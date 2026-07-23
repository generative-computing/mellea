# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the `execute_tool` tracing span through real hooks."""

from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip(
    "opentelemetry", reason="opentelemetry not installed — install mellea[telemetry]"
)
pytest.importorskip("cpex", reason="cpex not installed — install mellea[hooks]")

from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from mellea.backends.tools import MelleaTool
from mellea.core.backend import Backend
from mellea.core.base import ModelOutputThunk, ModelToolCall
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.session import MelleaSession
from mellea.telemetry import tracing
from test.telemetry.conftest import reset_tracing_state

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def enabled_tracing(monkeypatch):
    monkeypatch.setenv("MELLEA_TRACES_ENABLED", "true")
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


def _spans_by_name(exporter: InMemorySpanExporter) -> dict[str, Any]:
    tracing._tracer_provider.force_flush()  # type: ignore[union-attr]
    return {s.name: s for s in exporter.get_finished_spans()}


def _tool_call(func: Any, name: str) -> ModelToolCall:
    tool = MelleaTool.from_callable(func, name=name)
    return ModelToolCall(name=name, func=tool, args={})


class _MockBackend(Backend):
    """Minimal non-formatter backend; inference is patched, so it makes no calls."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._model_id = "mock-model"
        self._provider = "mock-provider"

    async def _generate_from_context(self, action: Any, ctx: Any, **kwargs: Any):
        raise NotImplementedError("inference is patched in these tests")

    async def _generate_from_raw(self, actions: Any, ctx: Any, **kwargs: Any):
        raise NotImplementedError("inference is patched in these tests")


# ---------------------------------------------------------------------------
# Integration tests through real hooks + real exporter
# ---------------------------------------------------------------------------


def test_session_tool_calls_emit_parented_spans_per_call(span_exporter):
    """A turn that calls two tools emits one `execute_tool` span per call, each
    parented under the session span, with per-call success/error status.

    Only inference (`act`) is faked; the real `transform` -> `_call_tools` ->
    `tool_*_invoke` hooks -> `ToolTracingPlugin` path runs and emits the spans.
    """

    def _ok() -> str:
        """A tool that succeeds."""
        return "done"

    def _boom() -> str:
        """A tool that raises."""
        raise ValueError("boom")

    mot: ModelOutputThunk = ModelOutputThunk(value="")
    mot.tool_calls = [_tool_call(_ok, "ok"), _tool_call(_boom, "boom")]

    class _Subject:
        """Trivial mify-able object to transform."""

        value = "subject"

    with MelleaSession(_MockBackend(), ctx=SimpleContext()) as m:
        # Patch inference so `transform` returns our scripted tool calls; the
        # downstream tool-execution + span emission is all real.
        with patch("mellea.stdlib.functional.act", return_value=(mot, m.ctx)):
            m.transform(_Subject(), "do the thing")

    by_name = _spans_by_name(span_exporter)

    # 1. Both tool spans emitted (assert existence before parentage/status so a
    #    missing span fails here, not misleadingly on a downstream assertion).
    assert "session" in by_name
    assert "execute_tool ok" in by_name
    assert "execute_tool boom" in by_name

    ok_span = by_name["execute_tool ok"]
    boom_span = by_name["execute_tool boom"]

    # 2. Per-call status.
    assert ok_span.attributes["mellea.tool.status"] == "success"
    assert boom_span.attributes["mellea.tool.status"] == "failure"
    assert boom_span.attributes["error.type"] == "ValueError"
    assert boom_span.status.status_code.name == "ERROR"

    # 3. Parentage: both nest under the session span.
    session_span = by_name["session"]
    assert session_span.parent is None
    assert ok_span.parent is not None
    assert ok_span.parent.span_id == session_span.context.span_id
    assert boom_span.parent is not None
    assert boom_span.parent.span_id == session_span.context.span_id
