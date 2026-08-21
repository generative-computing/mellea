# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for adapter-function tracing — the `adapter_function` span tree (#1466).

Covers prepare/activate/deactivate only, matching #1466's scope; generate/parse
have no firing hooks yet (blocked on #1465).

Three layers, mirroring `test_tracing_application.py`:

1. Helper-level unit tests (mock tracer): pin attribute shapes and key
   derivation in `tracing.py`'s `start_adapter_function_span`/
   `finish_adapter_function_span`/`start_adapter_function_phase_span`/
   `finish_adapter_function_phase_span`.
2. Plugin unit tests (mock tracer): `AdapterFunctionTracingPlugin`'s hooks
   translate payload fields into span opens/closes.
3. Integration tests (real OTel SDK, in-memory exporter, real
   `AdapterMixin.adapter_scope`/`LocalFileBinding.prepare` call sites): verify
   the full `adapter_function` > `adapter_function.<phase>` nesting, the
   dangling-child-span cleanup on a raised phase, and that the in-flight span
   registry drains to zero.
"""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip(
    "opentelemetry", reason="opentelemetry not installed — install mellea[telemetry]"
)
pytest.importorskip("cpex", reason="cpex not installed — install mellea[hooks]")

from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from mellea.backends.adapters._core import (
    Adapter,
    Identity,
    IOContract,
    LocalFileBinding,
)
from mellea.backends.adapters.adapter import AdapterMixin
from mellea.core import Component
from mellea.telemetry import tracing
from mellea.telemetry.tracing import (
    finish_adapter_function_phase_span,
    finish_adapter_function_span,
    start_adapter_function_phase_span,
    start_adapter_function_span,
)
from mellea.telemetry.tracing_plugins import AdapterFunctionTracingPlugin
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


@pytest.fixture
def adapter_function_plugin():
    return AdapterFunctionTracingPlugin()


def _patch_backend_tracer() -> tuple[MagicMock, MagicMock]:
    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span
    return fake_span, fake_tracer


def _attrs(span: MagicMock) -> dict:
    return {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}


def _spans_by_name(exporter: InMemorySpanExporter) -> dict:
    tracing._tracer_provider.force_flush()  # type: ignore[union-attr]
    return {s.name: s for s in exporter.get_finished_spans()}


# ---------------------------------------------------------------------------
# Helper-level unit tests (mock tracer)
# ---------------------------------------------------------------------------


def test_start_adapter_function_span_stamps_attrs_and_stashes_by_invocation_id(
    enabled_tracing,
):
    fake_span, fake_tracer = _patch_backend_tracer()
    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        start_adapter_function_span(
            "inv-1",
            name="answerability",
            revision="abc123",
            binding_type="local_file",
            adapter_type="lora",
        )

    fake_tracer.start_span.assert_called_once_with("adapter_function")
    assert "inv-1" in tracing._in_flight_spans
    assert tracing._in_flight_spans["inv-1"] == (fake_span, None)
    attrs = _attrs(fake_span)
    assert attrs["mellea.adapter_function.name"] == "answerability"
    assert attrs["mellea.adapter_function.revision"] == "abc123"
    assert attrs["mellea.adapter_function.binding_type"] == "local_file"
    assert attrs["mellea.adapter_function.adapter_type"] == "lora"


def test_start_adapter_function_span_omits_none_revision(enabled_tracing):
    fake_span, fake_tracer = _patch_backend_tracer()
    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        start_adapter_function_span(
            "inv-unpinned",
            name="answerability",
            revision=None,
            binding_type="local_file",
            adapter_type="lora",
        )

    assert "mellea.adapter_function.revision" not in _attrs(fake_span)


def test_finish_adapter_function_span_success_records_outcome(enabled_tracing):
    fake_span, fake_tracer = _patch_backend_tracer()
    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        start_adapter_function_span(
            "inv-2",
            name="answerability",
            revision="r1",
            binding_type="local_file",
            adapter_type="lora",
        )
        finish_adapter_function_span("inv-2", outcome="success", exception=None)

    fake_span.end.assert_called_once()
    assert _attrs(fake_span)["mellea.adapter_function.outcome"] == "success"
    fake_span.record_exception.assert_not_called()
    assert "inv-2" not in tracing._in_flight_spans


def test_finish_adapter_function_span_error_records_exception(enabled_tracing):
    fake_span, fake_tracer = _patch_backend_tracer()
    err = RuntimeError("boom")
    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        start_adapter_function_span(
            "inv-err",
            name="answerability",
            revision="r1",
            binding_type="local_file",
            adapter_type="lora",
        )
        finish_adapter_function_span("inv-err", outcome="error", exception=err)

    fake_span.record_exception.assert_called_once_with(err)
    fake_span.set_status.assert_called_once()
    attrs = _attrs(fake_span)
    assert attrs["mellea.adapter_function.outcome"] == "error"
    assert attrs["error.type"] == "RuntimeError"


def test_finish_adapter_function_span_no_op_when_not_in_flight(enabled_tracing):
    finish_adapter_function_span("never-opened", outcome="success", exception=None)
    assert "never-opened" not in tracing._in_flight_spans


def test_start_adapter_function_phase_span_stamps_phase_and_revision(enabled_tracing):
    fake_span, fake_tracer = _patch_backend_tracer()
    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        start_adapter_function_phase_span("inv-3", "prepare", revision="sha123")

    # No invocation "inv-3" in flight, so no explicit parent context is passed.
    fake_tracer.start_span.assert_called_once_with(
        "adapter_function.prepare", context=None
    )
    attrs = _attrs(fake_span)
    assert attrs["mellea.adapter_function.phase"] == "prepare"
    assert attrs["mellea.adapter_function.revision"] == "sha123"
    # Keyed distinctly from the parent's own `_in_flight_spans` entry.
    assert "inv-3" not in tracing._in_flight_spans
    assert "inv-3:phase:prepare" in tracing._in_flight_spans


def test_phase_span_key_does_not_collide_with_parent_or_other_phases(enabled_tracing):
    fake_tracer = MagicMock()
    fake_tracer.start_span.side_effect = lambda name, context=None: MagicMock(name=name)
    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        start_adapter_function_span(
            "inv-4",
            name="answerability",
            revision=None,
            binding_type="local_file",
            adapter_type="lora",
        )
        start_adapter_function_phase_span("inv-4", "activate")
        start_adapter_function_phase_span("inv-4", "deactivate")

    assert set(tracing._in_flight_spans) == {
        "inv-4",
        "inv-4:phase:activate",
        "inv-4:phase:deactivate",
    }


def test_finish_adapter_function_phase_span_closes_and_removes(enabled_tracing):
    fake_span, fake_tracer = _patch_backend_tracer()
    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        start_adapter_function_phase_span("inv-5", "prepare")
        finish_adapter_function_phase_span("inv-5", "prepare")

    fake_span.end.assert_called_once()
    assert "inv-5:phase:prepare" not in tracing._in_flight_spans


def test_finish_adapter_function_phase_span_no_op_when_not_in_flight(enabled_tracing):
    # Contract: a phase never opened (or already closed) is a silent no-op —
    # this is what lets finish_adapter_function_span's defensive cleanup run
    # unconditionally without double-closing a phase that completed normally.
    finish_adapter_function_phase_span("inv-6", "prepare")

    # The no-op must leave the registry untouched, not merely avoid raising.
    assert tracing._in_flight_spans == {}


def test_finish_adapter_function_span_closes_dangling_phase_span(enabled_tracing):
    """A phase that raised (start fired, complete never did) is closed by the invocation's own finish."""
    fake_tracer = MagicMock()
    fake_parent_span = MagicMock()
    fake_phase_span = MagicMock()
    fake_tracer.start_span.side_effect = [fake_parent_span, fake_phase_span]
    err = RuntimeError("activation failed")

    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        start_adapter_function_span(
            "inv-7",
            name="answerability",
            revision="r1",
            binding_type="local_file",
            adapter_type="lora",
        )
        start_adapter_function_phase_span("inv-7", "activate")
        # No matching finish_adapter_function_phase_span("inv-7", "activate") —
        # the phase itself raised.
        finish_adapter_function_span("inv-7", outcome="error", exception=err)

    fake_phase_span.record_exception.assert_called_once_with(err)
    fake_phase_span.set_status.assert_called_once()
    fake_phase_span.end.assert_called_once()
    fake_parent_span.end.assert_called_once()
    assert "inv-7" not in tracing._in_flight_spans
    assert "inv-7:phase:activate" not in tracing._in_flight_spans


def test_helpers_are_silent_when_tracing_disabled(disabled_tracing):
    assert (
        start_adapter_function_span(
            "inv-d",
            name="x",
            revision=None,
            binding_type="local_file",
            adapter_type="lora",
        )
        is None
    )
    assert start_adapter_function_phase_span("inv-d", "prepare") is None
    finish_adapter_function_span("inv-d", outcome="success", exception=None)
    finish_adapter_function_phase_span("inv-d", "prepare")
    assert tracing._in_flight_spans == {}


# ---------------------------------------------------------------------------
# Plugin unit tests (mock tracer)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plugin_invocation_start_opens_parent_span(
    adapter_function_plugin, enabled_tracing
):
    from mellea.plugins.hooks.adapter_function import (
        AdapterFunctionInvocationStartPayload,
    )

    _fake_span, fake_tracer = _patch_backend_tracer()
    payload = AdapterFunctionInvocationStartPayload(
        adapter_function_invocation_id="p-inv-1",
        name="answerability",
        revision="r1",
        binding_type="local_file",
        adapter_type="lora",
    )
    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        await adapter_function_plugin.on_invocation_start(payload, {})

    fake_tracer.start_span.assert_called_once_with("adapter_function")
    assert "p-inv-1" in tracing._in_flight_spans


@pytest.mark.asyncio
async def test_plugin_invocation_complete_closes_parent_span(
    adapter_function_plugin, enabled_tracing
):
    from mellea.plugins.hooks.adapter_function import (
        AdapterFunctionInvocationCompletePayload,
        AdapterFunctionInvocationStartPayload,
    )

    fake_span, fake_tracer = _patch_backend_tracer()
    start_payload = AdapterFunctionInvocationStartPayload(
        adapter_function_invocation_id="p-inv-2",
        name="answerability",
        revision="r1",
        binding_type="local_file",
        adapter_type="lora",
    )
    complete_payload = AdapterFunctionInvocationCompletePayload(
        adapter_function_invocation_id="p-inv-2",
        name="answerability",
        revision="r1",
        binding_type="local_file",
        adapter_type="lora",
        outcome="success",
    )
    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        await adapter_function_plugin.on_invocation_start(start_payload, {})
        await adapter_function_plugin.on_invocation_complete(complete_payload, {})

    fake_span.end.assert_called_once()
    assert "p-inv-2" not in tracing._in_flight_spans


@pytest.mark.asyncio
async def test_plugin_phase_start_and_complete_open_and_close_child_span(
    adapter_function_plugin, enabled_tracing
):
    from mellea.plugins.hooks.adapter_function import (
        AdapterFunctionPhaseCompletePayload,
        AdapterFunctionPhaseStartPayload,
    )

    fake_span, fake_tracer = _patch_backend_tracer()
    start_payload = AdapterFunctionPhaseStartPayload(
        adapter_function_invocation_id="p-inv-3",
        name="answerability",
        phase="prepare",
        revision="sha1",
    )
    complete_payload = AdapterFunctionPhaseCompletePayload(
        adapter_function_invocation_id="p-inv-3",
        name="answerability",
        phase="prepare",
        duration_ms=5.0,
    )
    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        await adapter_function_plugin.on_phase_start(start_payload, {})
        await adapter_function_plugin.on_phase_complete(complete_payload, {})

    # No invocation "p-inv-3" in flight, so no explicit parent context is passed.
    fake_tracer.start_span.assert_called_once_with(
        "adapter_function.prepare", context=None
    )
    fake_span.end.assert_called_once()
    assert "p-inv-3:phase:prepare" not in tracing._in_flight_spans


# ---------------------------------------------------------------------------
# Integration: real call sites + real OTel SDK + real adapter lifecycle
# ---------------------------------------------------------------------------


class _Contract(IOContract):
    def build_prompt(self, **kwargs: object) -> Component:
        raise NotImplementedError

    def parse(self, raw: str) -> dict[str, object]:
        return {}


def _make_scope_adapter():
    weights = MagicMock(spec=LocalFileBinding)
    weights.binding_type = "local_file"
    weights.revision = "abc123"
    weights.resolved_revision.return_value = "abc123"
    identity = Identity(name="answerability", adapter_type="lora")
    return Adapter(identity=identity, io_contract=_Contract(), weights=weights)


@pytest.mark.integration
def test_activate_deactivate_emits_nested_span_tree(span_exporter):
    """`adapter_scope` emits `adapter_function` > `adapter_function.{activate,deactivate}`.

    Nesting here is via explicit `trace.set_span_in_context` parenting (see
    `start_adapter_function_phase_span`), not ambient-context attach, so unlike
    every other span family in this codebase it holds on Python 3.11 too — no
    `_CONTEXT_ATTACH_SUPPORTED` gating needed.
    """
    adapter = _make_scope_adapter()
    mock_backend = MagicMock(spec=AdapterMixin)

    with AdapterMixin.adapter_scope(mock_backend, adapter):
        pass

    by_name = _spans_by_name(span_exporter)
    assert "adapter_function" in by_name
    assert "adapter_function.activate" in by_name
    assert "adapter_function.deactivate" in by_name

    parent = by_name["adapter_function"]
    activate_span = by_name["adapter_function.activate"]
    deactivate_span = by_name["adapter_function.deactivate"]

    assert parent.parent is None
    assert parent.attributes is not None
    assert parent.attributes.get("mellea.adapter_function.name") == "answerability"
    assert parent.attributes.get("mellea.adapter_function.outcome") == "success"

    assert activate_span.parent is not None
    assert activate_span.parent.span_id == parent.context.span_id
    assert deactivate_span.parent is not None
    assert deactivate_span.parent.span_id == parent.context.span_id

    assert tracing._in_flight_spans == {}


@pytest.mark.integration
def test_activate_raising_still_drains_registry_and_marks_error(span_exporter):
    """A phase that raises still gets its span closed via the invocation's own finish."""
    adapter = _make_scope_adapter()
    mock_backend = MagicMock(spec=AdapterMixin)
    adapter.weights.activate.side_effect = RuntimeError("activation failed")  # type: ignore[union-attr]

    with pytest.raises(RuntimeError, match="activation failed"):
        with AdapterMixin.adapter_scope(mock_backend, adapter):
            pytest.fail("body must not run when activate() raises")

    by_name = _spans_by_name(span_exporter)
    assert "adapter_function" in by_name
    assert "adapter_function.activate" in by_name
    # activate() raised before deactivate ever ran.
    assert "adapter_function.deactivate" not in by_name

    from opentelemetry.trace import StatusCode

    assert by_name["adapter_function"].status.status_code == StatusCode.ERROR
    assert by_name["adapter_function.activate"].status.status_code == StatusCode.ERROR

    # The registry drains to zero even though the phase never fired its own
    # completion hook — finish_adapter_function_span's defensive cleanup closed it.
    assert tracing._in_flight_spans == {}


@pytest.mark.integration
def test_prepare_raising_drains_registry_and_marks_error(span_exporter):
    """A failing `prepare()` still closes both spans and drains the registry.

    `prepare()`'s own invocation-start/complete pair (unlike `adapter_scope`'s)
    has never been exercised at the span level before — only via mocked hook
    counts in `test_local_file_binding.py`. This is the direct span-level
    counterpart to `test_activate_raising_still_drains_registry_and_marks_error`.
    """
    from opentelemetry.trace import StatusCode

    backend = MagicMock()
    backend.add_adapter.side_effect = lambda binding: setattr(
        binding, "backend", backend
    )
    backend.load_peft_adapter.side_effect = RuntimeError("load boom")
    binding = LocalFileBinding(name="answerability")
    binding.bind_backend(backend)

    with pytest.raises(RuntimeError, match="load boom"):
        binding.prepare()

    by_name = _spans_by_name(span_exporter)
    assert "adapter_function" in by_name
    assert "adapter_function.prepare" in by_name
    assert by_name["adapter_function"].status.status_code == StatusCode.ERROR
    assert by_name["adapter_function.prepare"].status.status_code == StatusCode.ERROR
    assert tracing._in_flight_spans == {}


@pytest.mark.integration
def test_prepare_emits_its_own_invocation_and_records_resolved_revision(span_exporter):
    """`LocalFileBinding.prepare()` opens its own `adapter_function` invocation.

    `adapter_function.prepare` records the resolved catalogue revision, not the
    unresolved `None` a lazily-pinned binding starts with — regression coverage
    for the moved-from-#1141 acceptance criterion ("not 'main'").
    """
    backend = MagicMock()
    backend.add_adapter.side_effect = lambda binding: setattr(
        binding, "backend", backend
    )
    binding = LocalFileBinding(name="answerability")  # revision=None, lazily resolved
    binding.bind_backend(backend)
    assert binding.revision is None

    binding.prepare()

    by_name = _spans_by_name(span_exporter)
    assert "adapter_function" in by_name
    assert "adapter_function.prepare" in by_name

    parent = by_name["adapter_function"]
    prepare_span = by_name["adapter_function.prepare"]

    resolved = binding.resolved_revision()
    assert resolved != "main"
    assert parent.attributes is not None
    assert parent.attributes.get("mellea.adapter_function.revision") == resolved
    assert prepare_span.attributes is not None
    assert prepare_span.attributes.get("mellea.adapter_function.revision") == resolved
    assert prepare_span.attributes.get("mellea.adapter_function.phase") == "prepare"

    assert prepare_span.parent is not None
    assert prepare_span.parent.span_id == parent.context.span_id

    assert tracing._in_flight_spans == {}


@pytest.mark.integration
def test_prepare_and_activate_open_independent_invocations(span_exporter):
    """`prepare()` and the later `adapter_scope()` call are separate invocations, not nested.

    `prepare()` typically runs once at setup, well before any `adapter_scope`
    call — they don't share a parent `adapter_function` span in this
    architecture; each gets its own.
    """
    backend = MagicMock()
    backend.add_adapter.side_effect = lambda binding: setattr(
        binding, "backend", backend
    )
    binding = LocalFileBinding(name="answerability")
    binding.bind_backend(backend)
    binding.prepare()

    adapter = Adapter(
        identity=Identity(name="answerability", adapter_type="lora"),
        io_contract=_Contract(),
        weights=binding,
    )
    mock_backend = MagicMock(spec=AdapterMixin)
    with AdapterMixin.adapter_scope(mock_backend, adapter):
        pass

    tracing._tracer_provider.force_flush()  # type: ignore[union-attr]
    parents = [
        s for s in span_exporter.get_finished_spans() if s.name == "adapter_function"
    ]
    assert len(parents) == 2
    assert parents[0].context.span_id != parents[1].context.span_id
    assert tracing._in_flight_spans == {}
