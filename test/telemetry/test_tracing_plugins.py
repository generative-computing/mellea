"""Unit tests for the BackendTracingPlugin span lifecycle."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip(
    "opentelemetry", reason="opentelemetry not installed — install mellea[telemetry]"
)
pytest.importorskip("cpex", reason="cpex not installed — install mellea[hooks]")

from mellea.core.base import GenerationMetadata, ModelOutputThunk
from mellea.core.requirement import PartialValidationResult, ValidationResult
from mellea.plugins.hooks.component import (
    ComponentPostErrorPayload,
    ComponentPostSuccessPayload,
    ComponentPreExecutePayload,
)
from mellea.plugins.hooks.generation import (
    GenerationBatchErrorPayload,
    GenerationBatchPostCallPayload,
    GenerationBatchPreCallPayload,
    GenerationErrorPayload,
    GenerationPostCallPayload,
    GenerationPreCallPayload,
)
from mellea.plugins.hooks.streaming import (
    StreamingEndPayload,
    StreamingEventPayload,
    StreamingStartPayload,
)
from mellea.stdlib.streaming import (
    ChunkEvent,
    ErrorEvent,
    FullValidationEvent,
    QuickCheckEvent,
    StreamingDoneEvent,
)
from mellea.telemetry import tracing
from mellea.telemetry.tracing_plugins import (
    BackendTracingPlugin,
    ComponentTracingPlugin,
    StreamingTracingPlugin,
)
from test.telemetry.conftest import reset_tracing_state


@pytest.fixture
def backend_plugin():
    return BackendTracingPlugin()


@pytest.fixture
def component_plugin():
    return ComponentTracingPlugin()


@pytest.fixture
def streaming_plugin():
    return StreamingTracingPlugin()


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
    """Collect `span.set_attribute(k, v)` calls into a dict."""
    return {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}


# BackendTracingPlugin tests


@pytest.mark.asyncio
async def test_pre_call_starts_span_and_stashes_by_generation_id(
    backend_plugin, enabled_tracing
):
    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span

    payload = GenerationPreCallPayload(action=None, context=None, generation_id="gid-1")

    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        await backend_plugin.on_pre_call(payload, {})

    fake_tracer.start_span.assert_called_once_with("chat")
    assert "gid-1" in tracing._in_flight_spans
    fake_span.end.assert_not_called()
    attrs = _attrs(fake_span)
    assert attrs["gen_ai.operation.name"] == "chat"


@pytest.mark.asyncio
async def test_post_call_finishes_span_with_usage_attrs(
    backend_plugin, enabled_tracing
):
    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span

    pre = GenerationPreCallPayload(action=None, context=None, generation_id="gid-2")
    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        await backend_plugin.on_pre_call(pre, {})

    mot = ModelOutputThunk("hello")
    mot.generation = GenerationMetadata(
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        model="gpt-4o",
        provider="openai",
    )
    post = GenerationPostCallPayload(
        prompt="p", model_output=mot, latency_ms=100.0, generation_id="gid-2"
    )
    await backend_plugin.on_post_call(post, {})

    fake_span.end.assert_called_once()
    attrs = _attrs(fake_span)
    assert attrs["gen_ai.usage.input_tokens"] == 10
    assert attrs["gen_ai.usage.output_tokens"] == 5
    assert attrs["gen_ai.usage.total_tokens"] == 15
    assert attrs["gen_ai.provider.name"] == "openai"
    assert attrs["gen_ai.request.model"] == "gpt-4o"
    assert "gid-2" not in tracing._in_flight_spans


@pytest.mark.asyncio
async def test_error_finishes_span_with_error_status(backend_plugin, enabled_tracing):
    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span

    pre = GenerationPreCallPayload(action=None, context=None, generation_id="gid-err")
    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        await backend_plugin.on_pre_call(pre, {})

    err = ValueError("rate limit")
    mot = ModelOutputThunk(None)
    mot.generation = GenerationMetadata(model="gpt-4o", provider="openai")
    err_payload = GenerationErrorPayload(
        exception=err, model_output=mot, generation_id="gid-err"
    )
    await backend_plugin.on_error(err_payload, {})

    fake_span.record_exception.assert_called_once_with(err)
    fake_span.set_status.assert_called_once()
    fake_span.end.assert_called_once()
    attrs = _attrs(fake_span)
    assert attrs["error.type"] == "ValueError"
    assert "gid-err" not in tracing._in_flight_spans


@pytest.mark.asyncio
async def test_pre_call_no_op_when_disabled(backend_plugin, disabled_tracing):
    payload = GenerationPreCallPayload(action=None, context=None, generation_id="x")
    await backend_plugin.on_pre_call(payload, {})
    assert "x" not in tracing._in_flight_spans


@pytest.mark.asyncio
async def test_pre_call_no_op_with_none_generation_id(backend_plugin, enabled_tracing):
    """generation_id=None (e.g. from a non-tracing-aware caller) is skipped."""
    fake_tracer = MagicMock()
    payload = GenerationPreCallPayload(action=None, context=None, generation_id=None)

    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        await backend_plugin.on_pre_call(payload, {})

    fake_tracer.start_span.assert_not_called()


@pytest.mark.asyncio
async def test_concurrent_generations_do_not_collide(backend_plugin, enabled_tracing):
    """Two concurrent generations with distinct UUIDs each get their own span."""
    fake_span_a = MagicMock(name="span-A")
    fake_span_b = MagicMock(name="span-B")
    fake_tracer = MagicMock()
    fake_tracer.start_span.side_effect = [fake_span_a, fake_span_b]

    pre_a = GenerationPreCallPayload(action=None, context=None, generation_id="A")
    pre_b = GenerationPreCallPayload(action=None, context=None, generation_id="B")

    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        await backend_plugin.on_pre_call(pre_a, {})
        await backend_plugin.on_pre_call(pre_b, {})

    assert "A" in tracing._in_flight_spans
    assert "B" in tracing._in_flight_spans
    assert tracing._in_flight_spans["A"][0] is fake_span_a
    assert tracing._in_flight_spans["B"][0] is fake_span_b

    mot_b = ModelOutputThunk("b")
    mot_b.generation = GenerationMetadata(model="m", provider="p")
    mot_a = ModelOutputThunk("a")
    mot_a.generation = GenerationMetadata(model="m", provider="p")

    await backend_plugin.on_post_call(
        GenerationPostCallPayload(
            prompt="", model_output=mot_a, latency_ms=1.0, generation_id="A"
        ),
        {},
    )
    fake_span_a.end.assert_called_once()
    fake_span_b.end.assert_not_called()

    await backend_plugin.on_post_call(
        GenerationPostCallPayload(
            prompt="", model_output=mot_b, latency_ms=1.0, generation_id="B"
        ),
        {},
    )
    fake_span_b.end.assert_called_once()


@pytest.mark.asyncio
async def test_cache_and_reasoning_attrs_emitted(backend_plugin, enabled_tracing):
    """Cache and reasoning token attrs are emitted from the usage dict."""
    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span

    pre = GenerationPreCallPayload(action=None, context=None, generation_id="cache-gid")
    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        await backend_plugin.on_pre_call(pre, {})

    mot = ModelOutputThunk("x")
    mot.generation = GenerationMetadata(
        model="claude-x",
        provider="anthropic",
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "cache_creation_input_tokens": 200,
            "prompt_tokens_details": {"cached_tokens": 75},
            "completion_tokens_details": {"reasoning_tokens": 30},
        },
    )
    post = GenerationPostCallPayload(
        prompt="", model_output=mot, latency_ms=1.0, generation_id="cache-gid"
    )
    await backend_plugin.on_post_call(post, {})

    attrs = _attrs(fake_span)
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 75
    assert attrs["gen_ai.usage.cache_creation.input_tokens"] == 200
    assert attrs["gen_ai.usage.reasoning.output_tokens"] == 30


@pytest.mark.asyncio
async def test_conversation_id_set_from_session_id(backend_plugin, enabled_tracing):
    """`gen_ai.conversation.id` is sourced from the session_id ContextVar."""
    from mellea.telemetry.context import with_context

    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span

    pre = GenerationPreCallPayload(action=None, context=None, generation_id="conv-gid")
    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        with with_context(session_id="sess-123"):
            await backend_plugin.on_pre_call(pre, {})

    attrs = _attrs(fake_span)
    assert attrs["gen_ai.conversation.id"] == "sess-123"


class _DummyFormat:
    """Stand-in for a structured-output BaseModel subclass."""


@pytest.mark.asyncio
async def test_pre_call_emits_request_side_mellea_attrs(
    backend_plugin, enabled_tracing
):
    """`mellea.has_format`, `mellea.format_type`, and `mellea.tool_calls_enabled` come through the pre_call payload."""
    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span

    pre = GenerationPreCallPayload(
        action=None,
        context=None,
        generation_id="req-gid",
        format=_DummyFormat,
        tool_calls=True,
    )
    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        await backend_plugin.on_pre_call(pre, {})

    attrs = _attrs(fake_span)
    assert "mellea.backend" not in attrs
    assert attrs["mellea.has_format"] is True
    assert attrs["mellea.format_type"] == "_DummyFormat"
    assert attrs["mellea.tool_calls_enabled"] is True
    # OTel GenAI semconv attribute for structured output
    assert attrs["gen_ai.output.type"] == "json_schema"


@pytest.mark.asyncio
async def test_pre_call_omits_format_type_when_no_format(
    backend_plugin, enabled_tracing
):
    """`mellea.format_type` is not emitted when no format is supplied; `mellea.has_format` is still emitted as False."""
    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span

    pre = GenerationPreCallPayload(
        action=None,
        context=None,
        generation_id="req-gid-2",
        format=None,
        tool_calls=False,
    )
    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        await backend_plugin.on_pre_call(pre, {})

    attrs = _attrs(fake_span)
    assert attrs["mellea.has_format"] is False
    assert "mellea.format_type" not in attrs
    assert attrs["mellea.tool_calls_enabled"] is False
    assert "gen_ai.output.type" not in attrs


@pytest.mark.asyncio
async def test_post_call_emits_response_side_attrs(backend_plugin, enabled_tracing):
    """`gen_ai.response.{model,id,finish_reasons}` populated from `mot.generation` in post_call."""
    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span

    pre = GenerationPreCallPayload(action=None, context=None, generation_id="resp-gid")
    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        await backend_plugin.on_pre_call(pre, {})

    mot = ModelOutputThunk("hi")
    mot.generation = GenerationMetadata(
        model="gpt-4o",
        provider="openai",
        response_model="gpt-4o-2024-08-06",
        response_id="chatcmpl-abc123",
        finish_reasons=["stop"],
    )
    post = GenerationPostCallPayload(
        prompt="p", model_output=mot, latency_ms=1.0, generation_id="resp-gid"
    )
    await backend_plugin.on_post_call(post, {})

    attrs = _attrs(fake_span)
    assert attrs["gen_ai.response.model"] == "gpt-4o-2024-08-06"
    assert attrs["gen_ai.response.id"] == "chatcmpl-abc123"
    assert attrs["gen_ai.response.finish_reasons"] == ["stop"]


@pytest.mark.asyncio
async def test_post_call_skips_response_attrs_when_unset(
    backend_plugin, enabled_tracing
):
    """Response attrs are skipped when the backend didn't populate them
    (e.g. HuggingFace, which has no provider response object)."""
    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span

    pre = GenerationPreCallPayload(action=None, context=None, generation_id="hf-gid")
    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        await backend_plugin.on_pre_call(pre, {})

    mot = ModelOutputThunk("hi")
    mot.generation = GenerationMetadata(model="m", provider="huggingface")
    post = GenerationPostCallPayload(
        prompt="p", model_output=mot, latency_ms=1.0, generation_id="hf-gid"
    )
    await backend_plugin.on_post_call(post, {})

    attrs = _attrs(fake_span)
    assert "gen_ai.response.model" not in attrs
    assert "gen_ai.response.id" not in attrs
    assert "gen_ai.response.finish_reasons" not in attrs


@pytest.mark.asyncio
async def test_batch_pre_call_starts_text_completion_span(
    backend_plugin, enabled_tracing
):
    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span

    payload = GenerationBatchPreCallPayload(
        actions=(),
        generation_id="batch-1",
        num_actions=3,
        model="m-x",
        provider="openai",
        format=_DummyFormat,
        tool_calls=False,
    )

    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        await backend_plugin.on_batch_pre_call(payload, {})

    fake_tracer.start_span.assert_called_once_with("text_completion")
    assert "batch-1" in tracing._in_flight_spans
    attrs = _attrs(fake_span)
    assert attrs["gen_ai.operation.name"] == "text_completion"
    assert attrs["mellea.num_actions"] == 3
    assert attrs["gen_ai.provider.name"] == "openai"
    assert attrs["gen_ai.request.model"] == "m-x"
    assert "mellea.backend" not in attrs
    assert attrs["mellea.has_format"] is True
    assert attrs["mellea.format_type"] == "_DummyFormat"
    assert attrs["mellea.tool_calls_enabled"] is False


@pytest.mark.asyncio
async def test_batch_post_call_finishes_with_aggregate_usage(
    backend_plugin, enabled_tracing
):
    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span

    pre = GenerationBatchPreCallPayload(
        actions=(),
        generation_id="batch-2",
        num_actions=2,
        model="m-y",
        provider="watsonx",
    )
    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        await backend_plugin.on_batch_pre_call(pre, {})

    post = GenerationBatchPostCallPayload(
        generation_id="batch-2",
        model_outputs=[],
        usage={"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45},
        model="m-y",
        provider="watsonx",
        latency_ms=200.0,
    )
    await backend_plugin.on_batch_post_call(post, {})

    fake_span.end.assert_called_once()
    attrs = _attrs(fake_span)
    assert attrs["gen_ai.usage.input_tokens"] == 30
    assert attrs["gen_ai.usage.output_tokens"] == 15
    assert attrs["gen_ai.usage.total_tokens"] == 45
    assert "batch-2" not in tracing._in_flight_spans


@pytest.mark.asyncio
async def test_batch_error_finishes_with_error_status(backend_plugin, enabled_tracing):
    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span

    pre = GenerationBatchPreCallPayload(
        actions=(), generation_id="batch-err", num_actions=1, model="m", provider="p"
    )
    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        await backend_plugin.on_batch_pre_call(pre, {})

    err = RuntimeError("boom")
    err_payload = GenerationBatchErrorPayload(
        generation_id="batch-err",
        exception=err,
        model="m",
        provider="p",
        latency_ms=50.0,
    )
    await backend_plugin.on_batch_error(err_payload, {})

    fake_span.record_exception.assert_called_once_with(err)
    fake_span.set_status.assert_called_once()
    fake_span.end.assert_called_once()
    attrs = _attrs(fake_span)
    assert attrs["error.type"] == "RuntimeError"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_span_has_correct_parent(backend_plugin, enabled_tracing):
    """A backend span started inside a user span gets the user span as parent."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    tracing.get_backend_tracer()
    exporter = InMemorySpanExporter()
    tracing._tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    user_tracer = tracing._tracer_provider.get_tracer("test-user")
    with user_tracer.start_as_current_span("user-outer") as user_span:
        user_trace_id = user_span.get_span_context().trace_id
        user_span_id = user_span.get_span_context().span_id

        pre = GenerationPreCallPayload(
            action=None, context=None, generation_id="parent-gid"
        )
        await backend_plugin.on_pre_call(pre, {})

        mot = ModelOutputThunk("x")
        mot.generation = GenerationMetadata(model="m", provider="p")
        post = GenerationPostCallPayload(
            prompt="", model_output=mot, latency_ms=1.0, generation_id="parent-gid"
        )
        await backend_plugin.on_post_call(post, {})

    tracing._tracer_provider.force_flush()
    spans = exporter.get_finished_spans()
    chat_spans = [s for s in spans if s.name == "chat"]
    assert len(chat_spans) == 1
    chat = chat_spans[0]

    assert chat.context.trace_id == user_trace_id
    assert chat.parent is not None
    assert chat.parent.span_id == user_span_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_nested_span_during_call_parents_under_backend_span(
    backend_plugin, enabled_tracing
):
    """A span started between pre_call and post_call parents under the backend span."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    tracing.get_backend_tracer()
    exporter = InMemorySpanExporter()
    tracing._tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    pre = GenerationPreCallPayload(
        action=None, context=None, generation_id="active-gid"
    )
    await backend_plugin.on_pre_call(pre, {})

    backend_span_id = (
        tracing._in_flight_spans["active-gid"][0].get_span_context().span_id
    )

    nested_tracer = tracing._tracer_provider.get_tracer("test-nested")
    with nested_tracer.start_as_current_span("nested-caller-task"):
        pass

    async def in_background_task() -> None:
        with nested_tracer.start_as_current_span("nested-background-task"):
            pass

    await asyncio.create_task(in_background_task())

    mot = ModelOutputThunk("x")
    mot.generation = GenerationMetadata(model="m", provider="p")
    post = GenerationPostCallPayload(
        prompt="", model_output=mot, latency_ms=1.0, generation_id="active-gid"
    )
    await backend_plugin.on_post_call(post, {})

    tracing._tracer_provider.force_flush()
    by_name = {s.name: s for s in exporter.get_finished_spans()}

    assert by_name["nested-caller-task"].parent is not None
    assert by_name["nested-caller-task"].parent.span_id == backend_span_id
    assert by_name["nested-background-task"].parent is not None
    assert by_name["nested-background-task"].parent.span_id == backend_span_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sequential_backend_calls_produce_siblings(
    backend_plugin, enabled_tracing
):
    """Two back-to-back backend calls in the same task with no enclosing app span are siblings, not parent/child."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    tracing.get_backend_tracer()
    exporter = InMemorySpanExporter()
    tracing._tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    for gid in ("call-a", "call-b"):
        pre = GenerationPreCallPayload(action=None, context=None, generation_id=gid)
        await backend_plugin.on_pre_call(pre, {})

        mot = ModelOutputThunk("x")
        mot.generation = GenerationMetadata(model="m", provider="p")
        post = GenerationPostCallPayload(
            prompt="", model_output=mot, latency_ms=1.0, generation_id=gid
        )
        await backend_plugin.on_post_call(post, {})

    tracing._tracer_provider.force_flush()
    chat_spans = [s for s in exporter.get_finished_spans() if s.name == "chat"]
    assert len(chat_spans) == 2

    # Neither span should be parented to the other; with no enclosing app
    # span, both should have no parent.
    for s in chat_spans:
        assert s.parent is None, (
            f"backend span {s.context.span_id:x} unexpectedly has parent "
            f"{s.parent.span_id:x} — context token detach is missing"
        )


# ComponentTracingPlugin tests


@pytest.mark.asyncio
async def test_action_pre_execute_emits_request_attrs(
    component_plugin, enabled_tracing
):
    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span

    class Foo:
        pass

    class FakeStrategy:
        pass

    pre = ComponentPreExecutePayload(
        action_id="cid-1",
        component_type="Foo",
        action=Foo(),
        requirements=["r1"],
        strategy=FakeStrategy(),
        format=_DummyFormat,
        tool_calls_enabled=True,
    )
    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        await component_plugin.on_component_pre_execute(pre, {})

    fake_tracer.start_span.assert_called_once_with("action")
    assert "cid-1" in tracing._in_flight_spans
    attrs = _attrs(fake_span)
    assert attrs["mellea.action_type"] == "Foo"
    assert attrs["mellea.has_requirements"] is True
    assert attrs["mellea.has_strategy"] is True
    assert attrs["mellea.strategy_type"] == "FakeStrategy"
    assert attrs["mellea.has_format"] is True
    assert attrs["mellea.tool_calls"] is True


@pytest.mark.asyncio
async def test_action_post_success_records_response_length_always(
    component_plugin, enabled_tracing, monkeypatch
):
    """`mellea.response_length` is recorded regardless of MELLEA_TRACES_CONTENT."""
    monkeypatch.delenv("MELLEA_TRACES_CONTENT", raising=False)
    monkeypatch.delenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False
    )

    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span

    pre = ComponentPreExecutePayload(action_id="cid-2", component_type="X")
    result = MagicMock()
    result.value = "hello world"
    success = ComponentPostSuccessPayload(
        action_id="cid-2", result=result, generate_log=MagicMock()
    )

    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        await component_plugin.on_component_pre_execute(pre, {})
        await component_plugin.on_component_post_success(success, {})

    attrs = _attrs(fake_span)
    assert attrs["mellea.response_length"] == len("hello world")
    assert "mellea.response" not in attrs


@pytest.mark.asyncio
async def test_action_post_success_records_response_when_content_enabled(
    component_plugin, enabled_tracing, monkeypatch
):
    """`mellea.response` is recorded only when `MELLEA_TRACES_CONTENT=true`."""
    monkeypatch.setenv("MELLEA_TRACES_CONTENT", "true")

    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span

    pre = ComponentPreExecutePayload(action_id="cid-3", component_type="X")
    result = MagicMock()
    result.value = "captured text"
    success = ComponentPostSuccessPayload(
        action_id="cid-3", result=result, generate_log=MagicMock()
    )

    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        await component_plugin.on_component_pre_execute(pre, {})
        await component_plugin.on_component_post_success(success, {})

    attrs = _attrs(fake_span)
    assert attrs["mellea.response"] == "captured text"
    assert attrs["mellea.response_length"] == len("captured text")


@pytest.mark.asyncio
async def test_action_post_success_truncates_long_response(
    component_plugin, enabled_tracing, monkeypatch
):
    monkeypatch.setenv("MELLEA_TRACES_CONTENT", "true")

    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span

    pre = ComponentPreExecutePayload(action_id="cid-4", component_type="X")
    long_text = "a" * 800
    result = MagicMock()
    result.value = long_text
    success = ComponentPostSuccessPayload(
        action_id="cid-4", result=result, generate_log=MagicMock()
    )

    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        await component_plugin.on_component_pre_execute(pre, {})
        await component_plugin.on_component_post_success(success, {})

    attrs = _attrs(fake_span)
    assert attrs["mellea.response"].endswith("...")
    assert len(attrs["mellea.response"]) == 503
    assert attrs["mellea.response_length"] == 800


@pytest.mark.asyncio
async def test_action_post_error_marks_error_status(component_plugin, enabled_tracing):
    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span

    pre = ComponentPreExecutePayload(action_id="cid-err", component_type="X")
    err = ValueError("nope")
    error_payload = ComponentPostErrorPayload(
        action_id="cid-err", error=err, error_type="ValueError"
    )

    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        await component_plugin.on_component_pre_execute(pre, {})
        await component_plugin.on_component_post_error(error_payload, {})

    fake_span.record_exception.assert_called_once_with(err)
    fake_span.set_status.assert_called_once()
    fake_span.end.assert_called_once()
    attrs = _attrs(fake_span)
    assert attrs["error.type"] == "ValueError"
    assert "cid-err" not in tracing._in_flight_spans


@pytest.mark.asyncio
async def test_action_pre_execute_no_op_with_empty_action_id(
    component_plugin, enabled_tracing
):
    fake_tracer = MagicMock()
    pre = ComponentPreExecutePayload(action_id="", component_type="X")
    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        await component_plugin.on_component_pre_execute(pre, {})
    fake_tracer.start_span.assert_not_called()


# StreamingTracingPlugin tests


def _events(span: MagicMock) -> list[tuple[str, dict]]:
    """Collect `span.add_event(name, attrs)` calls into a list."""
    return [(c.args[0], c.args[1]) for c in span.add_event.call_args_list]


async def _open_streaming_span(
    streaming_plugin, fake_tracer, streaming_id: str
) -> None:
    """Fire streaming_start so a span is stashed under streaming_id."""
    pre = StreamingStartPayload(streaming_id=streaming_id, chunking_strategy="x")
    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        await streaming_plugin.on_streaming_start(pre, {})


@pytest.mark.asyncio
async def test_streaming_start_starts_span_and_stashes_by_streaming_id(
    streaming_plugin, enabled_tracing
):
    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span

    payload = StreamingStartPayload(
        streaming_id="sid-1",
        has_requirements=True,
        requirement_count=2,
        chunking_strategy="SentenceChunker",
    )

    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        await streaming_plugin.on_streaming_start(payload, {})

    fake_tracer.start_span.assert_called_once_with("stream_with_chunking")
    assert "sid-1" in tracing._in_flight_spans
    fake_span.end.assert_not_called()
    attrs = _attrs(fake_span)
    assert attrs["mellea.has_requirements"] is True
    assert attrs["mellea.requirement_count"] == 2
    assert attrs["mellea.chunking_strategy"] == "SentenceChunker"
    # The correlation id is the in-flight key, not a span attribute.
    assert "mellea.streaming_id" not in attrs


@pytest.mark.asyncio
async def test_streaming_end_records_completed_event_then_closes_span(
    streaming_plugin, enabled_tracing
):
    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span
    await _open_streaming_span(streaming_plugin, fake_tracer, "sid-2")

    end = StreamingEndPayload(
        streaming_id="sid-2",
        success=True,
        model="gpt-4o",
        provider="openai",
        full_text_length=11,
    )
    await streaming_plugin.on_streaming_end(end, {})

    fake_span.end.assert_called_once()
    events = _events(fake_span)
    assert any(name == "completed" for name, _ in events)
    completed_attrs = next(attrs for name, attrs in events if name == "completed")
    assert completed_attrs["success"] is True
    assert completed_attrs["full_text_length"] == 11

    attrs = _attrs(fake_span)
    assert attrs["mellea.full_text_length"] == 11
    assert attrs["gen_ai.request.model"] == "gpt-4o"
    assert attrs["gen_ai.provider.name"] == "openai"
    assert "sid-2" not in tracing._in_flight_spans


@pytest.mark.asyncio
async def test_streaming_end_validation_fail_marks_span_error_without_exception(
    streaming_plugin, enabled_tracing
):
    """Validation-fail early-exit: span ERROR via failure_reason, no recorded exception."""
    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span
    await _open_streaming_span(streaming_plugin, fake_tracer, "sid-fail")

    end = StreamingEndPayload(
        streaming_id="sid-fail",
        success=False,
        failure_reason="Streaming validation failed: too short",
        full_text_length=4,
    )
    await streaming_plugin.on_streaming_end(end, {})

    fake_span.end.assert_called_once()
    fake_span.set_status.assert_called_once()
    fake_span.record_exception.assert_not_called()


@pytest.mark.asyncio
async def test_streaming_end_with_exception_records_exception(
    streaming_plugin, enabled_tracing
):
    """Exception outcome: span ERROR with the exception recorded."""
    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span
    await _open_streaming_span(streaming_plugin, fake_tracer, "sid-err")

    exc = ValueError("boom")
    end = StreamingEndPayload(
        streaming_id="sid-err", success=False, exception=exc, model="m", provider="p"
    )
    await streaming_plugin.on_streaming_end(end, {})

    fake_span.end.assert_called_once()
    fake_span.record_exception.assert_called_once_with(exc)
    assert "sid-err" not in tracing._in_flight_spans


@pytest.mark.asyncio
async def test_streaming_event_records_mid_stream_events(
    streaming_plugin, enabled_tracing
):
    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span
    await _open_streaming_span(streaming_plugin, fake_tracer, "sid-ev")

    qc = QuickCheckEvent(
        chunk_index=0, attempt=1, passed=True, results=[PartialValidationResult("pass")]
    )
    chunk = ChunkEvent(text="hello", chunk_index=0, attempt=1)
    done = StreamingDoneEvent(attempt=1, full_text="hello world")
    full_val = FullValidationEvent(
        attempt=1, passed=True, results=[ValidationResult(result=True)]
    )

    for ev in (qc, chunk, done, full_val):
        await streaming_plugin.on_streaming_event(
            StreamingEventPayload(streaming_id="sid-ev", event=ev), {}
        )

    events = _events(fake_span)
    names = [name for name, _ in events]
    assert names == ["quick_check", "chunk", "streaming_done", "full_validation"]
    qc_attrs = events[0][1]
    assert qc_attrs["chunk_index"] == 0
    assert qc_attrs["passed"] is True
    assert qc_attrs["requirement_count"] == 1
    chunk_attrs = events[1][1]
    assert chunk_attrs["text_length"] == 5

    # streaming_event never closes the span.
    fake_span.end.assert_not_called()


@pytest.mark.asyncio
async def test_streaming_event_records_error_event(streaming_plugin, enabled_tracing):
    """The `error` span event is recorded by streaming_event (the orchestrator emits ErrorEvent)."""
    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span
    await _open_streaming_span(streaming_plugin, fake_tracer, "sid-e")

    error_event = ErrorEvent(exception_type="ValueError", detail="boom")
    await streaming_plugin.on_streaming_event(
        StreamingEventPayload(streaming_id="sid-e", event=error_event), {}
    )

    events = _events(fake_span)
    assert any(name == "error" for name, _ in events)
    error_attrs = next(attrs for name, attrs in events if name == "error")
    assert error_attrs["exception_type"] == "ValueError"
    assert error_attrs["detail"] == "boom"
    fake_span.end.assert_not_called()


@pytest.mark.asyncio
async def test_streaming_event_skipped_when_span_not_in_flight(
    streaming_plugin, enabled_tracing
):
    qc = QuickCheckEvent(chunk_index=0, attempt=1, passed=True, results=[])
    # No matching start — _in_flight_spans is empty for this id; should be a no-op.
    await streaming_plugin.on_streaming_event(
        StreamingEventPayload(streaming_id="sid-missing", event=qc), {}
    )


@pytest.mark.asyncio
async def test_streaming_start_end_skip_when_streaming_id_empty(
    streaming_plugin, enabled_tracing
):
    """Empty streaming_id is a no-op so plugins don't break unrelated callers."""
    fake_tracer = MagicMock()
    with patch(
        "mellea.telemetry.tracing.get_application_tracer", return_value=fake_tracer
    ):
        await streaming_plugin.on_streaming_start(
            StreamingStartPayload(streaming_id=""), {}
        )
    fake_tracer.start_span.assert_not_called()

    await streaming_plugin.on_streaming_end(
        StreamingEndPayload(streaming_id="", success=True), {}
    )
