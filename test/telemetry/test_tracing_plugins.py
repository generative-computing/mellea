"""Unit tests for the BackendTracingPlugin span lifecycle."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip(
    "opentelemetry", reason="opentelemetry not installed — install mellea[telemetry]"
)
pytest.importorskip("cpex", reason="cpex not installed — install mellea[hooks]")

from mellea.core.base import GenerationMetadata, ModelOutputThunk
from mellea.plugins.hooks.generation import (
    GenerationBatchErrorPayload,
    GenerationBatchPostCallPayload,
    GenerationBatchPreCallPayload,
    GenerationErrorPayload,
    GenerationPostCallPayload,
    GenerationPreCallPayload,
)
from mellea.telemetry import tracing
from mellea.telemetry.tracing_plugins import BackendTracingPlugin


def _reset_tracing_state() -> None:
    """Reset module state and re-run setup so env-var changes take effect."""
    tracing._tracer_provider = None
    tracing._application_tracer = None
    tracing._backend_tracer = None
    tracing._in_flight_spans.clear()
    tracing._setup_tracing()


@pytest.fixture
def plugin():
    return BackendTracingPlugin()


@pytest.fixture
def enabled_tracing(monkeypatch):
    monkeypatch.setenv("MELLEA_TRACES_ENABLED", "true")
    _reset_tracing_state()
    yield
    _reset_tracing_state()


@pytest.fixture
def disabled_tracing(monkeypatch):
    monkeypatch.delenv("MELLEA_TRACES_ENABLED", raising=False)
    _reset_tracing_state()
    yield
    _reset_tracing_state()


def _attrs(span: MagicMock) -> dict:
    """Collect `span.set_attribute(k, v)` calls into a dict."""
    return {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}


@pytest.mark.asyncio
async def test_pre_call_starts_span_and_stashes_by_generation_id(
    plugin, enabled_tracing
):
    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span

    payload = GenerationPreCallPayload(action=None, context=None, generation_id="gid-1")

    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        await plugin.on_pre_call(payload, {})

    fake_tracer.start_span.assert_called_once_with("chat")
    assert "gid-1" in tracing._in_flight_spans
    fake_span.end.assert_not_called()
    attrs = _attrs(fake_span)
    assert attrs["gen_ai.operation.name"] == "chat"


@pytest.mark.asyncio
async def test_post_call_finishes_span_with_usage_attrs(plugin, enabled_tracing):
    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span

    pre = GenerationPreCallPayload(action=None, context=None, generation_id="gid-2")
    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        await plugin.on_pre_call(pre, {})

    mot = ModelOutputThunk("hello")
    mot.generation = GenerationMetadata(
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        model="gpt-4o",
        provider="openai",
    )
    post = GenerationPostCallPayload(
        prompt="p", model_output=mot, latency_ms=100.0, generation_id="gid-2"
    )
    await plugin.on_post_call(post, {})

    fake_span.end.assert_called_once()
    attrs = _attrs(fake_span)
    assert attrs["gen_ai.usage.input_tokens"] == 10
    assert attrs["gen_ai.usage.output_tokens"] == 5
    assert attrs["gen_ai.usage.total_tokens"] == 15
    assert attrs["gen_ai.provider.name"] == "openai"
    assert attrs["gen_ai.request.model"] == "gpt-4o"
    assert "gid-2" not in tracing._in_flight_spans


@pytest.mark.asyncio
async def test_post_call_no_op_when_no_matching_pre_call(plugin, enabled_tracing):
    """If pre_call didn't fire (e.g. tracing came up mid-flight), post_call must no-op."""
    fake_tracer = MagicMock()
    mot = ModelOutputThunk("hello")
    mot.generation = GenerationMetadata(model="gpt-4o", provider="openai")
    payload = GenerationPostCallPayload(
        prompt="p", model_output=mot, latency_ms=100.0, generation_id="never-pre"
    )

    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        await plugin.on_post_call(payload, {})

    fake_tracer.start_span.assert_not_called()


@pytest.mark.asyncio
async def test_error_finishes_span_with_error_status(plugin, enabled_tracing):
    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span

    pre = GenerationPreCallPayload(action=None, context=None, generation_id="gid-err")
    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        await plugin.on_pre_call(pre, {})

    err = ValueError("rate limit")
    mot = ModelOutputThunk(None)
    mot.generation = GenerationMetadata(model="gpt-4o", provider="openai")
    err_payload = GenerationErrorPayload(
        exception=err, model_output=mot, generation_id="gid-err"
    )
    await plugin.on_error(err_payload, {})

    fake_span.record_exception.assert_called_once_with(err)
    fake_span.set_status.assert_called_once()
    fake_span.end.assert_called_once()
    attrs = _attrs(fake_span)
    assert attrs["error.type"] == "ValueError"
    assert "gid-err" not in tracing._in_flight_spans


@pytest.mark.asyncio
async def test_pre_call_no_op_when_disabled(plugin, disabled_tracing):
    payload = GenerationPreCallPayload(action=None, context=None, generation_id="x")
    await plugin.on_pre_call(payload, {})
    assert "x" not in tracing._in_flight_spans


@pytest.mark.asyncio
async def test_pre_call_no_op_with_missing_generation_id(plugin, enabled_tracing):
    """Missing generation_id (e.g. from a non-tracing-aware caller) is skipped."""
    fake_tracer = MagicMock()
    payload = GenerationPreCallPayload(action=None, context=None, generation_id=None)

    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        await plugin.on_pre_call(payload, {})

    fake_tracer.start_span.assert_not_called()


@pytest.mark.asyncio
async def test_concurrent_generations_do_not_collide(plugin, enabled_tracing):
    """Two concurrent generations with distinct UUIDs each get their own span."""
    fake_span_a = MagicMock(name="span-A")
    fake_span_b = MagicMock(name="span-B")
    fake_tracer = MagicMock()
    fake_tracer.start_span.side_effect = [fake_span_a, fake_span_b]

    pre_a = GenerationPreCallPayload(action=None, context=None, generation_id="A")
    pre_b = GenerationPreCallPayload(action=None, context=None, generation_id="B")

    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        await plugin.on_pre_call(pre_a, {})
        await plugin.on_pre_call(pre_b, {})

    assert "A" in tracing._in_flight_spans
    assert "B" in tracing._in_flight_spans
    assert tracing._in_flight_spans["A"] is fake_span_a
    assert tracing._in_flight_spans["B"] is fake_span_b

    mot_b = ModelOutputThunk("b")
    mot_b.generation = GenerationMetadata(model="m", provider="p")
    mot_a = ModelOutputThunk("a")
    mot_a.generation = GenerationMetadata(model="m", provider="p")

    await plugin.on_post_call(
        GenerationPostCallPayload(
            prompt="", model_output=mot_a, latency_ms=1.0, generation_id="A"
        ),
        {},
    )
    fake_span_a.end.assert_called_once()
    fake_span_b.end.assert_not_called()

    await plugin.on_post_call(
        GenerationPostCallPayload(
            prompt="", model_output=mot_b, latency_ms=1.0, generation_id="B"
        ),
        {},
    )
    fake_span_b.end.assert_called_once()


@pytest.mark.asyncio
async def test_cache_and_reasoning_attrs_emitted(plugin, enabled_tracing):
    """Cache and reasoning token attrs are emitted from the usage dict."""
    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span

    pre = GenerationPreCallPayload(action=None, context=None, generation_id="cache-gid")
    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        await plugin.on_pre_call(pre, {})

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
    await plugin.on_post_call(post, {})

    attrs = _attrs(fake_span)
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 75
    assert attrs["gen_ai.usage.cache_creation.input_tokens"] == 200
    assert attrs["gen_ai.usage.reasoning.output_tokens"] == 30


@pytest.mark.asyncio
async def test_conversation_id_set_from_session_id(plugin, enabled_tracing):
    """`gen_ai.conversation.id` is sourced from the session_id ContextVar."""
    from mellea.telemetry.context import with_context

    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span

    pre = GenerationPreCallPayload(action=None, context=None, generation_id="conv-gid")
    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        with with_context(session_id="sess-123"):
            await plugin.on_pre_call(pre, {})

    attrs = _attrs(fake_span)
    assert attrs["gen_ai.conversation.id"] == "sess-123"


class _DummyFormat:
    """Stand-in for a structured-output BaseModel subclass."""


@pytest.mark.asyncio
async def test_pre_call_emits_request_side_mellea_attrs(plugin, enabled_tracing):
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
        await plugin.on_pre_call(pre, {})

    attrs = _attrs(fake_span)
    assert "mellea.backend" not in attrs
    assert attrs["mellea.has_format"] is True
    assert attrs["mellea.format_type"] == "_DummyFormat"
    assert attrs["mellea.tool_calls_enabled"] is True
    # OTel GenAI semconv attribute for structured output
    assert attrs["gen_ai.output.type"] == "json_schema"


@pytest.mark.asyncio
async def test_pre_call_omits_format_type_when_no_format(plugin, enabled_tracing):
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
        await plugin.on_pre_call(pre, {})

    attrs = _attrs(fake_span)
    assert attrs["mellea.has_format"] is False
    assert "mellea.format_type" not in attrs
    assert attrs["mellea.tool_calls_enabled"] is False
    assert "gen_ai.output.type" not in attrs


@pytest.mark.asyncio
async def test_post_call_emits_response_side_attrs(plugin, enabled_tracing):
    """`gen_ai.response.{model,id,finish_reasons}` populated from `mot.generation` in post_call."""
    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span

    pre = GenerationPreCallPayload(action=None, context=None, generation_id="resp-gid")
    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        await plugin.on_pre_call(pre, {})

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
    await plugin.on_post_call(post, {})

    attrs = _attrs(fake_span)
    assert attrs["gen_ai.response.model"] == "gpt-4o-2024-08-06"
    assert attrs["gen_ai.response.id"] == "chatcmpl-abc123"
    assert attrs["gen_ai.response.finish_reasons"] == ["stop"]


@pytest.mark.asyncio
async def test_post_call_skips_response_attrs_when_unset(plugin, enabled_tracing):
    """Response attrs are skipped when the backend didn't populate them
    (e.g. HuggingFace, which has no provider response object)."""
    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span

    pre = GenerationPreCallPayload(action=None, context=None, generation_id="hf-gid")
    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        await plugin.on_pre_call(pre, {})

    mot = ModelOutputThunk("hi")
    mot.generation = GenerationMetadata(model="m", provider="huggingface")
    post = GenerationPostCallPayload(
        prompt="p", model_output=mot, latency_ms=1.0, generation_id="hf-gid"
    )
    await plugin.on_post_call(post, {})

    attrs = _attrs(fake_span)
    assert "gen_ai.response.model" not in attrs
    assert "gen_ai.response.id" not in attrs
    assert "gen_ai.response.finish_reasons" not in attrs


@pytest.mark.asyncio
async def test_batch_pre_call_starts_text_completion_span(plugin, enabled_tracing):
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
        await plugin.on_batch_pre_call(payload, {})

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
async def test_batch_post_call_finishes_with_aggregate_usage(plugin, enabled_tracing):
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
        await plugin.on_batch_pre_call(pre, {})

    post = GenerationBatchPostCallPayload(
        generation_id="batch-2",
        model_outputs=[],
        usage={"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45},
        model="m-y",
        provider="watsonx",
        latency_ms=200.0,
    )
    await plugin.on_batch_post_call(post, {})

    fake_span.end.assert_called_once()
    attrs = _attrs(fake_span)
    assert attrs["gen_ai.usage.input_tokens"] == 30
    assert attrs["gen_ai.usage.output_tokens"] == 15
    assert attrs["gen_ai.usage.total_tokens"] == 45
    assert "batch-2" not in tracing._in_flight_spans


@pytest.mark.asyncio
async def test_batch_error_finishes_with_error_status(plugin, enabled_tracing):
    fake_span = MagicMock()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = fake_span

    pre = GenerationBatchPreCallPayload(
        actions=(), generation_id="batch-err", num_actions=1, model="m", provider="p"
    )
    with patch("mellea.telemetry.tracing.get_backend_tracer", return_value=fake_tracer):
        await plugin.on_batch_pre_call(pre, {})

    err = RuntimeError("boom")
    err_payload = GenerationBatchErrorPayload(
        generation_id="batch-err",
        exception=err,
        model="m",
        provider="p",
        latency_ms=50.0,
    )
    await plugin.on_batch_error(err_payload, {})

    fake_span.record_exception.assert_called_once_with(err)
    fake_span.set_status.assert_called_once()
    fake_span.end.assert_called_once()
    attrs = _attrs(fake_span)
    assert attrs["error.type"] == "RuntimeError"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_span_has_correct_parent(plugin, enabled_tracing):
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
        await plugin.on_pre_call(pre, {})

        mot = ModelOutputThunk("x")
        mot.generation = GenerationMetadata(model="m", provider="p")
        post = GenerationPostCallPayload(
            prompt="", model_output=mot, latency_ms=1.0, generation_id="parent-gid"
        )
        await plugin.on_post_call(post, {})

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
    plugin, enabled_tracing
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
    await plugin.on_pre_call(pre, {})

    backend_span_id = tracing._in_flight_spans["active-gid"].get_span_context().span_id

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
    await plugin.on_post_call(post, {})

    tracing._tracer_provider.force_flush()
    by_name = {s.name: s for s in exporter.get_finished_spans()}

    assert by_name["nested-caller-task"].parent is not None
    assert by_name["nested-caller-task"].parent.span_id == backend_span_id
    assert by_name["nested-background-task"].parent is not None
    assert by_name["nested-background-task"].parent.span_id == backend_span_id
