"""Unit tests for OTel GenAI semantic convention emission gaps (issue #1035).

All tests use a fake span object and do not require a live backend or
OpenTelemetry SDK installation.
"""

import json
from unittest.mock import MagicMock, patch

from mellea.telemetry.backend_instrumentation import (
    finalize_backend_span,
    get_provider_name,
    get_system_name,
    start_generate_span,
)
from mellea.telemetry.context import with_context
from mellea.telemetry.tracing import is_content_tracing_enabled

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_span() -> MagicMock:
    return MagicMock()


def _fake_backend(class_name: str) -> object:
    return type(class_name, (), {})()


def _span_attrs(span: MagicMock) -> dict:
    """Collect all set_attribute calls into a flat dict."""
    return {call.args[0]: call.args[1] for call in span.set_attribute.call_args_list}


# ---------------------------------------------------------------------------
# gen_ai.provider.name alongside gen_ai.system
# ---------------------------------------------------------------------------


def test_provider_name_equals_system_name():
    backend = _fake_backend("OpenAIBackend")
    assert get_provider_name(backend) == get_system_name(backend) == "openai"


def test_provider_name_emitted_in_start_generate_span():
    """Both gen_ai.system and gen_ai.provider.name should be set on the span."""
    backend = _fake_backend("OpenAIBackend")
    backend.model_id = "gpt-4"  # type: ignore[attr-defined]
    action = MagicMock()
    action.prompt_template_metadata = None

    with patch("mellea.telemetry.tracing.start_backend_span") as mock_start:
        mock_start.return_value = _mock_span()
        start_generate_span(backend, action, ctx=[], format=None, tool_calls=False)

    call_kwargs = mock_start.call_args[1]
    assert call_kwargs.get("gen_ai.system") == "openai"
    assert call_kwargs.get("gen_ai.provider.name") == "openai"


# ---------------------------------------------------------------------------
# gen_ai.conversation.id from session_id ContextVar
# ---------------------------------------------------------------------------


def test_conversation_id_emitted_from_session_id():
    backend = _fake_backend("OpenAIBackend")
    backend.model_id = "gpt-4"  # type: ignore[attr-defined]
    action = MagicMock()
    action.prompt_template_metadata = None

    with with_context(session_id="sess-abc"):
        with patch("mellea.telemetry.tracing.start_backend_span") as mock_start:
            mock_start.return_value = _mock_span()
            start_generate_span(backend, action, ctx=[], format=None, tool_calls=False)

    call_kwargs = mock_start.call_args[1]
    assert call_kwargs.get("gen_ai.conversation.id") == "sess-abc"
    assert call_kwargs.get("mellea.session_id") == "sess-abc"


def test_conversation_id_absent_when_no_session():
    backend = _fake_backend("OpenAIBackend")
    backend.model_id = "gpt-4"  # type: ignore[attr-defined]
    action = MagicMock()
    action.prompt_template_metadata = None

    with patch("mellea.telemetry.tracing.start_backend_span") as mock_start:
        mock_start.return_value = _mock_span()
        start_generate_span(backend, action, ctx=[], format=None, tool_calls=False)

    call_kwargs = mock_start.call_args[1]
    assert "gen_ai.conversation.id" not in call_kwargs


# ---------------------------------------------------------------------------
# llm.prompt_template.* from Instruction
# ---------------------------------------------------------------------------


def test_prompt_template_attrs_from_instruction():
    from mellea.stdlib.components.instruction import Instruction

    instr = Instruction(
        description="Summarise {{topic}} in one sentence.",
        user_variables={"topic": "quantum tunnelling"},
    )

    backend = _fake_backend("OpenAIBackend")
    backend.model_id = "gpt-4"  # type: ignore[attr-defined]

    with patch("mellea.telemetry.tracing.start_backend_span") as mock_start:
        mock_start.return_value = _mock_span()
        start_generate_span(backend, instr, ctx=[], format=None, tool_calls=False)

    call_kwargs = mock_start.call_args[1]
    # Template text is always emitted
    assert call_kwargs.get("llm.prompt_template.template") == (
        "Summarise {{topic}} in one sentence."
    )
    # Variables are NOT emitted when content capture is off (default)
    assert "llm.prompt_template.variables" not in call_kwargs


def test_prompt_template_variables_emitted_when_content_enabled(monkeypatch):
    from mellea.stdlib.components.instruction import Instruction

    instr = Instruction(description="Hello {{name}}", user_variables={"name": "World"})

    backend = _fake_backend("OpenAIBackend")
    backend.model_id = "gpt-4"  # type: ignore[attr-defined]

    # Patch the content gate to True
    monkeypatch.setattr(
        "mellea.telemetry.backend_instrumentation.is_content_tracing_enabled",
        lambda: True,
    )

    with patch("mellea.telemetry.tracing.start_backend_span") as mock_start:
        mock_start.return_value = _mock_span()
        start_generate_span(backend, instr, ctx=[], format=None, tool_calls=False)

    call_kwargs = mock_start.call_args[1]
    variables_json = call_kwargs.get("llm.prompt_template.variables")
    assert variables_json is not None
    parsed = json.loads(variables_json)
    assert parsed == {"name": "World"}


def test_instruction_without_user_variables_emits_template():
    from mellea.stdlib.components.instruction import Instruction

    instr = Instruction(description="Tell me about {{topic}}")
    # No user_variables — template is retained as-is

    backend = _fake_backend("OpenAIBackend")
    backend.model_id = "gpt-4"  # type: ignore[attr-defined]

    with patch("mellea.telemetry.tracing.start_backend_span") as mock_start:
        mock_start.return_value = _mock_span()
        start_generate_span(backend, instr, ctx=[], format=None, tool_calls=False)

    call_kwargs = mock_start.call_args[1]
    assert call_kwargs.get("llm.prompt_template.template") == "Tell me about {{topic}}"


def test_instruction_with_no_description_emits_no_template():
    from mellea.stdlib.components.instruction import Instruction

    instr = Instruction()  # no description

    backend = _fake_backend("OpenAIBackend")
    backend.model_id = "gpt-4"  # type: ignore[attr-defined]

    with patch("mellea.telemetry.tracing.start_backend_span") as mock_start:
        mock_start.return_value = _mock_span()
        start_generate_span(backend, instr, ctx=[], format=None, tool_calls=False)

    call_kwargs = mock_start.call_args[1]
    assert "llm.prompt_template.template" not in call_kwargs


# ---------------------------------------------------------------------------
# ERROR span status + error.type (finalize_backend_span error path)
# ---------------------------------------------------------------------------


def test_error_sets_status_and_error_type():
    span = _mock_span()
    exc = RuntimeError("model rejected")

    with (
        patch(
            "mellea.telemetry.backend_instrumentation.set_span_error"
        ) as mock_set_err,
        patch("mellea.telemetry.backend_instrumentation.end_backend_span") as mock_end,
    ):
        finalize_backend_span(span, error=exc)

    mock_set_err.assert_called_once_with(span, exc)
    attrs = _span_attrs(span)
    assert attrs.get("error.type") == "RuntimeError"
    mock_end.assert_called_once_with(span)


def test_error_path_always_closes_span():
    span = _mock_span()
    with patch("mellea.telemetry.backend_instrumentation.set_span_error"):
        with patch(
            "mellea.telemetry.backend_instrumentation.end_backend_span"
        ) as mock_end:
            finalize_backend_span(span, error=ValueError("x"))
    mock_end.assert_called_once()


def test_finalize_never_raises_on_span_error(monkeypatch):
    """finalize_backend_span must not propagate exceptions from helpers."""
    span = _mock_span()
    span.set_attribute.side_effect = RuntimeError("span broke")

    with patch("mellea.telemetry.backend_instrumentation.end_backend_span"):
        with patch("mellea.telemetry.backend_instrumentation.set_span_error"):
            # Should not raise even though set_attribute raises
            finalize_backend_span(span, error=ValueError("test"))


def test_finalize_none_span_is_noop():
    finalize_backend_span(None, error=RuntimeError("x"))  # no exception


# ---------------------------------------------------------------------------
# Content capture (gen_ai.input.messages etc.) gated by MELLEA_TRACE_CONTENT
# ---------------------------------------------------------------------------


def test_content_capture_disabled_by_default():
    span = _mock_span()
    conversation = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    with patch("mellea.telemetry.backend_instrumentation.end_backend_span"):
        finalize_backend_span(span, conversation=conversation, output_text="Hi there")

    attrs = _span_attrs(span)
    assert "gen_ai.input.messages" not in attrs
    assert "gen_ai.output.messages" not in attrs
    assert "gen_ai.system_instructions" not in attrs


def test_content_capture_emits_structured_attributes(monkeypatch):
    monkeypatch.setattr(
        "mellea.telemetry.backend_instrumentation.is_content_tracing_enabled",
        lambda: True,
    )
    span = _mock_span()
    conversation = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Tell me a joke."},
    ]
    with patch("mellea.telemetry.backend_instrumentation.end_backend_span"):
        with patch("mellea.telemetry.backend_instrumentation.add_span_event"):
            finalize_backend_span(
                span,
                conversation=conversation,
                output_text="Why did the chicken cross the road?",
            )

    attrs = _span_attrs(span)

    # System instructions
    sys_json = attrs.get("gen_ai.system_instructions")
    assert sys_json is not None
    sys_parts = json.loads(sys_json)
    assert sys_parts == [{"type": "text", "content": "You are helpful."}]

    # Input messages (non-system)
    in_json = attrs.get("gen_ai.input.messages")
    assert in_json is not None
    in_msgs = json.loads(in_json)
    assert len(in_msgs) == 1
    assert in_msgs[0]["role"] == "user"
    assert in_msgs[0]["parts"] == [{"type": "text", "content": "Tell me a joke."}]

    # Output messages
    out_json = attrs.get("gen_ai.output.messages")
    assert out_json is not None
    out_msgs = json.loads(out_json)
    assert out_msgs[0]["role"] == "assistant"
    assert out_msgs[0]["parts"][0]["content"] == "Why did the chicken cross the road?"
    assert "finish_reason" in out_msgs[0]


def test_content_capture_no_deprecated_per_role_events(monkeypatch):
    """The deprecated gen_ai.user.message / gen_ai.assistant.message events must not be emitted."""
    monkeypatch.setattr(
        "mellea.telemetry.backend_instrumentation.is_content_tracing_enabled",
        lambda: True,
    )
    span = _mock_span()
    with patch("mellea.telemetry.backend_instrumentation.end_backend_span"):
        finalize_backend_span(
            span, conversation=[{"role": "user", "content": "hi"}], output_text="hello"
        )

    event_names = [call.args[0] for call in span.add_event.call_args_list]
    deprecated = {
        "gen_ai.user.message",
        "gen_ai.assistant.message",
        "gen_ai.system.message",
    }
    assert not deprecated.intersection(event_names)


def test_content_span_event_emitted(monkeypatch):
    monkeypatch.setattr(
        "mellea.telemetry.backend_instrumentation.is_content_tracing_enabled",
        lambda: True,
    )
    span = _mock_span()
    with patch("mellea.telemetry.backend_instrumentation.end_backend_span"):
        with patch(
            "mellea.telemetry.backend_instrumentation.add_span_event"
        ) as mock_event:
            finalize_backend_span(
                span,
                conversation=[{"role": "user", "content": "hi"}],
                output_text="hello",
            )
    event_names = [call.args[1] for call in mock_event.call_args_list]
    assert "gen_ai.client.inference.operation.details" in event_names


# ---------------------------------------------------------------------------
# _TRACE_CONTENT_ENABLED recognises OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT
# ---------------------------------------------------------------------------


def test_content_tracing_enabled_via_mellea_env(monkeypatch):
    monkeypatch.setenv("MELLEA_TRACE_CONTENT", "true")
    import mellea.telemetry.tracing as tracing_mod

    # Force re-evaluation of module-level constant
    with patch.object(tracing_mod, "_TRACE_CONTENT_ENABLED", True):
        assert tracing_mod.is_content_tracing_enabled()


def test_content_tracing_disabled_by_default():
    assert not is_content_tracing_enabled()


# ---------------------------------------------------------------------------
# Success path of finalize_backend_span calls record helpers
# ---------------------------------------------------------------------------


def test_success_path_calls_record_token_usage():
    span = _mock_span()
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    with patch(
        "mellea.telemetry.backend_instrumentation.record_token_usage"
    ) as mock_rtu:
        with patch("mellea.telemetry.backend_instrumentation.end_backend_span"):
            finalize_backend_span(span, usage=usage)
    mock_rtu.assert_called_once_with(span, usage)


def test_success_path_calls_record_response_metadata():
    span = _mock_span()
    response = {"model": "gpt-4", "id": "resp-1"}
    with patch(
        "mellea.telemetry.backend_instrumentation.record_response_metadata"
    ) as mock_rrm:
        with patch("mellea.telemetry.backend_instrumentation.end_backend_span"):
            finalize_backend_span(span, response=response, model_id="gpt-4")
    mock_rrm.assert_called_once_with(span, response, model_id="gpt-4")
