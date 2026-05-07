"""Unit tests for OTel GenAI semantic convention emission gaps (issue #1035).

Covers gaps 1-4. Gap 5 (content capture) is deferred; see cs/issue-1035-full
for the full implementation.

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
# Gap 1: gen_ai.provider.name alongside gen_ai.system
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
# Gap 2: gen_ai.conversation.id from session_id ContextVar
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
# Gap 3: llm.prompt_template.* from Instruction
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
    assert json.loads(variables_json) == {"name": "World"}


def test_instruction_without_user_variables_emits_template():
    from mellea.stdlib.components.instruction import Instruction

    instr = Instruction(description="Tell me about {{topic}}")

    backend = _fake_backend("OpenAIBackend")
    backend.model_id = "gpt-4"  # type: ignore[attr-defined]

    with patch("mellea.telemetry.tracing.start_backend_span") as mock_start:
        mock_start.return_value = _mock_span()
        start_generate_span(backend, instr, ctx=[], format=None, tool_calls=False)

    assert (
        mock_start.call_args[1].get("llm.prompt_template.template")
        == "Tell me about {{topic}}"
    )


def test_instruction_with_no_description_emits_no_template():
    from mellea.stdlib.components.instruction import Instruction

    instr = Instruction()

    backend = _fake_backend("OpenAIBackend")
    backend.model_id = "gpt-4"  # type: ignore[attr-defined]

    with patch("mellea.telemetry.tracing.start_backend_span") as mock_start:
        mock_start.return_value = _mock_span()
        start_generate_span(backend, instr, ctx=[], format=None, tool_calls=False)

    assert "llm.prompt_template.template" not in mock_start.call_args[1]


# ---------------------------------------------------------------------------
# Gap 4: ERROR span status + error.type
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
    assert _span_attrs(span).get("error.type") == "RuntimeError"
    mock_end.assert_called_once_with(span)


def test_error_path_always_closes_span():
    span = _mock_span()
    with patch("mellea.telemetry.backend_instrumentation.set_span_error"):
        with patch(
            "mellea.telemetry.backend_instrumentation.end_backend_span"
        ) as mock_end:
            finalize_backend_span(span, error=ValueError("x"))
    mock_end.assert_called_once()


def test_finalize_never_raises_on_span_error():
    """finalize_backend_span must not propagate exceptions from helpers."""
    span = _mock_span()
    span.set_attribute.side_effect = RuntimeError("span broke")

    with patch("mellea.telemetry.backend_instrumentation.end_backend_span"):
        with patch("mellea.telemetry.backend_instrumentation.set_span_error"):
            finalize_backend_span(span, error=ValueError("test"))


def test_finalize_none_span_is_noop():
    finalize_backend_span(None, error=RuntimeError("x"))


# ---------------------------------------------------------------------------
# Content tracing default (infrastructure for deferred gap 5)
# ---------------------------------------------------------------------------


def test_content_tracing_disabled_by_default():
    assert not is_content_tracing_enabled()
