"""Unit tests for the deprecated ``target_role`` path of ``guardian_check``.

Exercises the sentinel/mapping logic without touching a model. We monkeypatch
``call_intrinsic`` and assert on (a) the ``kwargs["scoring_schema"]`` that
reaches the adapter boundary and (b) the warnings/errors the caller sees.
"""

import warnings

import pytest

from mellea.stdlib.components.intrinsic import guardian
from mellea.stdlib.context import ChatContext


@pytest.fixture
def capture_kwargs(monkeypatch):
    """Replace call_intrinsic with a spy that returns a stub yes=1.0 result."""
    captured: dict = {}

    def fake_call_intrinsic(name, context, backend, /, kwargs=None, model_options=None):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return {"guardian": {"score": 1.0}}

    monkeypatch.setattr(guardian, "call_intrinsic", fake_call_intrinsic)
    return captured


def test_default_scoring_schema_resolves_to_assistant_response(capture_kwargs):
    guardian.guardian_check(ChatContext(), object(), criteria="harm")
    assert (
        capture_kwargs["kwargs"]["scoring_schema"]
        == guardian.SCORING_SCHEMA_BANK["assistant_response"]
    )


def test_target_role_user_maps_to_user_prompt_with_deprecation_warning(capture_kwargs):
    with pytest.warns(DeprecationWarning, match="target_role"):
        guardian.guardian_check(
            ChatContext(), object(), criteria="harm", target_role="user"
        )
    assert (
        capture_kwargs["kwargs"]["scoring_schema"]
        == guardian.SCORING_SCHEMA_BANK["user_prompt"]
    )


def test_target_role_assistant_maps_to_assistant_response_with_warning(capture_kwargs):
    with pytest.warns(DeprecationWarning, match="target_role"):
        guardian.guardian_check(
            ChatContext(), object(), criteria="harm", target_role="assistant"
        )
    assert (
        capture_kwargs["kwargs"]["scoring_schema"]
        == guardian.SCORING_SCHEMA_BANK["assistant_response"]
    )


def test_target_role_invalid_value_raises_value_error(capture_kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(ValueError, match="target_role must be"):
            guardian.guardian_check(
                ChatContext(), object(), criteria="harm", target_role="system"
            )


def test_passing_both_scoring_schema_and_target_role_raises_type_error(capture_kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(TypeError, match="not both"):
            guardian.guardian_check(
                ChatContext(),
                object(),
                criteria="harm",
                scoring_schema="user_prompt",
                target_role="user",
            )


def test_positional_user_logs_warning_and_sends_literal(capture_kwargs, caplog):
    """Positional 'user' is NOT auto-remapped — it's sent as a literal schema
    sentence, with a logger warning pointing the caller at the fix.
    """
    with caplog.at_level("WARNING"):
        guardian.guardian_check(ChatContext(), object(), "harm", "user")
    # The literal "user" flows to the adapter unchanged.
    assert capture_kwargs["kwargs"]["scoring_schema"] == "user"
    # The warning text nudges the caller toward the bank key.
    assert any("user_prompt" in rec.message for rec in caplog.records)


def test_scoring_schema_bank_key_resolves_to_full_sentence(capture_kwargs):
    guardian.guardian_check(
        ChatContext(), object(), criteria="harm", scoring_schema="tool_call"
    )
    assert (
        capture_kwargs["kwargs"]["scoring_schema"]
        == guardian.SCORING_SCHEMA_BANK["tool_call"]
    )


def test_custom_scoring_schema_passes_through(capture_kwargs):
    custom = "If the previous turn mentions cats, return 'yes'; otherwise, return 'no'."
    guardian.guardian_check(
        ChatContext(), object(), criteria="harm", scoring_schema=custom
    )
    assert capture_kwargs["kwargs"]["scoring_schema"] == custom
