"""Unit tests for core/requirement.py pure helpers — ValidationResult, default_output_to_bool."""

import pytest

from mellea.core import CBlock, ModelOutputThunk
from mellea.core.requirement import ValidationResult, default_output_to_bool

# --- ValidationResult ---


def test_validation_result_pass():
    r = ValidationResult(result=True)
    assert r.as_bool() is True
    assert bool(r) is True


def test_validation_result_fail():
    r = ValidationResult(result=False)
    assert r.as_bool() is False
    assert bool(r) is False


def test_validation_result_reason():
    r = ValidationResult(result=True, reason="looks good")
    assert r.reason == "looks good"


def test_validation_result_score():
    r = ValidationResult(result=True, score=0.95)
    assert r.score == pytest.approx(0.95)


def test_validation_result_thunk():
    mot = ModelOutputThunk(value="x")
    r = ValidationResult(result=True, thunk=mot)
    assert r.thunk is mot


def test_validation_result_context():
    from mellea.stdlib.context import SimpleContext

    ctx = SimpleContext()
    r = ValidationResult(result=True, context=ctx)
    assert r.context is ctx


def test_validation_result_defaults_none():
    r = ValidationResult(result=False)
    assert r.reason is None
    assert r.score is None
    assert r.thunk is None
    assert r.context is None


# --- default_output_to_bool ---


def test_yes_exact_passes():
    assert default_output_to_bool(CBlock("yes")) is True


def test_yes_uppercase_passes():
    assert default_output_to_bool(CBlock("YES")) is True


def test_y_passes():
    assert default_output_to_bool(CBlock("y")) is True


def test_yes_in_sentence():
    assert default_output_to_bool(CBlock("Yes, it meets the requirement.")) is True


def test_no_fails():
    assert default_output_to_bool(CBlock("no")) is False


def test_empty_string_fails():
    assert default_output_to_bool(CBlock("")) is False


def test_random_text_fails():
    assert default_output_to_bool(CBlock("the output looks reasonable")) is False


def test_plain_string_yes():
    assert default_output_to_bool("YES") is True  # type: ignore


# --- Binary detection for repair feedback (PR #1248) ---
# Tests for Requirement.validate() binary detection: bare "yes"/"no" responses
# are replaced with the requirement description for more actionable repair feedback.
# @pytest: unit


async def _reason_for(backend, judge_value: str, description: str) -> str | None:
    """Helper to get the reason from Requirement.validate with a mocked backend."""
    from unittest.mock import AsyncMock, patch

    from mellea.core import ModelOutputThunk, Requirement
    from mellea.stdlib.context import ChatContext

    req = Requirement(description)
    ctx = ChatContext().add(ModelOutputThunk("some output"))
    judge_thunk = ModelOutputThunk(value=judge_value)
    val_ctx = ctx.add(judge_thunk)

    with patch.object(
        backend,
        "generate_from_context",
        new=AsyncMock(return_value=(judge_thunk, val_ctx)),
    ):
        result = await req.validate(backend, ctx)

    return result.reason


@pytest.fixture
def mock_backend():
    """Mock backend for testing Requirement.validate."""
    from unittest.mock import MagicMock

    return MagicMock()


@pytest.mark.unit
async def test_validate_binary_no_uses_description(mock_backend):
    """Binary 'no' should use requirement description as reason."""
    reason = await _reason_for(mock_backend, "no", "The email should have a salutation")
    assert reason == "The email should have a salutation"


@pytest.mark.unit
async def test_validate_binary_yes_uses_description(mock_backend):
    """Binary 'yes' should use requirement description as reason."""
    reason = await _reason_for(
        mock_backend, "yes", "The email should have a salutation"
    )
    assert reason == "The email should have a salutation"


@pytest.mark.unit
async def test_validate_detailed_answer_preserves_output(mock_backend):
    """Detailed judge output (not binary) should preserve the actual answer."""
    reason = await _reason_for(
        mock_backend,
        "The email contains a proper greeting",
        "The email should have a salutation",
    )
    assert reason == "The email contains a proper greeting"


@pytest.mark.unit
async def test_validate_binary_detection_whitespace_and_case(mock_backend):
    """Binary detection should handle whitespace and case variation."""
    reason = await _reason_for(mock_backend, "  NO  ", "Check requirement")
    assert reason == "Check requirement"


@pytest.mark.unit
async def test_validate_empty_string_preserves_output(mock_backend):
    """Empty string should not trigger fallback."""
    reason = await _reason_for(mock_backend, "", "The email should have a salutation")
    assert reason == ""


@pytest.mark.unit
async def test_validate_yes_in_sentence_preserves_output(mock_backend):
    """'Yes' as part of sentence should preserve judge output."""
    reason = await _reason_for(
        mock_backend,
        "Yes, the email has a salutation",
        "The email should have a salutation",
    )
    assert reason == "Yes, the email has a salutation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
