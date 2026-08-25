# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for core/requirement.py pure helpers — ValidationResult, default_output_to_bool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mellea.core import CBlock, ModelOutputThunk, Requirement
from mellea.core.requirement import ValidationResult, default_output_to_bool
from mellea.stdlib.context import ChatContext

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
    assert r.error is None


# --- ValidationResult error state (issue #1358) ---


def test_validation_result_error_field():
    """The error field stores an exception and defaults to None."""
    exc = ValueError("bad output")
    r = ValidationResult(result=False, error=exc)
    assert r.error is exc


def test_bool_result_false_when_error_set():
    """`bool(result)` / `as_bool()` fail closed when error is set, even if result=True."""
    r = ValidationResult(result=True, error=RuntimeError("parse failed"))
    assert r.as_bool() is False
    assert bool(r) is False


def test_bool_result_uses_result_when_no_error():
    """Without an error, as_bool reflects the underlying result verbatim."""
    assert ValidationResult(result=True).as_bool() is True
    assert ValidationResult(result=False).as_bool() is False


def test_callers_can_distinguish_failure_from_error():
    """A plain failure has error=None; a parse failure carries the exception."""
    plain_failure = ValidationResult(result=False)
    parse_failure = ValidationResult(result=False, error=KeyError("score"))

    assert plain_failure.error is None
    assert isinstance(parse_failure.error, KeyError)


def test_validation_result_repr_includes_error():
    """repr surfaces the error so a swallowed exception is still visible in logs."""
    r = ValidationResult(result=False, error=ValueError("boom"))
    assert "error=" in repr(r)


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


async def _validate_result(backend, judge_value: str, description: str):
    """Helper to get the full validation result from Requirement.validate with a mocked backend."""
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

    return result


@pytest.fixture
def mock_backend():
    """Mock backend for testing Requirement.validate."""
    return MagicMock()


async def test_validate_binary_no_uses_description(mock_backend):
    """Binary 'no' should fail with requirement description as reason."""
    result = await _validate_result(
        mock_backend, "no", "The email should have a salutation"
    )
    assert result.as_bool() is False
    assert result.reason == "The email should have a salutation"


async def test_validate_binary_yes_uses_description(mock_backend):
    """Binary 'yes' should pass with requirement description as reason."""
    result = await _validate_result(
        mock_backend, "yes", "The email should have a salutation"
    )
    assert result.as_bool() is True
    assert result.reason == "The email should have a salutation"


async def test_validate_detailed_answer_preserves_output(mock_backend):
    """Detailed judge output (not binary) should preserve the actual answer."""
    result = await _validate_result(
        mock_backend,
        "Yes, the email contains a proper greeting",
        "The email should have a salutation",
    )
    assert result.as_bool() is True
    assert result.reason == "Yes, the email contains a proper greeting"


async def test_validate_binary_detection_whitespace_and_case(mock_backend):
    """Binary detection should handle whitespace and case variation."""
    result = await _validate_result(mock_backend, "  NO  ", "Check requirement")
    assert result.as_bool() is False
    assert result.reason == "Check requirement"


async def test_validate_empty_string_preserves_output(mock_backend):
    """Empty string should not trigger fallback."""
    result = await _validate_result(
        mock_backend, "", "The email should have a salutation"
    )
    assert result.as_bool() is False
    assert result.reason == ""


async def test_validate_yes_in_sentence_preserves_output(mock_backend):
    """'Yes' as part of sentence should preserve judge output."""
    result = await _validate_result(
        mock_backend,
        "Yes, the email has a salutation",
        "The email should have a salutation",
    )
    assert result.as_bool() is True
    assert result.reason == "Yes, the email has a salutation"


# --- validate() surfaces parse errors as a third outcome (issue #1358) ---


async def _validate_with_output_to_bool(backend, output_to_bool):
    """Run Requirement.validate with a mocked backend and a custom output_to_bool."""
    req = Requirement("some requirement", output_to_bool=output_to_bool)
    ctx = ChatContext().add(ModelOutputThunk("some output"))
    judge_thunk = ModelOutputThunk(value="some judge output")
    val_ctx = ctx.add(judge_thunk)

    with patch.object(
        backend,
        "generate_from_context",
        new=AsyncMock(return_value=(judge_thunk, val_ctx)),
    ):
        return await req.validate(backend, ctx)


async def test_validate_returns_error_state_on_schema_mismatch(mock_backend):
    """A raising output_to_bool becomes a ValidationResult carrying the exception."""
    exc = ValueError("unrecognizable adapter output")

    def raising_output_to_bool(x):
        raise exc

    result = await _validate_with_output_to_bool(mock_backend, raising_output_to_bool)

    assert result.error is exc
    assert bool(result) is False


async def test_validate_happy_path_unaffected(mock_backend):
    """A non-raising output_to_bool leaves error unset and honors the result."""
    result = await _validate_with_output_to_bool(mock_backend, lambda x: True)

    assert result.error is None
    assert bool(result) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
