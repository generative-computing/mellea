"""Tests for built-in debug plugins (generation, sampling, validation).

Verifies that each built-in debug plugin hook fires correctly and logs expected
output, including edge cases with missing or None-valued payload fields.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

pytest.importorskip("cpex.framework")

from mellea.plugins import HookType, hook, register
from mellea.plugins.builtin_debug import (
    log_generation_post_call,
    log_generation_pre_call,
    log_sampling_iteration,
    log_sampling_loop_end,
    log_sampling_loop_start,
    log_sampling_repair,
    log_validation_post_check,
    log_validation_pre_check,
)
from mellea.plugins.builtin_debug.generation import (
    _get_prompt_preview,
    _get_response_preview,
    _get_token_usage,
)
from mellea.plugins.hooks.generation import (
    GenerationPostCallPayload,
    GenerationPreCallPayload,
)
from mellea.plugins.hooks.sampling import (
    SamplingIterationPayload,
    SamplingLoopEndPayload,
    SamplingLoopStartPayload,
    SamplingRepairPayload,
)
from mellea.plugins.hooks.validation import (
    ValidationPostCheckPayload,
    ValidationPreCheckPayload,
)
from mellea.plugins.manager import invoke_hook, shutdown_plugins


@pytest.fixture(autouse=True)
async def _reset_plugins():
    """Shut down plugins before and after each test for isolation."""
    await shutdown_plugins()
    yield
    await shutdown_plugins()


# ---------------------------------------------------------------------------
# Generation plugin tests
# ---------------------------------------------------------------------------


class TestGenerationPreCallPlugin:
    """Tests for log_generation_pre_call hook."""

    async def test_generation_pre_call_fires_and_logs(self, caplog) -> None:
        """Pre-call hook fires and logs request details."""
        register(log_generation_pre_call)

        payload = GenerationPreCallPayload(
            action="test action", context=MagicMock(), generation_id="gen-123"
        )

        with caplog.at_level(
            logging.DEBUG, logger="mellea.plugins.builtin_debug.generation"
        ):
            await invoke_hook(HookType.GENERATION_PRE_CALL, payload)

        assert any("GEN-PRE-CALL" in r.message for r in caplog.records)
        assert any("gen_id=gen-123" in r.message for r in caplog.records)

    async def test_generation_pre_call_handles_none_action(self, caplog) -> None:
        """Pre-call hook gracefully handles None action."""
        register(log_generation_pre_call)

        payload = GenerationPreCallPayload(
            action=None, context=MagicMock(), generation_id="gen-456"
        )

        with caplog.at_level(
            logging.DEBUG, logger="mellea.plugins.builtin_debug.generation"
        ):
            await invoke_hook(HookType.GENERATION_PRE_CALL, payload)

        assert any("GEN-PRE-CALL" in r.message for r in caplog.records)
        assert any("(no action)" in r.message for r in caplog.records)

    async def test_generation_pre_call_handles_empty_generation_id(
        self, caplog
    ) -> None:
        """Pre-call hook handles payload with None generation_id."""
        register(log_generation_pre_call)

        payload = GenerationPreCallPayload(
            action="test", context=MagicMock(), generation_id=None
        )

        with caplog.at_level(
            logging.DEBUG, logger="mellea.plugins.builtin_debug.generation"
        ):
            await invoke_hook(HookType.GENERATION_PRE_CALL, payload)

        assert any("GEN-PRE-CALL" in r.message for r in caplog.records)
        assert any("no-id" in r.message for r in caplog.records)


class TestGenerationPostCallPlugin:
    """Tests for log_generation_post_call hook."""

    async def test_generation_post_call_fires_and_logs(self, caplog) -> None:
        """Post-call hook fires and logs response details."""
        register(log_generation_post_call)

        model_output = MagicMock()
        model_output.value = "test response"
        model_output.generation = MagicMock()
        model_output.generation.model = "test-model"
        model_output.generation.usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

        payload = GenerationPostCallPayload(
            model_output=model_output, generation_id="gen-123", latency_ms=150.5
        )

        with caplog.at_level(
            logging.DEBUG, logger="mellea.plugins.builtin_debug.generation"
        ):
            await invoke_hook(HookType.GENERATION_POST_CALL, payload)

        assert any("GEN-POST-CALL" in r.message for r in caplog.records)
        assert any("gen_id=gen-123" in r.message for r in caplog.records)
        assert any("test response" in r.message for r in caplog.records)

    async def test_generation_post_call_handles_none_model_output(self, caplog) -> None:
        """Post-call hook handles None model_output."""
        register(log_generation_post_call)

        payload = GenerationPostCallPayload(
            model_output=None, generation_id="gen-456", latency_ms=0.0
        )

        with caplog.at_level(
            logging.DEBUG, logger="mellea.plugins.builtin_debug.generation"
        ):
            await invoke_hook(HookType.GENERATION_POST_CALL, payload)

        assert any("GEN-POST-CALL" in r.message for r in caplog.records)
        assert any("(no output)" in r.message for r in caplog.records)

    async def test_generation_post_call_handles_none_value(self, caplog) -> None:
        """Post-call hook handles model_output.value = None."""
        register(log_generation_post_call)

        model_output = MagicMock()
        model_output.value = None
        model_output.generation = MagicMock()
        model_output.generation.model = "test-model"

        payload = GenerationPostCallPayload(
            model_output=model_output, generation_id="gen-789", latency_ms=0.0
        )

        with caplog.at_level(
            logging.DEBUG, logger="mellea.plugins.builtin_debug.generation"
        ):
            await invoke_hook(HookType.GENERATION_POST_CALL, payload)

        assert any("GEN-POST-CALL" in r.message for r in caplog.records)
        assert any("(no value)" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Helper function tests for generation plugin
# ---------------------------------------------------------------------------


class TestGenerationPluginHelpers:
    """Tests for generation plugin helper functions."""

    def test_get_prompt_preview_with_valid_action(self) -> None:
        """_get_prompt_preview returns short action string."""
        payload = MagicMock()
        payload.action = "test action"

        result = _get_prompt_preview(payload)
        assert "test action" in result

    def test_get_prompt_preview_with_none_action(self) -> None:
        """_get_prompt_preview handles None action."""
        payload = MagicMock()
        payload.action = None

        result = _get_prompt_preview(payload)
        assert result == "(no action)"

    def test_get_prompt_preview_with_string_action(self) -> None:
        """_get_prompt_preview with plain string action."""
        payload = MagicMock()
        payload.action = "simple string action"

        result = _get_prompt_preview(payload)
        assert "simple string action" in result

    def test_get_prompt_preview_truncates_long_text(self) -> None:
        """_get_prompt_preview truncates long action strings."""
        payload = MagicMock()
        long_action = "x" * 200
        payload.action = long_action

        result = _get_prompt_preview(payload)
        assert len(result) <= 100
        assert "..." in result

    def test_get_response_preview_with_valid_output(self) -> None:
        """_get_response_preview returns short response string."""
        payload = MagicMock()
        model_output = MagicMock()
        model_output.value = "test response"
        payload.model_output = model_output

        result = _get_response_preview(payload)
        assert "test response" in result

    def test_get_response_preview_with_none_model_output(self) -> None:
        """_get_response_preview handles None model_output."""
        payload = MagicMock()
        payload.model_output = None

        result = _get_response_preview(payload)
        assert result == "(no output)"

    def test_get_response_preview_with_none_value(self) -> None:
        """_get_response_preview handles None value."""
        payload = MagicMock()
        model_output = MagicMock()
        model_output.value = None
        payload.model_output = model_output

        result = _get_response_preview(payload)
        assert result == "(no value)"

    def test_get_response_preview_with_multiline_value(self) -> None:
        """_get_response_preview normalizes newlines and spaces."""
        payload = MagicMock()
        model_output = MagicMock()
        model_output.value = "line1\nline2  extra  spaces"
        payload.model_output = model_output

        result = _get_response_preview(payload)
        assert "\n" not in result
        assert "  " not in result
        assert "line1 line2 extra spaces" in result

    def test_get_token_usage_with_valid_usage(self) -> None:
        """_get_token_usage extracts token counts."""
        payload = MagicMock()
        model_output = MagicMock()
        gen = MagicMock()
        gen.usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        model_output.generation = gen
        payload.model_output = model_output

        result = _get_token_usage(payload)
        assert "10" in result
        assert "5" in result
        assert "15" in result

    def test_get_token_usage_with_missing_fields(self) -> None:
        """_get_token_usage returns unknown when usage dict is empty."""
        payload = MagicMock()
        model_output = MagicMock()
        gen = MagicMock()
        gen.usage = {}
        model_output.generation = gen
        payload.model_output = model_output

        result = _get_token_usage(payload)
        assert result == "unknown"


# ---------------------------------------------------------------------------
# Sampling plugin tests
# ---------------------------------------------------------------------------


class TestSamplingLoopStartPlugin:
    """Tests for log_sampling_loop_start hook."""

    async def test_sampling_loop_start_fires_and_logs(self, caplog) -> None:
        """Loop start hook fires and logs strategy info."""
        register(log_sampling_loop_start)

        req1 = MagicMock()
        req1.description = "Requirement 1"

        payload = SamplingLoopStartPayload(
            strategy_name="RejectionSampling", loop_budget=10, requirements=[req1]
        )

        with caplog.at_level(
            logging.DEBUG, logger="mellea.plugins.builtin_debug.sampling"
        ):
            await invoke_hook(HookType.SAMPLING_LOOP_START, payload)

        assert any("SAMPLING-START" in r.message for r in caplog.records)
        assert any("RejectionSampling" in r.message for r in caplog.records)
        assert any("loop_budget=10" in r.message for r in caplog.records)


class TestSamplingIterationPlugin:
    """Tests for log_sampling_iteration hook."""

    async def test_sampling_iteration_logs_success(self, caplog) -> None:
        """Iteration hook logs successful validation."""
        register(log_sampling_iteration)

        req1 = MagicMock()
        req1.description = "Requirement 1"
        result1 = MagicMock()
        result1.as_bool.return_value = True

        payload = SamplingIterationPayload(
            iteration=1,
            valid_count=1,
            total_count=1,
            all_validations_passed=True,
            validation_results=[(req1, result1)],
        )

        with caplog.at_level(
            logging.DEBUG, logger="mellea.plugins.builtin_debug.sampling"
        ):
            await invoke_hook(HookType.SAMPLING_ITERATION, payload)

        assert any("SAMPLING-ITER 1" in r.message for r in caplog.records)
        assert any("SUCCESS" in r.message for r in caplog.records)

    async def test_sampling_iteration_logs_failure(self, caplog) -> None:
        """Iteration hook logs failed validation."""
        register(log_sampling_iteration)

        req1 = MagicMock()
        req1.description = "Requirement 1"
        result1 = MagicMock()
        result1.as_bool.return_value = False
        result1.reason = "output too long"

        payload = SamplingIterationPayload(
            iteration=2,
            valid_count=0,
            total_count=1,
            all_validations_passed=False,
            validation_results=[(req1, result1)],
        )

        with caplog.at_level(
            logging.DEBUG, logger="mellea.plugins.builtin_debug.sampling"
        ):
            await invoke_hook(HookType.SAMPLING_ITERATION, payload)

        assert any("SAMPLING-ITER 2" in r.message for r in caplog.records)
        assert any("FAILED" in r.message for r in caplog.records)


class TestSamplingRepairPlugin:
    """Tests for log_sampling_repair hook."""

    async def test_sampling_repair_fires_and_logs(self, caplog) -> None:
        """Repair hook fires and logs repair trigger."""
        register(log_sampling_repair)

        req1 = MagicMock()
        req1.description = "Requirement 1"
        result1 = MagicMock()
        result1.as_bool.return_value = False

        payload = SamplingRepairPayload(
            repair_iteration=2,
            repair_type="instructional",
            failed_validations=[(req1, result1)],
        )

        with caplog.at_level(
            logging.DEBUG, logger="mellea.plugins.builtin_debug.sampling"
        ):
            await invoke_hook(HookType.SAMPLING_REPAIR, payload)

        assert any("REPAIR-TRIGGERED" in r.message for r in caplog.records)
        assert any("iteration 2" in r.message for r in caplog.records)
        assert any("instructional" in r.message for r in caplog.records)


class TestSamplingLoopEndPlugin:
    """Tests for log_sampling_loop_end hook."""

    async def test_sampling_loop_end_logs_success(self, caplog) -> None:
        """Loop end hook logs successful completion."""
        register(log_sampling_loop_end)

        req1 = MagicMock()
        result1 = MagicMock()
        result1.as_bool.return_value = True

        payload = SamplingLoopEndPayload(
            strategy_name="RejectionSampling",
            success=True,
            iterations_used=3,
            failure_reason="",
            all_results=["result1", "result2", "result3"],
            all_validations=[[(req1, result1)]],
        )

        with caplog.at_level(
            logging.DEBUG, logger="mellea.plugins.builtin_debug.sampling"
        ):
            await invoke_hook(HookType.SAMPLING_LOOP_END, payload)

        assert any("SAMPLING-END" in r.message for r in caplog.records)
        assert any("SUCCESS" in r.message for r in caplog.records)
        assert any("3 iteration" in r.message for r in caplog.records)

    async def test_sampling_loop_end_logs_failure(self, caplog) -> None:
        """Loop end hook logs failed completion."""
        register(log_sampling_loop_end)

        payload = SamplingLoopEndPayload(
            strategy_name="RejectionSampling",
            success=False,
            iterations_used=10,
            failure_reason="budget exceeded",
            all_results=["r1", "r2"] * 5,
            all_validations=[],
        )

        with caplog.at_level(
            logging.DEBUG, logger="mellea.plugins.builtin_debug.sampling"
        ):
            await invoke_hook(HookType.SAMPLING_LOOP_END, payload)

        assert any("SAMPLING-END" in r.message for r in caplog.records)
        assert any("FAILED" in r.message for r in caplog.records)
        assert any("budget exceeded" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Validation plugin tests
# ---------------------------------------------------------------------------


class TestValidationPreCheckPlugin:
    """Tests for log_validation_pre_check hook."""

    async def test_validation_pre_check_fires_and_logs(self, caplog) -> None:
        """Pre-check hook fires and logs requirement setup."""
        register(log_validation_pre_check)

        req1 = MagicMock()
        req1.description = "Requirement 1"

        payload = ValidationPreCheckPayload(requirements=[req1], target="test target")

        with caplog.at_level(
            logging.DEBUG, logger="mellea.plugins.builtin_debug.validation"
        ):
            await invoke_hook(HookType.VALIDATION_PRE_CHECK, payload)

        assert any("VALIDATION-PRE-CHECK" in r.message for r in caplog.records)
        assert any("requirements=1" in r.message for r in caplog.records)

    async def test_validation_pre_check_handles_none_target(self, caplog) -> None:
        """Pre-check hook handles None target."""
        register(log_validation_pre_check)

        req1 = MagicMock()
        req1.description = "Requirement 1"

        payload = ValidationPreCheckPayload(requirements=[req1], target=None)

        with caplog.at_level(
            logging.DEBUG, logger="mellea.plugins.builtin_debug.validation"
        ):
            await invoke_hook(HookType.VALIDATION_PRE_CHECK, payload)

        assert any("VALIDATION-PRE-CHECK" in r.message for r in caplog.records)
        assert any("target=None" in r.message for r in caplog.records)


class TestValidationPostCheckPlugin:
    """Tests for log_validation_post_check hook."""

    async def test_validation_post_check_logs_all_passed(self, caplog) -> None:
        """Post-check hook logs all passed validation."""
        register(log_validation_post_check)

        req1 = MagicMock()
        req1.description = "Requirement 1"
        result1 = MagicMock()
        result1.as_bool.return_value = True
        result1.reason = None

        payload = ValidationPostCheckPayload(
            requirements=[req1],
            passed_count=1,
            failed_count=0,
            all_validations_passed=True,
            results=[result1],
        )

        with caplog.at_level(
            logging.DEBUG, logger="mellea.plugins.builtin_debug.validation"
        ):
            await invoke_hook(HookType.VALIDATION_POST_CHECK, payload)

        assert any("VALIDATION-POST-CHECK" in r.message for r in caplog.records)
        assert any("ALL PASSED" in r.message for r in caplog.records)
        assert any("1/1" in r.message for r in caplog.records)

    async def test_validation_post_check_logs_mixed_results(self, caplog) -> None:
        """Post-check hook logs mixed validation results."""
        register(log_validation_post_check)

        req1 = MagicMock()
        req1.description = "Requirement 1"
        result1 = MagicMock()
        result1.as_bool.return_value = False
        result1.reason = "constraint failed"
        result1.score = None

        req2 = MagicMock()
        req2.description = "Requirement 2"
        result2 = MagicMock()
        result2.as_bool.return_value = True
        result2.reason = None

        payload = ValidationPostCheckPayload(
            requirements=[req1, req2],
            passed_count=1,
            failed_count=1,
            all_validations_passed=False,
            results=[result1, result2],
        )

        with caplog.at_level(
            logging.DEBUG, logger="mellea.plugins.builtin_debug.validation"
        ):
            await invoke_hook(HookType.VALIDATION_POST_CHECK, payload)

        assert any("VALIDATION-POST-CHECK" in r.message for r in caplog.records)
        assert any("MIXED RESULTS" in r.message for r in caplog.records)
        assert any("1/2 passed" in r.message for r in caplog.records)

    async def test_validation_post_check_handles_result_mismatch(self, caplog) -> None:
        """Post-check hook detects mismatch between requirements and results."""
        register(log_validation_post_check)

        req1 = MagicMock()
        req1.description = "Requirement 1"
        req2 = MagicMock()
        req2.description = "Requirement 2"

        result1 = MagicMock()
        result1.as_bool.return_value = False
        result1.reason = None
        result1.score = None  # Ensure score is None to avoid format error

        payload = ValidationPostCheckPayload(
            requirements=[req1, req2],  # 2 requirements
            passed_count=0,
            failed_count=2,
            all_validations_passed=False,
            results=[result1],  # Only 1 result - mismatch!
        )

        with caplog.at_level(
            logging.DEBUG, logger="mellea.plugins.builtin_debug.validation"
        ):
            await invoke_hook(HookType.VALIDATION_POST_CHECK, payload)

        # Check that the mismatch warning was logged
        records = [r for r in caplog.records if "Result mismatch" in r.message]
        assert len(records) > 0, (
            f"Expected mismatch warning, got records: {[r.message for r in caplog.records]}"
        )
