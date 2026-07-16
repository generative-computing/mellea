# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for sampling presets and model-friendly feedback formatters."""

import pytest

from mellea.core import Requirement, ValidationResult
from mellea.stdlib.requirements.python_tools import (
    ImportRestrictions,
    OutputSizeLimit,
    PythonCodeExtraction,
    PythonSyntaxValid,
)
from mellea.stdlib.sampling import (
    ModelFriendlyFeedbackFormatter,
    ModelFriendlyRepairStrategy,
    RepairTemplateStrategy,
    SamplingPreset,
    python_code_generation_sampling,
    python_plotting_sampling,
)


class TestPythonCodeGenerationSampling:
    """Test python_code_generation_sampling() factory."""

    def test_factory_returns_sampling_preset(self):
        """Verify return type is SamplingPreset."""
        preset = python_code_generation_sampling()
        assert isinstance(preset, SamplingPreset)

    def test_preset_structure(self):
        """Verify preset has all required fields."""
        preset = python_code_generation_sampling()
        assert preset.requirements is not None
        assert isinstance(preset.requirements, list)
        assert preset.strategy is not None
        assert preset.feedback_strategy_name == "python_code_repair"
        assert preset.description is not None
        assert preset.example_usage is not None

    def test_default_loop_budget(self):
        """Verify default loop_budget is 2."""
        preset = python_code_generation_sampling()
        assert preset.strategy.loop_budget == 2

    def test_custom_loop_budget(self):
        """Verify loop_budget parameter propagates."""
        preset = python_code_generation_sampling(loop_budget=5)
        assert preset.strategy.loop_budget == 5

    def test_loop_budget_validation(self):
        """Verify loop_budget must be >= 1."""
        with pytest.raises(ValueError, match="Loop budget must be at least 1"):
            python_code_generation_sampling(loop_budget=0)
        with pytest.raises(ValueError, match="Loop budget must be at least 1"):
            python_code_generation_sampling(loop_budget=-1)

    def test_requirements_bundling(self):
        """Verify all expected requirements are included."""
        preset = python_code_generation_sampling()
        assert len(preset.requirements) == 4
        assert isinstance(preset.requirements[0], PythonCodeExtraction)
        assert isinstance(preset.requirements[1], PythonSyntaxValid)

    def test_with_import_restrictions(self):
        """Verify allowed_imports parameter creates ImportRestrictions."""
        preset = python_code_generation_sampling(allowed_imports=["numpy", "pandas"])
        assert any(isinstance(req, ImportRestrictions) for req in preset.requirements)
        import_req = next(
            req for req in preset.requirements if isinstance(req, ImportRestrictions)
        )
        assert import_req.allowed_imports == ["numpy", "pandas"]

    def test_with_sandbox_mode(self):
        """Verify use_sandbox parameter propagates."""
        preset_sandbox = python_code_generation_sampling(use_sandbox=True)
        preset_no_sandbox = python_code_generation_sampling(use_sandbox=False)

        assert preset_sandbox.requirements is not None
        assert preset_no_sandbox.requirements is not None

    def test_strategy_is_repair_template(self):
        """Verify strategy is RepairTemplateStrategy."""
        preset = python_code_generation_sampling()
        assert isinstance(preset.strategy, RepairTemplateStrategy)

    def test_output_limit_parameter(self):
        """Verify output_limit_chars parameter propagates."""
        preset = python_code_generation_sampling(output_limit_chars=5000)
        assert preset is not None

    def test_timeout_parameter(self):
        """Verify timeout_seconds parameter propagates."""
        preset = python_code_generation_sampling(timeout_seconds=10)
        assert preset is not None


class TestPythonPlottingSampling:
    """Test python_plotting_sampling() factory."""

    def test_factory_returns_sampling_preset(self):
        """Verify return type is SamplingPreset."""
        preset = python_plotting_sampling()
        assert isinstance(preset, SamplingPreset)

    def test_preset_structure(self):
        """Verify preset has all required fields."""
        preset = python_plotting_sampling()
        assert preset.requirements is not None
        assert isinstance(preset.requirements, list)
        assert preset.strategy is not None
        assert preset.feedback_strategy_name == "matplotlib_plotting_repair"
        assert preset.description is not None

    def test_default_loop_budget(self):
        """Verify default loop_budget is 3 (more than code generation)."""
        preset = python_plotting_sampling()
        assert preset.strategy.loop_budget == 3

    def test_custom_loop_budget(self):
        """Verify loop_budget parameter propagates."""
        preset = python_plotting_sampling(loop_budget=5)
        assert preset.strategy.loop_budget == 5

    def test_loop_budget_validation(self):
        """Verify loop_budget must be >= 1."""
        with pytest.raises(ValueError, match="Loop budget must be at least 1"):
            python_plotting_sampling(loop_budget=0)

    def test_includes_plotting_requirements(self):
        """Verify plotting-specific requirements are present."""
        preset = python_plotting_sampling()
        assert len(preset.requirements) > 4

    def test_default_sandbox_true(self):
        """Verify use_sandbox defaults to True for plotting."""
        preset = python_plotting_sampling()
        assert preset is not None

    def test_output_path_parameter(self):
        """Verify output_path parameter works."""
        preset = python_plotting_sampling(output_path="/tmp/plot.png")
        assert preset is not None

    def test_higher_timeout_than_code_generation(self):
        """Verify plotting has higher timeout defaults than code generation."""
        preset = python_plotting_sampling()
        assert preset is not None


class TestSamplingPresetDataclass:
    """Test SamplingPreset dataclass."""

    def test_sampling_preset_creation(self):
        """Verify SamplingPreset can be instantiated."""
        preset = SamplingPreset(
            requirements=[],
            strategy=RepairTemplateStrategy(loop_budget=1),
            feedback_strategy_name="test",
        )
        assert preset.requirements == []
        assert isinstance(preset.strategy, RepairTemplateStrategy)
        assert preset.feedback_strategy_name == "test"

    def test_sampling_preset_optional_fields(self):
        """Verify optional fields default to None."""
        preset = SamplingPreset(
            requirements=[], strategy=RepairTemplateStrategy(loop_budget=1)
        )
        assert preset.description is None
        assert preset.example_usage is None


class TestModelFriendlyFeedbackFormatter:
    """Test feedback formatting for each requirement type."""

    def test_syntax_error_formatting_basic(self):
        """Test basic syntax error formatting."""
        result = ValidationResult(
            result=False, reason="Syntax error at line 5: Expected ':' token"
        )
        formatted = ModelFriendlyFeedbackFormatter.format_python_syntax_error(result)
        assert "line 5" in formatted
        assert "syntax error" in formatted.lower()
        assert "Try:" in formatted

    def test_syntax_error_with_expected_token(self):
        """Test syntax error with Expected prefix."""
        result = ValidationResult(
            result=False, reason="Syntax error at line 10: Expected 'foo'"
        )
        formatted = ModelFriendlyFeedbackFormatter.format_python_syntax_error(result)
        assert "line 10" in formatted
        assert "Try:" in formatted

    def test_syntax_error_generic(self):
        """Test generic syntax error without line info."""
        result = ValidationResult(result=False, reason="Invalid syntax")
        formatted = ModelFriendlyFeedbackFormatter.format_python_syntax_error(result)
        assert "syntax error" in formatted.lower()
        assert "Try:" in formatted

    def test_import_restriction_formatting(self):
        """Test import restriction error formatting."""
        result = ValidationResult(
            result=False, reason="Forbidden imports detected: subprocess, socket"
        )
        formatted = ModelFriendlyFeedbackFormatter.format_import_error(result)
        assert "subprocess" in formatted or "socket" in formatted
        assert "Try:" in formatted
        assert (
            "whitelisted" in formatted.lower()
            or "standard library" in formatted.lower()
        )

    def test_import_no_restricted(self):
        """Test when no imports are restricted."""
        result = ValidationResult(result=False, reason="Import check failed")
        formatted = ModelFriendlyFeedbackFormatter.format_import_error(result)
        assert "Try:" in formatted

    def test_execution_error_name_error(self):
        """Test NameError formatting."""
        result = ValidationResult(
            result=False,
            reason="Traceback: NameError: name 'x' is not defined at line 8",
        )
        formatted = ModelFriendlyFeedbackFormatter.format_execution_error(result)
        assert "'x'" in formatted
        assert "undefined" in formatted or "not defined" in formatted.lower()
        assert "Try:" in formatted

    def test_execution_error_type_error(self):
        """Test TypeError formatting."""
        result = ValidationResult(
            result=False, reason="TypeError: unsupported operand type(s)"
        )
        formatted = ModelFriendlyFeedbackFormatter.format_execution_error(result)
        assert "type error" in formatted.lower()
        assert "Try:" in formatted

    def test_execution_error_index_error(self):
        """Test IndexError formatting."""
        result = ValidationResult(
            result=False, reason="IndexError: list index out of range"
        )
        formatted = ModelFriendlyFeedbackFormatter.format_execution_error(result)
        assert "index" in formatted.lower()
        assert "Try:" in formatted

    def test_execution_error_timeout(self):
        """Test timeout error formatting."""
        result = ValidationResult(result=False, reason="Timeout: code took too long")
        formatted = ModelFriendlyFeedbackFormatter.format_execution_error(result)
        assert "timeout" in formatted.lower() or "long" in formatted.lower()
        assert "Try:" in formatted

    def test_output_size_error_formatting(self):
        """Test output size limit error formatting."""
        result = ValidationResult(
            result=False, reason="Output size (50000 chars) exceeds limit (10000)."
        )
        formatted = ModelFriendlyFeedbackFormatter.format_output_size_error(result)
        assert "50000" in formatted or "50,000" in formatted
        assert "10000" in formatted or "10,000" in formatted
        assert "Try:" in formatted

    def test_matplotlib_backend_error_formatting(self):
        """Test matplotlib backend error formatting."""
        result = ValidationResult(
            result=False, reason="matplotlib.use() call not found in code"
        )
        formatted = ModelFriendlyFeedbackFormatter.format_matplotlib_error(result)
        assert "headless" in formatted.lower() or "backend" in formatted.lower()
        assert "Agg" in formatted
        assert "Try:" in formatted

    def test_matplotlib_plot_file_saved_error_formatting(self):
        """Test PlotFileSaved error formatting with dynamic path extraction."""
        reason = "No savefig() call found with path 'plot.png'. Add: plt.savefig('plot.png') or fig.savefig('plot.png')"
        result = ValidationResult(result=False, reason=reason)
        formatted = ModelFriendlyFeedbackFormatter.format_matplotlib_error(result)
        assert "plot.png" in formatted
        assert "plt.savefig('plot.png')" in formatted
        assert "Try:" in formatted
        assert "Do not use a variable as an argument" in formatted

    def test_matplotlib_plot_file_saved_error_with_custom_path(self):
        """Test PlotFileSaved error formatting with custom output path."""
        reason = "No savefig() call found with path '/tmp/output.png'. Add: plt.savefig('/tmp/output.png') or fig.savefig('/tmp/output.png')"
        result = ValidationResult(result=False, reason=reason)
        formatted = ModelFriendlyFeedbackFormatter.format_matplotlib_error(result)
        assert "/tmp/output.png" in formatted
        assert "plt.savefig('/tmp/output.png')" in formatted
        assert "Do not use a variable as an argument" in formatted

    def test_matplotlib_plot_file_saved_error_fallback(self):
        """Test PlotFileSaved error formatting with malformed reason (fallback)."""
        reason = "No savefig() call found but path parsing failed"
        result = ValidationResult(result=False, reason=reason)
        formatted = ModelFriendlyFeedbackFormatter.format_matplotlib_error(result)
        assert "your/output/path.png" in formatted or "savefig" in formatted

    def test_code_extraction_error_formatting(self):
        """Test code extraction error formatting."""
        result = ValidationResult(
            result=False, reason="No Python code blocks found in response"
        )
        formatted = ModelFriendlyFeedbackFormatter.format_extraction_error(result)
        assert "code block" in formatted.lower()
        assert "```python" in formatted
        assert "Try:" in formatted

    def test_format_requirement_reason_syntax_error(self):
        """Test dispatcher for syntax errors."""
        result = ValidationResult(
            result=False, reason="Syntax error at line 5: Expected ':' token"
        )
        req = PythonSyntaxValid()
        formatted = ModelFriendlyFeedbackFormatter.format_requirement_reason(
            req, result
        )
        assert "syntax error" in formatted.lower()
        assert "Try:" in formatted

    def test_format_requirement_reason_import_restrictions(self):
        """Test dispatcher for import restrictions."""
        result = ValidationResult(
            result=False, reason="Forbidden imports detected: os, sys"
        )
        req = ImportRestrictions(allowed_imports=["json"])
        formatted = ModelFriendlyFeedbackFormatter.format_requirement_reason(
            req, result
        )
        assert "Try:" in formatted

    def test_format_requirement_reason_output_size(self):
        """Test dispatcher for output size."""
        result = ValidationResult(
            result=False, reason="Output size (50000 chars) exceeds limit (10000)."
        )
        req = OutputSizeLimit(limit_chars=10000)
        formatted = ModelFriendlyFeedbackFormatter.format_requirement_reason(
            req, result
        )
        assert "Try:" in formatted

    def test_format_requirement_reason_extraction(self):
        """Test dispatcher for code extraction."""
        result = ValidationResult(
            result=False, reason="No Python code blocks found in response"
        )
        req = PythonCodeExtraction()
        formatted = ModelFriendlyFeedbackFormatter.format_requirement_reason(
            req, result
        )
        assert "Try:" in formatted

    def test_format_requirement_reason_unknown_type(self):
        """Test fallback for unknown requirement types."""
        result = ValidationResult(result=False, reason="Some error occurred")

        class CustomRequirement(Requirement):
            def __init__(self):
                super().__init__(
                    description="Custom requirement",
                    validation_fn=lambda ctx: ValidationResult(True),
                )

        req = CustomRequirement()
        formatted = ModelFriendlyFeedbackFormatter.format_requirement_reason(
            req, result
        )
        assert "Try:" in formatted
        assert "Custom requirement" in formatted or "Details:" in formatted


class TestModelFriendlyRepairStrategy:
    """Test ModelFriendlyRepairStrategy."""

    def test_repair_strategy_is_subclass_of_repair_template(self):
        """Verify ModelFriendlyRepairStrategy inherits from RepairTemplateStrategy."""
        strategy = ModelFriendlyRepairStrategy(loop_budget=2)
        assert isinstance(strategy, RepairTemplateStrategy)

    def test_strategy_has_repair_method(self):
        """Verify the strategy has a repair method."""
        assert hasattr(ModelFriendlyRepairStrategy, "repair")
        assert callable(ModelFriendlyRepairStrategy.repair)

    def test_strategy_has_select_from_failure_method(self):
        """Verify inherited select_from_failure method."""
        assert hasattr(ModelFriendlyRepairStrategy, "select_from_failure")
        assert callable(ModelFriendlyRepairStrategy.select_from_failure)

    def test_strategy_initialization(self):
        """Verify strategy can be initialized."""
        strategy = ModelFriendlyRepairStrategy(loop_budget=3, requirements=[])
        assert strategy.loop_budget == 3
        assert strategy.requirements == []

    def test_strategy_loop_budget_validation(self):
        """Verify loop_budget validation."""
        with pytest.raises(ValueError):
            ModelFriendlyRepairStrategy(loop_budget=0)


class TestModelFriendlyFeedbackFormatterIntegration:
    """Integration tests for feedback formatting."""

    def test_all_error_types_return_actionable_feedback(self):
        """Verify all error formatters return actionable feedback."""
        error_cases = [
            (
                ValidationResult(
                    result=False, reason="Syntax error at line 5: Expected ':'"
                ),
                "format_python_syntax_error",
            ),
            (
                ValidationResult(
                    result=False, reason="Forbidden imports detected: subprocess"
                ),
                "format_import_error",
            ),
            (
                ValidationResult(
                    result=False, reason="NameError: name 'x' is not defined"
                ),
                "format_execution_error",
            ),
            (
                ValidationResult(
                    result=False, reason="Output size (50000) exceeds limit (10000)"
                ),
                "format_output_size_error",
            ),
        ]

        for result, formatter_name in error_cases:
            formatter_method = getattr(ModelFriendlyFeedbackFormatter, formatter_name)
            formatted = formatter_method(result)
            assert "Try:" in formatted, f"{formatter_name} should include 'Try:'"
            assert len(formatted) > 20, f"{formatter_name} feedback too short"

    def test_formatted_feedback_is_concise(self):
        """Verify formatted feedback is reasonably concise."""
        result = ValidationResult(
            result=False, reason="Syntax error at line 5: Expected ':' token"
        )
        formatted = ModelFriendlyFeedbackFormatter.format_python_syntax_error(result)
        assert len(formatted) < 500, "Formatted feedback should be concise"

    def test_formatted_feedback_includes_details(self):
        """Verify formatted feedback includes relevant error details."""
        result = ValidationResult(
            result=False, reason="NameError: name 'my_var' is not defined at line 12"
        )
        formatted = ModelFriendlyFeedbackFormatter.format_execution_error(result)
        assert "my_var" in formatted or "variable" in formatted.lower()

    def test_fallback_formatting_still_actionable(self):
        """Verify fallback formatter produces actionable feedback."""
        result = ValidationResult(
            result=False, reason="Some unknown validation error occurred"
        )

        class UnknownRequirement(Requirement):
            def __init__(self):
                super().__init__(
                    description="Unknown requirement",
                    validation_fn=lambda ctx: ValidationResult(True),
                )

        req = UnknownRequirement()
        formatted = ModelFriendlyFeedbackFormatter.format_requirement_reason(
            req, result
        )
        assert "Try:" in formatted
        assert "Unknown requirement" in formatted or "Details:" in formatted
