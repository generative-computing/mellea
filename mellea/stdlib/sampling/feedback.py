# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-friendly feedback formatters for validation failures.

This module converts generic validation failure reasons into actionable guidance
that language models can understand and act upon. Different requirement types
(syntax errors, import violations, execution failures, etc.) receive specialized
formatting that highlights what went wrong and suggests concrete fixes.

The key insight is that LLM repair performs better when feedback is:
1. **Specific**: Names the exact error location and type
2. **Actionable**: Suggests concrete steps to fix it
3. **Concise**: Avoids unnecessary details that confuse the model

For example:
- Generic: "Syntax error at line 5"
- Model-friendly: "Your code has a syntax error on line 5. Try: Add a colon ':'
  at the end of your if statement."
"""

import re
from typing import Any

from ...core import (
    Backend,
    Component,
    Context,
    MelleaLogger,
    Requirement,
    ValidationResult,
)
from ..components import Instruction
from ..requirements.python_tools import (
    ImportRestrictions,
    OutputSizeLimit,
    PythonCodeExtraction,
    PythonSyntaxValid,
)
from .base import RepairTemplateStrategy

logger = MelleaLogger.get_logger()

# Pre-compiled regex patterns for performance (avoid recompilation in hot paths)
_SYNTAX_ERROR_PATTERN = re.compile(r"at line (\d+)(?:,\s*column \d+)?:\s*(.+)")
_FORBIDDEN_IMPORTS_PATTERN = re.compile(
    r"Forbidden imports[:\s]*([^.]+)", re.IGNORECASE
)
_NAME_ERROR_PATTERN = re.compile(r"name '([^']+)' is not defined")
_TYPE_ERROR_PATTERN = re.compile(
    r"unsupported operand type\(s\) for (.+):|expected .+ but got .+|'(.+)' object"
)
_INDEX_ERROR_PATTERN = re.compile(r"index out of range|list index out of range")
_KEY_ERROR_PATTERN = re.compile(r"key '(.+)' not found|KeyError: '(.+)'")
_ZERO_DIVISION_PATTERN = re.compile(r"(integer )?division by zero|modulo by zero")
_OUTPUT_SIZE_PATTERN = re.compile(r"(\d+)\s+chars?\)?\s+exceeds\s+limit\s+\((\d+)")
_MATPLOTLIB_BACKEND_PATTERN = re.compile(r"matplotlib|Matplotlib|backend|display")
_EXPECTED_TOKEN_PATTERN = re.compile(r"expected (.+)")


class ModelFriendlyFeedbackFormatter:
    """Converts validation failures into model-friendly repair instructions.

    This class provides static methods to format each requirement type's error
    messages into actionable guidance. Each formatter takes a ValidationResult
    and produces a concise, model-understandable message.

    The formatters are designed to be called from sampling strategies (like
    RepairTemplateStrategy) to improve the quality of repair feedback.
    """

    @staticmethod
    def format_python_syntax_error(validation_result: ValidationResult) -> str:
        """Format syntax errors into actionable guidance.

        Input: "Syntax error at line 5: Expected ':' token"
        Output: "Your code has a syntax error on line 5. Try: Add a colon ':' at
                 the end of your statement."

        Args:
            validation_result: ValidationResult from PythonSyntaxValid.

        Returns:
            Model-friendly feedback string.
        """
        reason = validation_result.reason or "Syntax error"

        if "at line" in reason and ":" in reason:
            parts = reason.split(":", 1)
            location = parts[0].strip()
            error_desc = parts[1].strip() if len(parts) > 1 else "Unknown syntax error"

            if "Expected" in error_desc:
                return (
                    f"Your code has a syntax error ({location}). "
                    f"The parser {error_desc.lower()}. "
                    f"Try: Review your statement and add any missing punctuation or keywords."
                )
            else:
                return (
                    f"Your code has a syntax error ({location}): {error_desc}. "
                    f"Try: Check the line and fix any malformed code."
                )
        else:
            return (
                f"Your code has a syntax error. Try: Check that all code blocks are "
                f"properly indented and all statements end correctly. Details: {reason}"
            )

    @staticmethod
    def format_import_error(validation_result: ValidationResult) -> str:
        """Format import restriction violations into actionable guidance.

        Input: "Forbidden imports detected: subprocess, socket"
        Output: "Your code imports forbidden modules: subprocess, socket. These are
                 not available. Try: Use only allowed modules like numpy, json."

        Args:
            validation_result: ValidationResult from ImportRestrictions.

        Returns:
            Model-friendly feedback string.
        """
        reason = validation_result.reason or "Import error"

        if "Forbidden imports" in reason:
            match = _FORBIDDEN_IMPORTS_PATTERN.search(reason)
            if match:
                forbidden = match.group(1).strip()
                return (
                    f"Your code imports forbidden modules: {forbidden}. These are not "
                    f"available in this environment. Try: Remove these imports and use "
                    f"only standard library or whitelisted modules instead."
                )

        return (
            f"Your code imports modules that are not allowed. "
            f"Try: Check your imports and remove any forbidden modules. "
            f"Details: {reason}"
        )

    @staticmethod
    def format_execution_error(validation_result: ValidationResult) -> str:
        """Format runtime/execution errors into actionable guidance.

        Input: "Traceback: NameError: name 'x' is not defined at line 8"
        Output: "Your code has a runtime error: variable 'x' is not defined on line 8.
                 Try: Check that all variables are defined before use."

        Args:
            validation_result: ValidationResult from PythonExecutionReq.

        Returns:
            Model-friendly feedback string.
        """
        reason = validation_result.reason or "Execution failed"

        if "NameError" in reason:
            match = _NAME_ERROR_PATTERN.search(reason)
            if match:
                var_name = match.group(1)
                return (
                    f"Your code has a runtime error: variable '{var_name}' is not "
                    f"defined. Try: Make sure all variables are assigned before use. "
                    f"Check for typos in variable names."
                )
            return (
                f"Your code has a NameError. Try: Check that all variables are "
                f"defined before use. Details: {reason}"
            )
        elif "TypeError" in reason:
            return (
                f"Your code has a type error (wrong type of argument). "
                f"Try: Check that you're passing the correct types to functions. "
                f"Details: {reason}"
            )
        elif "IndexError" in reason:
            return (
                f"Your code has an index error (accessing a list/array out of bounds). "
                f"Try: Check that your indices are within valid range. "
                f"Details: {reason}"
            )
        elif "KeyError" in reason:
            return (
                f"Your code has a key error (accessing a dictionary with missing key). "
                f"Try: Check that the key exists in the dictionary before accessing it. "
                f"Details: {reason}"
            )
        elif "ZeroDivisionError" in reason:
            return (
                f"Your code has a division by zero error. "
                f"Try: Add a check to ensure the divisor is not zero before dividing. "
                f"Details: {reason}"
            )
        elif "Timeout" in reason or "timeout" in reason.lower():
            return (
                "Your code took too long to execute and was terminated. "
                "Try: Look for infinite loops or very slow operations. "
                "Optimize your algorithm or add early exit conditions."
            )
        else:
            return (
                f"Your code execution failed. Try: Check the error details and fix "
                f"any issues. Common problems: undefined variables, type errors, "
                f"or infinite loops. Details: {reason}"
            )

    @staticmethod
    def format_output_size_error(validation_result: ValidationResult) -> str:
        """Format output size limit violations into actionable guidance.

        Input: "Output size (50000 chars) exceeds limit (10000)."
        Output: "Your code produces too much output (50000 chars, limit is 10000).
                 Try: Reduce printed output or logging."

        Args:
            validation_result: ValidationResult from OutputSizeLimit.

        Returns:
            Model-friendly feedback string.
        """
        reason = validation_result.reason or "Output exceeds limit"

        match = _OUTPUT_SIZE_PATTERN.search(reason)
        if match:
            actual = match.group(1)
            limit = match.group(2)
            return (
                f"Your code produces too much output ({actual} characters, "
                f"limit is {limit}). Try: Remove unnecessary print statements or "
                f"logging. Print only the final result."
            )
        else:
            return (
                f"Your code output is too large. Try: Print less information. "
                f"Remove logging or debug output statements. Details: {reason}"
            )

    @staticmethod
    def format_matplotlib_error(validation_result: ValidationResult) -> str:
        """Format matplotlib-specific errors into actionable guidance.

        Input: "matplotlib.use() call not found in code"
        Output: "Your code doesn't set up a headless backend. Try: Add
                 'import matplotlib; matplotlib.use('Agg')' at the start."

        Args:
            validation_result: ValidationResult from matplotlib requirements.

        Returns:
            Model-friendly feedback string.
        """
        reason = validation_result.reason or "Matplotlib error"

        if "matplotlib.use()" in reason or "backend" in reason.lower():
            return (
                "Your code doesn't configure matplotlib for headless rendering. "
                "Try: Add these lines at the start of your code:\n"
                "    import matplotlib\n"
                "    matplotlib.use('Agg')"
            )
        elif "No savefig() call found" in reason:
            # Extract the expected path from the reason string so the repair
            # instruction names the actual target path, not a hardcoded placeholder.
            path_match = re.search(r"with path '([^']+)'", reason)
            expected_path = (
                path_match.group(1) if path_match else "your/output/path.png"
            )
            return (
                f"Your code didn't save the plot to the expected file. "
                f"Try: Add plt.savefig('{expected_path}') before plt.show() or plt.close(). "
                f"Do not use a variable as an argument of plt.savefig.  Use a string literal as the argument like plt.savefig('{expected_path}'). "
            )
        else:
            return (
                f"Your matplotlib code failed. Try: Check that you're using the "
                f"'Agg' backend and saving plots correctly with 'plt.savefig()'. "
                f"Details: {reason}"
            )

    @staticmethod
    def format_extraction_error(validation_result: ValidationResult) -> str:
        """Format code extraction errors into actionable guidance.

        Input: "No Python code blocks found in response"
        Output: "Your response doesn't contain a code block. Try: Make sure to
                 include your code in a ```python ... ``` block."

        Args:
            validation_result: ValidationResult from PythonCodeExtraction.

        Returns:
            Model-friendly feedback string.
        """
        reason = validation_result.reason or "Could not extract code"

        if "No Python code blocks" in reason:
            return (
                "Your response doesn't contain a Python code block. Try: "
                "Make sure to include your code in a ```python ... ``` block."
            )
        else:
            return (
                f"Could not extract code from your response. Try: Include your "
                f"code in a ```python ... ``` code block with clear markers. "
                f"Details: {reason}"
            )

    @classmethod
    def format_requirement_reason(
        cls, requirement: Requirement, validation_result: ValidationResult
    ) -> str:
        """Intelligently format feedback based on requirement type.

        Dispatches to specific formatter methods based on the requirement's
        type. Falls back to generic formatting if no specific handler exists.

        Args:
            requirement: The Requirement instance that failed.
            validation_result: The ValidationResult from the failed check.

        Returns:
            Model-friendly feedback string.
        """
        requirement_type: type[Requirement] = type(requirement)

        formatters: dict[type[Requirement], Any] = {
            PythonSyntaxValid: cls.format_python_syntax_error,
            ImportRestrictions: cls.format_import_error,
            OutputSizeLimit: cls.format_output_size_error,
            PythonCodeExtraction: cls.format_extraction_error,
        }

        formatter = formatters.get(requirement_type)
        if formatter:
            return formatter(validation_result)

        type_name = requirement_type.__name__
        if "Matplotlib" in type_name or "Plot" in type_name:
            return cls.format_matplotlib_error(validation_result)

        if "Execution" in type_name:
            return cls.format_execution_error(validation_result)

        return (
            f"{requirement.description} (failed). Try: Review your code and "
            f"ensure it meets this requirement. Details: {validation_result.reason}"
        )


class ModelFriendlyRepairStrategy(RepairTemplateStrategy):
    """RepairTemplateStrategy with model-friendly feedback formatting.

    Extends RepairTemplateStrategy to use ModelFriendlyFeedbackFormatter for
    converting validation failures into actionable repair guidance. This typically
    improves LLM performance on repair tasks compared to generic validation reasons.

    Example:
        >>> strategy = ModelFriendlyRepairStrategy(loop_budget=2, requirements=[...])
        >>> result = session.instruct(
        ...     "Write Python code",
        ...     requirements=[...],
        ...     strategy=strategy,
        ... )
    """

    @staticmethod
    def repair(
        old_ctx: Context,
        new_ctx: Context,
        past_actions: list[Component],
        past_results: list[Any],
        past_val: list[list[tuple[Requirement, ValidationResult]]],
    ) -> tuple[Component, Context]:
        """Repair with model-friendly feedback formatting.

        Identical to RepairTemplateStrategy.repair() but uses
        ModelFriendlyFeedbackFormatter to format each failure reason.

        Args:
            old_ctx: Context without the failed action output.
            new_ctx: Context including the failed action output.
            past_actions: Previous actions executed.
            past_results: Previous generation results.
            past_val: Previous validation results for each requirement.

        Returns:
            Tuple of (repaired action component, original context).
        """
        pa = past_actions[-1]
        if isinstance(pa, Instruction):
            failed_items = [
                (req, val) for req, val in past_val[-1] if not val.as_bool()
            ]

            repair_lines = []
            for req, validation in failed_items:
                formatted_reason = (
                    ModelFriendlyFeedbackFormatter.format_requirement_reason(
                        req, validation
                    )
                )
                repair_lines.append(f"* {formatted_reason}")

            repair_string = "The following requirements failed before:\n" + "\n".join(
                repair_lines
            )

            return pa.copy_and_repair(repair_string=repair_string), old_ctx
        return pa, old_ctx
