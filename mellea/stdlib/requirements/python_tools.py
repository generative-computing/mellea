"""Requirement factories for Python tool invocation and code validation.

This module provides requirements for validating Python code, including syntax,
imports, and plotting. The python_tool_requirements() function bundles these
together, while specialized validators can be used independently.
"""

import ast
from collections.abc import Callable

from ...core import Context, Requirement, ValidationResult
from ..tools.interpreter import get_unauthorized_imports
from .plotting import python_plotting_requirements
from .python_reqs import extract_python_code
from .tool_reqs import tool_arg_validator, uses_tool


def _code_parses(code: str) -> tuple[bool, str | None]:
    """Check if code parses as valid Python.

    Validates syntax without executing code using AST parsing.

    Returns:
        (True, None) if code parses
        (False, error_message) if syntax error
    """
    try:
        ast.parse(code)
    except SyntaxError as e:
        error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
        if e.text:
            error_msg += f"\n  {e.text.rstrip()}"
            if e.offset:
                error_msg += "\n  " + " " * (e.offset - 1) + "^"
        return False, error_msg

    return True, None


# region Individual Requirement Validators


def _python_code_arg_present(arg_value: object) -> bool:
    """Return True when the python tool code argument is present and non-empty."""
    return isinstance(arg_value, str) and bool(arg_value.strip())


def _make_code_parses_validator() -> Callable[[Context], ValidationResult]:
    """Create a validator that checks if extracted code parses.

    This validator searches for Python code in the context, checking both
    direct python tool calls and markdown code blocks. The python tool is
    invoked synchronously as part of the LLM's response generation, so code
    is available in the context for validation at check time.
    """

    def validate(ctx: Context) -> ValidationResult:
        extraction_result = extract_python_code(ctx)
        if not extraction_result.as_bool() or extraction_result.reason is None:
            return ValidationResult(
                result=False,
                reason=(
                    "Could not extract Python code from your response. "
                    "Make sure to include code in the python tool call or "
                    "in ```python ... ``` blocks."
                ),
            )

        parses, error = _code_parses(extraction_result.reason)
        if not parses:
            return ValidationResult(
                result=False,
                reason=f"Your code contains a syntax error. {error}\n\nPlease fix the syntax and try again.",
            )

        return ValidationResult(result=True)

    return validate


def _make_imports_allowed_validator(
    allowed_imports: list[str] | None,
) -> Callable[[Context], ValidationResult]:
    """Create a validator that checks if code imports are in allowlist.

    This validator extracts Python code from the context (tool calls or markdown
    blocks) and checks that all imports are in the allowed list.
    """

    def validate(ctx: Context) -> ValidationResult:
        if allowed_imports is None:
            return ValidationResult(result=True)

        extraction_result = extract_python_code(ctx)
        if not extraction_result.as_bool() or extraction_result.reason is None:
            return ValidationResult(
                result=False, reason="Could not extract Python code"
            )

        unauthorized = get_unauthorized_imports(
            extraction_result.reason, allowed_imports
        )
        if unauthorized:
            allowed_str = ", ".join(sorted(set(allowed_imports)))
            return ValidationResult(
                result=False,
                reason=(
                    f"Your code imports forbidden modules: "
                    f"{', '.join(sorted(set(unauthorized)))}.\n"
                    f"You may only import: {allowed_str}\n"
                    f"Please rewrite your code without these imports."
                ),
            )

        return ValidationResult(result=True)

    return validate


# endregion


def python_tool_requirements(
    output_path: str | None = None,
    allowed_imports: list[str] | None = None,
    check_output_artifacts: bool | None = None,
) -> list[Requirement]:
    """Build requirements for Python code generation via the python tool.

    Args:
        output_path: Path where plotting output should be saved; enables plot-related checks.
        allowed_imports: List of allowed import module names; if provided, code must only import these.
        check_output_artifacts: Whether to verify output file exists after execution; auto-enabled if output_path is set.

    Returns:
        List of Requirement objects that validate python tool usage and code correctness.
    """
    reqs: list[Requirement] = []

    if check_output_artifacts is None:
        check_output_artifacts = output_path is not None

    reqs.append(uses_tool("python"))

    reqs.append(
        tool_arg_validator(
            description="The python tool call must include a code argument.",
            tool_name="python",
            arg_name="code",
            validation_fn=_python_code_arg_present,
        )
    )

    reqs.append(
        Requirement(
            description="The Python code must parse correctly.",
            validation_fn=_make_code_parses_validator(),
            check_only=False,
        )
    )

    if allowed_imports is not None:
        reqs.append(
            Requirement(
                description=f"Imports must be from allowed list: {', '.join(allowed_imports)}",
                validation_fn=_make_imports_allowed_validator(allowed_imports),
                check_only=False,
            )
        )

    reqs.extend(
        python_plotting_requirements(
            output_path=output_path, check_output_artifacts=check_output_artifacts
        )
    )

    return reqs
