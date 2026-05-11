"""Requirement factories for Python tool invocation and code validation.

This module provides generic requirements for Python-tool usage and code
correctness. Plotting-specific checks are exposed separately through
``plotting.python_plotting_requirements(...)`` so they are not implied to be
universal Python-tool requirements.
"""

from collections.abc import Callable

from ...core import Context, Requirement, ValidationResult
from ..tools.interpreter import StaticAnalysisEnvironment
from .imports import get_unauthorized_imports
from .plotting import python_plotting_requirements
from .python_reqs import extract_python_code
from .tool_reqs import tool_arg_validator, uses_tool


def _code_parses(code: str) -> tuple[bool, str | None]:
    """Check if code parses as valid Python using StaticAnalysisEnvironment.

    Validates syntax without executing code. Reuses StaticAnalysisEnvironment
    to avoid duplicating AST parsing logic.

    Returns:
        (True, None) if code parses
        (False, error_message) if syntax error
    """
    env = StaticAnalysisEnvironment(allowed_imports=None)
    result = env.execute(code, timeout=0)

    if not result.success and isinstance(result.analysis_result, SyntaxError):
        e = result.analysis_result
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
    """Create a validator that checks if extracted code parses."""

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
    """Create a validator that checks if code imports are in allowlist."""

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


def _make_output_limit_validator(
    limit_bytes: int,
) -> Callable[[Context], ValidationResult]:
    """Create a validator that checks stdout/stderr size limits."""

    def validate(ctx: Context) -> ValidationResult:
        output = ctx.last_output()
        if output is None:
            return ValidationResult(result=True)

        stdout = getattr(output, "stdout", "")
        stderr = getattr(output, "stderr", "")
        total_output = ""
        if isinstance(stdout, str):
            total_output += stdout
        if isinstance(stderr, str):
            total_output += stderr

        size = len(total_output.encode("utf-8"))
        if size > limit_bytes:
            return ValidationResult(
                result=False,
                reason=f"Your code produced {size} bytes of output, exceeding the limit of {limit_bytes} bytes.\n"
                f"Add output limiting (e.g., redirect to /dev/null) or optimize your code.",
            )

        return ValidationResult(result=True)

    return validate


# endregion


def python_tool_requirements(
    output_path: str | None = None,
    allowed_imports: list[str] | None = None,
    output_limit_bytes: int = 50_000,
    check_output_artifacts: bool | None = None,
) -> list[Requirement]:
    """Build requirements for Python code generation via the python tool."""
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

    reqs.append(
        Requirement(
            description=f"Output must not exceed {output_limit_bytes} bytes.",
            validation_fn=_make_output_limit_validator(output_limit_bytes),
            check_only=False,
        )
    )

    return reqs
