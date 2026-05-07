"""Pre-composed requirement bundles for Python tool invocation and execution.

This module provides bundled requirements for validating Python code generated
via the Python tool, with focus on reactive failure detection and repair:

- Tool invocation validation (must call Python tool with code argument)
- Syntax validation (code must parse correctly)
- Import validation (code imports must be in allowlist)
- Matplotlib headless backend detection (plt.show() without backend)
- Plot artifact validation (savefig must be called, output files must exist)
- Output limiting (stdout/stderr must not exceed configured limits)

Failure messages are written as feedback to the model, not to developers.
They state the failure, include relevant code/stderr, and explain the
correction well enough for the model to act on it.

FAILURE MATRIX — How each requirement catches the canonical plotting failures:

Scenario: Model generates plotting code with matplotlib

Attempt 1: No tool call
  → MustInvokePythonTool fails
  → Repair: "Call the `python` tool with your code"

Attempt 2: Tool called but no 'code' arg
  → PythonToolHasCodeArg fails
  → Repair: "The python tool requires a 'code' argument"

Attempt 3: Code has syntax error
  → PythonCodeParses fails
  → Repair: "Your code has a syntax error at line X: {error}"

Attempt 4: Code imports matplotlib (not in allowed_imports)
  → PythonImportsAllowed fails
  → Repair: "matplotlib is not allowed. Use only: {allowed_list}"

Attempt 5: Code uses plt.show() without headless backend
  → MatplotlibHeadless fails
  → Repair: "Add matplotlib.use('Agg') and replace plt.show() with plt.savefig(...)"

Attempt 6: Code has plt.plot() but no plt.savefig()
  → PlotsAreSaved fails
  → Repair: "Add plt.savefig('{output_path}') to save the plot"

Attempt 7: Code runs, but output file not created
  → OutputArtifactsExist fails
  → Repair: "File '{output_path}' was not created. Check plt.savefig() call"

Attempt 8: Success
  → All requirements pass
  → Result: plot file exists and is non-empty
"""

import ast
from collections.abc import Callable
from pathlib import Path

from ...core import Context, Requirement, ValidationResult
from .python_reqs import _has_python_code_listing


def _extract_code(ctx: Context) -> str | None:
    """Extract Python code from either tool calls or markdown blocks.

    Checks tool_calls dict first (for tool calling), then falls back to
    markdown code blocks in response text.

    Returns the code string, or None if no code found.
    """
    # Try tool_calls first (tool calling format)
    output = ctx.last_output()
    if output and output.tool_calls and "python" in output.tool_calls:
        tool_call = output.tool_calls["python"]
        if hasattr(tool_call, "args") and "code" in tool_call.args:
            return tool_call.args["code"]

    # Fall back to markdown code blocks in response text
    result = _has_python_code_listing(ctx)
    if result.as_bool() and result.reason:
        return result.reason
    return None


def _get_unauthorized_imports(
    code: str, allowed_imports: list[str] | None
) -> list[str]:
    """Return list of imports in code that are not in allowed_imports.

    Args:
        code: Python code to analyze
        allowed_imports: Allowlist of permitted top-level modules (None = allow all)

    Returns:
        List of unauthorized import module names, or empty list if all allowed.
    """
    if allowed_imports is None:
        return []

    unauthorized = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split(".")[0]
                    if module_name not in allowed_imports:
                        unauthorized.append(module_name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split(".")[0]
                    if module_name not in allowed_imports:
                        unauthorized.append(module_name)
    except (SyntaxError, ValueError):
        pass

    return list(set(unauthorized))


def _code_parses(code: str) -> tuple[bool, str | None]:
    """Check if code parses as valid Python.

    Returns:
        (True, None) if code parses
        (False, error_message) if syntax error
    """
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
        if e.text:
            error_msg += f"\n  {e.text.rstrip()}"
            if e.offset:
                error_msg += "\n  " + " " * (e.offset - 1) + "^"
        return False, error_msg


def _uses_pyplot_show(code: str) -> bool:
    """Check if code calls plt.show() or matplotlib.pyplot.show()."""
    # Simple string checks work for most cases
    return "plt.show" in code or ".show()" in code


def _sets_headless_backend(code: str) -> bool:
    """Check if code sets matplotlib to use a headless backend."""
    headless_backends = ("Agg", "Svg", "Cairo", "PDF", "PS", "WebAgg", "nbAgg")
    for backend in headless_backends:
        if (
            f"matplotlib.use('{backend}')" in code
            or f'matplotlib.use("{backend}")' in code
        ):
            return True
    return False


def _uses_pyplot_plot(code: str) -> bool:
    """Check if code calls pyplot plotting functions."""
    plot_functions = (
        "plt.plot",
        "plt.bar",
        "plt.scatter",
        "plt.hist",
        "plt.imshow",
        "plt.figure",
        "plt.subplot",
        ".plot(",
        ".bar(",
        ".scatter(",
        ".hist(",
    )
    return any(func in code for func in plot_functions)


def _calls_savefig(code: str) -> bool:
    """Check if code calls plt.savefig() or fig.savefig()."""
    return "savefig" in code


# region Individual Requirement Validators


def _validate_python_tool_invoked(ctx: Context) -> ValidationResult:
    """Requirement: Model must invoke the Python tool."""
    output = ctx.last_output()
    if output is None or output.tool_calls is None:
        return ValidationResult(
            result=False,
            reason=(
                "You did not invoke any tools. To execute Python code, "
                "call the `python` tool with your code."
            ),
        )
    if "python" not in output.tool_calls:
        return ValidationResult(
            result=False,
            reason=(
                "You did not call the `python` tool. Call it with your "
                "code to execute it."
            ),
        )
    return ValidationResult(result=True)


def _validate_python_tool_has_code_arg(ctx: Context) -> ValidationResult:
    """Requirement: Python tool call must include a 'code' argument."""
    output = ctx.last_output()
    if output is None or output.tool_calls is None:
        return ValidationResult(result=False, reason="No tool calls found")

    if "python" not in output.tool_calls:
        return ValidationResult(result=False, reason="Python tool not called")

    python_call = output.tool_calls["python"]
    if "code" not in python_call.args:
        return ValidationResult(
            result=False,
            reason="The `python` tool call must include a `code` argument with your Python code.",
        )

    return ValidationResult(result=True)


def _make_code_parses_validator() -> Callable[[Context], ValidationResult]:
    """Create a validator that checks if extracted code parses."""

    def validate(ctx: Context) -> ValidationResult:
        code = _extract_code(ctx)
        if not code:
            return ValidationResult(
                result=False,
                reason=(
                    "Could not extract Python code from your response. "
                    "Make sure to include code in ```python ... ``` blocks."
                ),
            )

        parses, error = _code_parses(code)
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

        code = _extract_code(ctx)
        if not code:
            return ValidationResult(
                result=False, reason="Could not extract Python code"
            )

        unauthorized = _get_unauthorized_imports(code, allowed_imports)
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


def _make_matplotlib_headless_validator() -> Callable[[Context], ValidationResult]:
    """Create a validator that checks matplotlib uses headless backend."""

    def validate(ctx: Context) -> ValidationResult:
        code = _extract_code(ctx)
        if not code:
            return ValidationResult(result=True)

        if _uses_pyplot_show(code) and not _sets_headless_backend(code):
            return ValidationResult(
                result=False,
                reason="Your code calls `plt.show()` but doesn't set a headless backend.\n"
                "This will fail in a headless environment (no display).\n\n"
                "Fix this by adding to the top of your code:\n"
                "  import matplotlib\n"
                "  matplotlib.use('Agg')\n\n"
                "Then replace `plt.show()` with `plt.savefig('{output_path}'); plt.close()`",
            )

        return ValidationResult(result=True)

    return validate


def _make_plots_saved_validator() -> Callable[[Context], ValidationResult]:
    """Create a validator that checks if code saves plots to a file."""

    def validate(ctx: Context) -> ValidationResult:
        code = _extract_code(ctx)
        if not code:
            return ValidationResult(result=True)

        if _uses_pyplot_plot(code) and not _calls_savefig(code):
            return ValidationResult(
                result=False,
                reason="Your code creates plots with pyplot but never calls `plt.savefig()` to save them.\n\n"
                "Add this before your plotting code or at the end:\n"
                "  plt.savefig('{output_path}')\n"
                "  plt.close()",
            )

        return ValidationResult(result=True)

    return validate


def _make_output_artifacts_validator(
    output_path: str,
) -> Callable[[Context], ValidationResult]:
    """Create a validator that checks if output file exists post-execution."""

    def validate(ctx: Context) -> ValidationResult:
        path = Path(output_path)

        if not path.exists():
            return ValidationResult(
                result=False,
                reason=f"The output file '{output_path}' was not created during execution.\n"
                f"Make sure your code calls `plt.savefig('{output_path}')` to save the plot.",
            )

        if path.stat().st_size == 0:
            return ValidationResult(
                result=False,
                reason=f"The output file '{output_path}' exists but is empty.\n"
                f"Check that your plot code executed correctly.",
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

        total_output = ""
        if hasattr(output, "stdout") and output.stdout:
            total_output += output.stdout
        if hasattr(output, "stderr") and output.stderr:
            total_output += output.stderr

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


class PythonToolRequirements:
    """Pre-composed bundle of requirements for Python code generation via the tool.

    This bundle validates the complete Python code generation flow: tool invocation,
    syntax, imports, execution, and output. It's designed to work with repair loops
    (SOFAI, MultiTurnStrategy) to iteratively fix common plotting failures.

    Markers:
    - **Deterministic** (unit-testable): tool invocation, syntax, imports, headless backend,
      savefig presence, file existence, output limits
    - **Qualitative** (needs model to evaluate): execution without error (captured via stderr)

    Args:
        output_path (str | None): Path where plots should be saved. If specified, enables
            output artifact validation. Defaults to None.
        allowed_imports (list[str] | None): Allowlist of importable top-level modules.
            None (default) allows any import. Set to list like ["numpy", "matplotlib"]
            to restrict imports.
        output_limit_bytes (int): Maximum bytes of stdout/stderr allowed. Defaults to 50000.
        check_output_artifacts (bool): If True, validate that output file exists and is
            non-empty after execution. Defaults to True if output_path is specified.

    Attributes:
        requirements (list[Requirement]): The composed list of requirements, suitable
            for use with sampling strategies.
    """

    def __init__(
        self,
        output_path: str | None = None,
        allowed_imports: list[str] | None = None,
        output_limit_bytes: int = 50_000,
        check_output_artifacts: bool | None = None,
    ):
        """Initialize the Python tool requirements bundle."""
        self.output_path = output_path
        self.allowed_imports = allowed_imports
        self.output_limit_bytes = output_limit_bytes

        # Auto-enable output artifact checking if output_path is specified
        if check_output_artifacts is None:
            check_output_artifacts = output_path is not None

        self._check_output_artifacts = check_output_artifacts

        self.requirements = self._build_requirements()

    def _build_requirements(self) -> list[Requirement]:
        """Build the list of requirements for this bundle."""
        reqs: list[Requirement] = []

        # Tool invocation requirements (deterministic)
        reqs.append(
            Requirement(
                description="Use the python tool to execute code.",
                validation_fn=_validate_python_tool_invoked,
                check_only=False,
            )
        )

        reqs.append(
            Requirement(
                description="The python tool call must include a code argument.",
                validation_fn=_validate_python_tool_has_code_arg,
                check_only=False,
            )
        )

        # Code quality requirements (deterministic)
        reqs.append(
            Requirement(
                description="The Python code must parse correctly.",
                validation_fn=_make_code_parses_validator(),
                check_only=False,
            )
        )

        # Import validation (deterministic)
        if self.allowed_imports is not None:
            reqs.append(
                Requirement(
                    description=f"Imports must be from allowed list: {', '.join(self.allowed_imports)}",
                    validation_fn=_make_imports_allowed_validator(self.allowed_imports),
                    check_only=False,
                )
            )

        # Matplotlib-specific requirements (deterministic)
        reqs.append(
            Requirement(
                description=(
                    "If using pyplot, must set headless backend and use savefig."
                ),
                validation_fn=_make_matplotlib_headless_validator(),
                check_only=False,
            )
        )

        reqs.append(
            Requirement(
                description="If creating plots, must call savefig to save them.",
                validation_fn=_make_plots_saved_validator(),
                check_only=False,
            )
        )

        # Output artifact validation (deterministic, post-execution)
        if self._check_output_artifacts and self.output_path:
            reqs.append(
                Requirement(
                    description=f"Output file must be created at {self.output_path}",
                    validation_fn=_make_output_artifacts_validator(self.output_path),
                    check_only=False,
                )
            )

        # Output limiting (deterministic)
        reqs.append(
            Requirement(
                description=f"Output must not exceed {self.output_limit_bytes} bytes.",
                validation_fn=_make_output_limit_validator(self.output_limit_bytes),
                check_only=False,
            )
        )

        return reqs

    def __repr__(self) -> str:
        """Return a developer-readable representation."""
        return (
            f"PythonToolRequirements("
            f"output_path={self.output_path!r}, "
            f"allowed_imports={self.allowed_imports!r}, "
            f"output_limit_bytes={self.output_limit_bytes}, "
            f"requirements={len(self.requirements)} items"
            f")"
        )
