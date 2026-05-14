"""Matplotlib-specific requirement validators for plotting code.

Provides plotting validation including headless backend detection, plot saving,
and output artifact validation. Uses AST-based analysis for robust detection
of matplotlib operations, with string matching fallback for edge cases.
"""

import ast
from collections.abc import Callable
from pathlib import Path

from ....core import Context, Requirement, ValidationResult
from ..python_reqs import extract_python_code

# Headless matplotlib backends that don't require a display server
HEADLESS_BACKENDS = {"Agg", "Svg", "Cairo", "PDF", "PS", "WebAgg", "nbAgg"}

# Pyplot plot creation methods
PYPLOT_PLOT_METHODS = {"plot", "bar", "scatter", "hist", "imshow", "figure", "subplot"}


def _strip_comments(code: str) -> str:
    """Remove Python comments from code while preserving strings.

    Splits code by lines and removes comments (text after # that's not in a string).
    Handles both single and double quoted strings.
    """
    lines = code.split("\n")
    result = []
    for line in lines:
        in_string = False
        string_char = None
        for i, char in enumerate(line):
            if char in ('"', "'") and (i == 0 or line[i - 1] != "\\"):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
            elif char == "#" and not in_string:
                result.append(line[:i])
                break
        else:
            result.append(line)
    return "\n".join(result)


def _find_attribute_calls(code: str, method_names: list[str]) -> bool:
    """Check if code calls any of the specified methods using AST.

    Handles import aliases (e.g., `import matplotlib.pyplot as plt`) and
    validates that methods are actually called, not just referenced.

    Args:
        code: Python source code to analyze
        method_names: Method names to look for (e.g., ["show", "savefig"])

    Returns:
        True if any of the methods are called, False otherwise
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False

    class CallFinder(ast.NodeVisitor):
        def __init__(self, method_names: list[str]):
            self.method_names = set(method_names)
            self.found = False

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in self.method_names:
                    self.found = True
            self.generic_visit(node)

    finder = CallFinder(method_names)
    finder.visit(tree)
    return finder.found


def _find_function_calls(code: str, func_names: list[str]) -> bool:
    """Check if code calls any of the specified functions using AST.

    Handles qualified names (e.g., `matplotlib.use()`) and detects actual
    function calls, not just references.

    Args:
        code: Python source code to analyze
        func_names: Function names to look for (e.g., ["matplotlib.use"])

    Returns:
        True if any of the functions are called, False otherwise
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False

    class FunctionCallFinder(ast.NodeVisitor):
        def __init__(self, func_names: list[str]):
            self.func_names = set(func_names)
            self.found = False

        def visit_Call(self, node: ast.Call) -> None:
            func_name = self._get_full_name(node.func)
            if func_name in self.func_names:
                self.found = True
            self.generic_visit(node)

        def _get_full_name(self, node: ast.expr) -> str:
            """Extract full qualified name from an AST node."""
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                value_name = self._get_full_name(node.value)
                if value_name:
                    return f"{value_name}.{node.attr}"
                return node.attr
            return ""

    finder = FunctionCallFinder(func_names)
    finder.visit(tree)
    return finder.found


def _code_contains_strings(code: str, patterns: list[str]) -> bool:
    """Check if code contains any of the given string patterns.

    Args:
        code: Python source code to search
        patterns: List of string patterns to look for

    Returns:
        True if any pattern is found in the code, False otherwise
    """
    clean_code = _strip_comments(code)
    return any(pattern in clean_code for pattern in patterns)


def _uses_pyplot_show(code: str) -> bool:
    """Check if code calls plt.show() or similar show() methods.

    Uses AST analysis to robustly detect show() calls regardless of import
    aliases (e.g., `import matplotlib.pyplot as mpl`). AST approach detects
    actual method calls, avoiding false positives from string literals.
    Falls back to string matching only if code doesn't parse.

    Note: May have false positives if code calls .show() on non-matplotlib objects.
    """
    if _find_attribute_calls(code, ["show"]):
        return True
    try:
        ast.parse(code)
    except (SyntaxError, ValueError):
        return _code_contains_strings(code, ["plt.show", ".show()"])
    return False


def _sets_headless_backend(code: str) -> bool:
    """Check if code sets matplotlib to use a headless backend.

    Uses AST analysis to detect matplotlib.use() calls with headless backends.
    Handles various matplotlib import styles and fallback to string matching.
    """
    if not _find_function_calls(code, ["matplotlib.use"]):
        return False

    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        backend_patterns = [f"matplotlib.use('{b}')" for b in HEADLESS_BACKENDS] + [
            f'matplotlib.use("{b}")' for b in HEADLESS_BACKENDS
        ]
        return _code_contains_strings(code, backend_patterns)

    class BackendFinder(ast.NodeVisitor):
        def __init__(self):
            self.has_headless = False

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "use":
                    if isinstance(node.func.value, ast.Name):
                        if node.func.value.id == "matplotlib":
                            if node.args and isinstance(node.args[0], ast.Constant):
                                if node.args[0].value in HEADLESS_BACKENDS:
                                    self.has_headless = True
            self.generic_visit(node)

    finder = BackendFinder()
    finder.visit(tree)
    return finder.has_headless


def _uses_pyplot_plot(code: str) -> bool:
    """Check if code calls pyplot plotting functions.

    Uses AST analysis to detect plot-related method calls. Handles import
    aliases and detects actual method calls, avoiding false positives from
    string literals or method references. Falls back to string matching
    only if code doesn't parse.

    Note: May have false positives if code calls these methods on non-pyplot objects.
    For accuracy, combine with matplotlib import checks.
    """
    if _find_attribute_calls(code, list(PYPLOT_PLOT_METHODS)):
        return True
    try:
        ast.parse(code)
    except (SyntaxError, ValueError):
        return _code_contains_strings(
            code,
            [f".{m}(" for m in PYPLOT_PLOT_METHODS]
            + [f"plt.{m}" for m in PYPLOT_PLOT_METHODS],
        )
    return False


def _calls_savefig(code: str) -> bool:
    """Check if code calls plt.savefig() or fig.savefig().

    Uses AST analysis to robustly detect savefig() calls regardless of
    how matplotlib was imported. Detects actual method calls, avoiding
    false positives from string literals. Falls back to string matching
    only if code doesn't parse.

    Note: May have false positives if code calls savefig() on non-matplotlib objects.
    """
    if _find_attribute_calls(code, ["savefig"]):
        return True
    try:
        ast.parse(code)
    except (SyntaxError, ValueError):
        return _code_contains_strings(code, ["savefig"])
    return False


def _make_matplotlib_headless_validator(
    output_path: str | None = None,
    show_patterns: list[str] | None = None,
    backend_patterns: list[str] | None = None,
) -> Callable[[Context], ValidationResult]:
    """Create a validator that checks matplotlib uses headless backend.

    This validator checks if code calls plt.show() without setting a headless
    backend. It differs from _sets_headless_backend in that it combines both
    checks into a single validation that flags the specific error condition.

    Args:
        output_path: Path where plots should be saved
        show_patterns: Patterns indicating plt.show() calls; defaults to ["plt.show", ".show()"]
        backend_patterns: Patterns indicating headless backend setup; defaults to all
            matplotlib.use() calls with HEADLESS_BACKENDS
    """
    if show_patterns is None:
        show_patterns = ["plt.show", ".show()"]
    if backend_patterns is None:
        backend_patterns = [f"matplotlib.use('{b}')" for b in HEADLESS_BACKENDS] + [
            f'matplotlib.use("{b}")' for b in HEADLESS_BACKENDS
        ]

    def validate(ctx: Context) -> ValidationResult:
        extraction_result = extract_python_code(ctx)
        if not extraction_result.as_bool() or extraction_result.reason is None:
            return ValidationResult(result=True)

        code = extraction_result.reason
        has_show = _code_contains_strings(code, show_patterns)
        has_backend = _code_contains_strings(code, backend_patterns)

        if has_show and not has_backend:
            savefig_instruction = (
                f"plt.savefig('{output_path}'); plt.close()"
                if output_path
                else "plt.savefig('<output_path>'); plt.close()"
            )
            return ValidationResult(
                result=False,
                reason=f"Your code calls `plt.show()` but doesn't set a headless backend.\n"
                f"This will fail in a headless environment (no display).\n\n"
                f"Fix this by adding to the top of your code:\n"
                f"  import matplotlib\n"
                f"  matplotlib.use('Agg')\n\n"
                f"Then replace `plt.show()` with `{savefig_instruction}`",
            )

        return ValidationResult(result=True)

    return validate


def _make_plots_saved_validator(
    output_path: str | None = None,
    plot_patterns: list[str] | None = None,
    save_patterns: list[str] | None = None,
) -> Callable[[Context], ValidationResult]:
    """Create a validator that checks if code saves plots to a file.

    Args:
        output_path: Path where plots should be saved
        plot_patterns: Patterns indicating plot creation; defaults to all PYPLOT_PLOT_METHODS
            prefixed with "plt." and "."
        save_patterns: Patterns indicating plot saving; defaults to ["savefig"]
    """
    if plot_patterns is None:
        plot_patterns = [f"plt.{m}" for m in PYPLOT_PLOT_METHODS] + [
            f".{m}(" for m in PYPLOT_PLOT_METHODS
        ]
    if save_patterns is None:
        save_patterns = ["savefig"]

    def validate(ctx: Context) -> ValidationResult:
        extraction_result = extract_python_code(ctx)
        if not extraction_result.as_bool() or extraction_result.reason is None:
            return ValidationResult(result=True)

        code = extraction_result.reason
        has_plot = _code_contains_strings(code, plot_patterns)
        has_save = _code_contains_strings(code, save_patterns)

        if has_plot and not has_save:
            savefig_instruction = (
                f"plt.savefig('{output_path}')\n  plt.close()"
                if output_path
                else "plt.savefig('<output_path>')\n  plt.close()"
            )
            return ValidationResult(
                result=False,
                reason=f"Your code creates plots with pyplot but never calls `plt.savefig()` to save them.\n\n"
                f"Add this before your plotting code or at the end:\n"
                f"  {savefig_instruction}",
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


def python_plotting_requirements(
    output_path: str | None = None,
    *,
    check_output_artifacts: bool | None = None,
    show_patterns: list[str] | None = None,
    backend_patterns: list[str] | None = None,
    plot_patterns: list[str] | None = None,
    save_patterns: list[str] | None = None,
) -> list[Requirement]:
    """Build plotting-specific requirements for Python tool responses.

    Args:
        output_path: Path where plots should be saved
        check_output_artifacts: Whether to verify the output file exists; defaults to False
        show_patterns: Patterns indicating plt.show() calls; defaults to ["plt.show", ".show()"]
        backend_patterns: Patterns indicating headless backend setup; defaults to all
            matplotlib.use() calls with HEADLESS_BACKENDS
        plot_patterns: Patterns indicating plot creation; defaults to all PYPLOT_PLOT_METHODS
            prefixed with "plt." and "."
        save_patterns: Patterns indicating plot saving; defaults to ["savefig"]

    Returns:
        List of Requirement objects that validate matplotlib usage and plot output.
    """
    reqs: list[Requirement] = []

    reqs.append(
        Requirement(
            description="If using pyplot, must set headless backend and use savefig.",
            validation_fn=_make_matplotlib_headless_validator(
                output_path,
                show_patterns=show_patterns,
                backend_patterns=backend_patterns,
            ),
            check_only=False,
        )
    )

    reqs.append(
        Requirement(
            description="If creating plots, must call savefig to save them.",
            validation_fn=_make_plots_saved_validator(
                output_path, plot_patterns=plot_patterns, save_patterns=save_patterns
            ),
            check_only=False,
        )
    )

    if check_output_artifacts and output_path:
        reqs.append(
            Requirement(
                description=f"Output file must be created at {output_path}",
                validation_fn=_make_output_artifacts_validator(output_path),
                check_only=False,
            )
        )

    return reqs
