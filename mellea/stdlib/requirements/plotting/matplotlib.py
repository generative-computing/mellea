"""Matplotlib-specific code generation requirements.

This module validates Python code that uses matplotlib for plotting, ensuring
proper headless backend configuration, file I/O, and dependency availability.
"""

import ast

from mellea.core import Context, Requirement, ValidationResult
from mellea.stdlib.requirements.python_reqs import _has_python_code_listing

# Matplotlib backends suitable for headless (non-interactive) execution
HEADLESS_BACKENDS = {
    "Agg",
    "Cairo",
    "pdf",
    "svg",
    "pgf",
    "nbAgg",
    "module://gr.matplotlib.backend_gr",
}


def _extract_code(ctx: Context) -> str | None:
    """Extract the primary code block from context.

    Args:
        ctx: Context containing model output with code blocks.

    Returns:
        Extracted code string, or None if extraction failed.
    """
    extraction_result = _has_python_code_listing(ctx)
    if not extraction_result.as_bool():
        return None
    return extraction_result.reason


def _is_matplotlib_use_call(node: ast.Call) -> bool:
    """Check if an AST Call node is a matplotlib.use() call.

    Args:
        node: AST Call node to check.

    Returns:
        True if node matches matplotlib.use(...) pattern.
    """
    if isinstance(node.func, ast.Attribute):
        if node.func.attr == "use":
            if isinstance(node.func.value, ast.Name):
                return node.func.value.id == "matplotlib"
    return False


def _extract_backend_name(node: ast.Call) -> str | None:
    """Extract the backend name from a matplotlib.use() call.

    Args:
        node: AST Call node representing matplotlib.use(...).

    Returns:
        Backend name as string, or None if not found or not a literal string.
    """
    if not node.args:
        return None

    first_arg = node.args[0]
    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
        return first_arg.value
    return None


def _find_matplotlib_use_backend(code: str) -> str | None:
    """Find the backend name in matplotlib.use() call within code.

    Args:
        code: Python code to analyze.

    Returns:
        Backend name if found, None otherwise.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_matplotlib_use_call(node):
            backend = _extract_backend_name(node)
            if backend:
                return backend
    return None


def _is_savefig_call(node: ast.Call) -> bool:
    """Check if an AST Call node is a savefig() call.

    Detects:
    - plt.savefig(...)
    - fig.savefig(...)
    - ax.savefig(...)
    - Any expression.savefig(...)

    Args:
        node: AST Call node to check.

    Returns:
        True if node is a savefig() call, False otherwise.
    """
    if isinstance(node.func, ast.Attribute):
        return node.func.attr == "savefig"
    return False


def _output_path_in_savefig_args(node: ast.Call, output_path: str) -> bool:
    """Check if output_path appears in savefig() arguments.

    Args:
        node: AST Call node representing a savefig() call.
        output_path: Output path to search for.

    Returns:
        True if output_path found in positional or keyword arguments.
    """
    # Check positional arguments
    for arg in node.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            if arg.value == output_path:
                return True

    # Check keyword arguments (e.g., fname="path", filename="path")
    for keyword in node.keywords:
        if isinstance(keyword.value, ast.Constant) and isinstance(
            keyword.value.value, str
        ):
            if keyword.value.value == output_path:
                return True
    return False


def _find_savefig_with_path(code: str, output_path: str) -> bool:
    """Find savefig() call with specific output path in code.

    Args:
        code: Python code to analyze.
        output_path: Path to search for in savefig() calls.

    Returns:
        True if savefig() with output_path found, False otherwise.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_savefig_call(node):
            if _output_path_in_savefig_args(node, output_path):
                return True
    return False


class MatplotlibHeadlessBackend(Requirement):
    """Validates that matplotlib is configured with a headless backend.

    Matplotlib must be explicitly configured with a headless backend (e.g., 'Agg')
    via matplotlib.use() before importing pyplot. Interactive backends like 'TkAgg'
    will fail because they require a display server.

    Raises:
        None — validation failure returns ValidationResult with False result.
    """

    def __init__(self) -> None:
        """Initialize MatplotlibHeadlessBackend requirement."""
        super().__init__(
            description="Matplotlib is configured with a headless backend (Agg, Cairo, pdf, svg, pgf, etc.).",
            validation_fn=self._validate_headless_backend,
            check_only=True,
        )

    def _validate_headless_backend(self, ctx: Context) -> ValidationResult:
        """Validate that matplotlib uses a headless backend.

        Args:
            ctx: Context containing model output with code blocks.

        Returns:
            ValidationResult with pass/fail and backend name or error details.
        """
        code = _extract_code(ctx)
        if code is None:
            return ValidationResult(
                result=False,
                reason="Could not extract code for matplotlib backend validation.",
            )

        backend = _find_matplotlib_use_backend(code)
        if backend is None:
            return ValidationResult(
                result=False,
                reason="No matplotlib.use() call found. Add matplotlib.use('Agg') before importing pyplot.",
            )

        if backend in HEADLESS_BACKENDS:
            return ValidationResult(
                result=True, reason=f"Headless backend configured: {backend}"
            )

        return ValidationResult(
            result=False,
            reason=f"Backend '{backend}' is not headless. Use one of: {', '.join(sorted(HEADLESS_BACKENDS))}",
        )


class PlotFileSaved(Requirement):
    """Validates that a plot is explicitly saved to a file.

    The plot must be saved using savefig() with the specified output_path.
    This prevents interactive plot displays (plt.show()) and ensures output
    can be captured and verified.

    Args:
        output_path: File path where the plot should be saved (e.g., '/tmp/plot.png').
    """

    def __init__(self, output_path: str) -> None:
        """Initialize PlotFileSaved requirement.

        Args:
            output_path: Expected file path for plot output.
        """
        self.output_path = output_path
        super().__init__(
            description=f"Plot is explicitly saved to file using savefig('{output_path}').",
            validation_fn=self._validate_plot_saved,
            check_only=True,
        )

    def _validate_plot_saved(self, ctx: Context) -> ValidationResult:
        """Validate that plot is saved to the expected output path.

        Args:
            ctx: Context containing model output with code blocks.

        Returns:
            ValidationResult with pass/fail and file path details or error message.
        """
        code = _extract_code(ctx)
        if code is None:
            return ValidationResult(
                result=False, reason="Could not extract code for plot file validation."
            )

        if _find_savefig_with_path(code, self.output_path):
            return ValidationResult(
                result=True, reason=f"Plot saved to {self.output_path}"
            )

        return ValidationResult(
            result=False,
            reason=f"No savefig() call found with path '{self.output_path}'. "
            f"Add: plt.savefig('{self.output_path}') or fig.savefig('{self.output_path}')",
        )


class PlotDependenciesAvailable(Requirement):
    """Validates that matplotlib and numpy are importable.

    Both matplotlib and numpy must be available in the execution environment.
    This requirement checks import availability but does not execute code.

    Raises:
        ImportError: If matplotlib or numpy cannot be imported.
    """

    def __init__(self) -> None:
        """Initialize PlotDependenciesAvailable requirement."""
        super().__init__(
            description="matplotlib and numpy are importable.",
            validation_fn=self._validate_dependencies,
            check_only=True,
        )

    def _validate_dependencies(self, ctx: Context) -> ValidationResult:
        """Validate that required plotting dependencies are available.

        Args:
            ctx: Context containing model output.

        Returns:
            ValidationResult with pass/fail and dependency details.

        Raises:
            ImportError: If matplotlib or numpy cannot be imported.
        """
        code = _extract_code(ctx)
        if code is None:
            return ValidationResult(
                result=False, reason="Could not extract code for dependency validation."
            )

        # Check if matplotlib and numpy can be imported
        for module_name in ["matplotlib", "numpy"]:
            try:
                __import__(module_name)
            except ImportError as e:
                raise ImportError(
                    f"Missing dependency: {module_name}. "
                    f"Install with: pip install {module_name}"
                ) from e

        return ValidationResult(
            result=True, reason="All dependencies available (matplotlib, numpy)."
        )
