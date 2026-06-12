"""Matplotlib-specific code generation requirements.

This module validates Python code that uses matplotlib for plotting, ensuring
proper headless backend configuration, file I/O, and dependency availability.
"""

import ast

from mellea.core import Context, Requirement, ValidationResult
from mellea.stdlib.requirements.python_reqs import _has_python_code_listing
from mellea.stdlib.requirements.python_tools import python_code_generation_requirements

# Matplotlib backends suitable for headless (non-interactive) execution.
# Includes standard raster (Agg, Cairo), vector (pdf, svg, pgf),
# and GR backend (module://gr.matplotlib.backend_gr) which is a high-performance
# graphics library that works in headless environments.
HEADLESS_BACKENDS = {
    "Agg",
    "Cairo",
    "pdf",
    "svg",
    "pgf",
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
    if node.args:
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            return first_arg.value
        return None

    for keyword in node.keywords:
        if (
            keyword.arg == "backend"
            and isinstance(keyword.value, ast.Constant)
            and isinstance(keyword.value.value, str)
        ):
            return keyword.value.value
    return None


def _matplotlib_use_call_exists(code: str) -> bool:
    """Check if matplotlib.use() is called in the code.

    Args:
        code: Python code to analyze.

    Returns:
        True if any matplotlib.use() call exists, False otherwise.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_matplotlib_use_call(node):
            return True
    return False


def _find_matplotlib_use_backend(code: str) -> str | None:
    """Find the backend name in matplotlib.use() call within code.

    Args:
        code: Python code to analyze.

    Returns:
        Backend name if found as a literal string, None otherwise.
        Returns None both when matplotlib.use() is absent and when it's called
        with a non-literal argument (e.g., matplotlib.use(BACKEND_VAR)).
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
        if keyword.arg in ("fname", "filename") and isinstance(
            keyword.value, ast.Constant
        ):
            if (
                isinstance(keyword.value.value, str)
                and keyword.value.value == output_path
            ):
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
            if _matplotlib_use_call_exists(code):
                return ValidationResult(
                    result=False,
                    reason="matplotlib.use() called with a non-literal argument; backend cannot be statically verified. Use a literal string like matplotlib.use('Agg').",
                )
            return ValidationResult(
                result=False,
                reason="No matplotlib.use() call found. Add matplotlib.use('Agg') before importing pyplot.",
            )

        canonical = {b.lower(): b for b in HEADLESS_BACKENDS}
        if backend.lower() in canonical:
            return ValidationResult(
                result=True,
                reason=f"Headless backend configured: {canonical[backend.lower()]}",
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
        """Initialize PlotFileSaved requirement."""
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
            ctx: Context (unused, requirement checks environment not code).

        Returns:
            ValidationResult with pass/fail and dependency details.
        """
        for module_name in ["matplotlib", "numpy"]:
            try:
                __import__(module_name)
            except ImportError:
                return ValidationResult(
                    result=False,
                    reason=f"Missing dependency: {module_name}. "
                    f"Install with: pip install {module_name}",
                )

        return ValidationResult(
            result=True, reason="All dependencies available (matplotlib, numpy)."
        )


def python_plotting_requirements(
    output_path: str,
    allowed_imports: list[str] | None = None,
    output_limit_chars: int = 10_000,
    timeout_seconds: int = 5,
    use_sandbox: bool = False,
) -> list[Requirement]:
    """Bundle matplotlib-specific requirements for plotting code validation.

    Factory function that creates a complete set of requirements for validating
    matplotlib plotting code, composing general Python code generation requirements
    with plotting-specific constraints for headless backend configuration, file
    output, and dependency availability.

    Args:
        output_path: File path where the plot should be saved (e.g., '/tmp/plot.png').
            This path must match the savefig() call in the generated code.
        allowed_imports: Whitelist of importable top-level modules. None allows all.
            Default None.
        output_limit_chars: Maximum allowed characters of captured stdout.
            Default 10,000.
        timeout_seconds: Maximum execution time in seconds. Default 5.
        use_sandbox: Use llm-sandbox for Docker-isolated execution. Default False.

    Returns:
        list[Requirement]: Seven requirements in validation order:
            1-4. PythonCodeExtraction, PythonSyntaxValid, PythonExecutionReq,
                 ImportRestrictions/NoImportRestrictions (from python_code_generation_requirements)
            5. MatplotlibHeadlessBackend — validates headless backend configuration
            6. PlotFileSaved — validates plot is saved to the specified output_path
            7. PlotDependenciesAvailable — validates matplotlib and numpy are available

    Raises:
        TypeError: If output_path is not a string.
        ValueError: If output_path is empty.

    Examples:
        >>> output_path = "/tmp/plot.png"
        >>> reqs = python_plotting_requirements(output_path=output_path)
        >>> len(reqs)
        7
        >>> isinstance(reqs[4], MatplotlibHeadlessBackend)
        True
        >>> isinstance(reqs[5], PlotFileSaved)
        True
        >>> isinstance(reqs[6], PlotDependenciesAvailable)
        True
        >>> reqs_restricted = python_plotting_requirements(
        ...     output_path=output_path,
        ...     allowed_imports=["matplotlib", "numpy"]
        ... )
        >>> len(reqs_restricted)
        7
    """
    if not isinstance(output_path, str):
        raise TypeError(
            f"output_path must be a string, got {type(output_path).__name__}"
        )

    if not output_path.strip():
        raise ValueError("output_path cannot be empty")

    requirements = python_code_generation_requirements(
        allowed_imports=allowed_imports,
        output_limit_chars=output_limit_chars,
        timeout_seconds=timeout_seconds,
        use_sandbox=use_sandbox,
    )
    requirements.extend(
        [
            MatplotlibHeadlessBackend(),
            PlotFileSaved(output_path=output_path),
            PlotDependenciesAvailable(),
        ]
    )
    return requirements
