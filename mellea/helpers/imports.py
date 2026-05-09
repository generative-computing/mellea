"""Utilities for analyzing Python code imports."""

import ast


def get_unauthorized_imports(
    code: str, allowed_imports: list[str] | None = None
) -> list[str]:
    r"""Extract unauthorized imports from Python code.

    Parses Python code and returns a sorted list of top-level modules that are
    imported but not in the allowed list. Handles both `import X` and `from X import Y`
    statements, extracting the root module name (e.g., "numpy" from "numpy.random").

    Args:
        code: Python source code to analyze.
        allowed_imports: Allowlist of permitted top-level modules. If None, allows all imports.

    Returns:
        Sorted list of unauthorized module names found in code. Empty list if code
        has syntax errors or if allowed_imports is None.

    Example:
        >>> code = "import numpy\nimport os\nimport forbidden_lib"
        >>> get_unauthorized_imports(code, ["os", "sys"])
        ['numpy', 'forbidden_lib']
    """
    if allowed_imports is None:
        return []

    unauthorized: set[str] = set()
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split(".")[0]
                if module not in allowed_imports:
                    unauthorized.add(module)
        elif isinstance(node, ast.ImportFrom) and node.module:
            module = node.module.split(".")[0]
            if module not in allowed_imports:
                unauthorized.add(module)

    return sorted(unauthorized)
