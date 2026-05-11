"""Import analysis helpers for Python requirements and execution environments."""

import ast


def get_unauthorized_imports(
    code: str, allowed_imports: list[str] | None = None
) -> list[str]:
    """Extract unauthorized top-level imports from Python code."""
    if allowed_imports is None:
        return []

    unauthorized: set[str] = set()
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        # Syntax errors are validated separately by dedicated validators.
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


# Made with Bob
