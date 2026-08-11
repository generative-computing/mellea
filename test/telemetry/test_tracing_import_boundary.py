# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enforce the "library code fires hooks; plugins open spans" import boundary.

See #1464: `mellea/backends/` (and the rest of `mellea/stdlib/`) must never
import `mellea.telemetry.tracing` directly, because a direct import is a
direct span-opening call — the thing that has to happen from a
`tracing_plugins.py` plugin instead so telemetry stays optional and
removable. `mellea/stdlib/session.py` is the one sanctioned exception,
documented at its import site: OTel `Token` attach/detach is task-affine and
can't be delegated to a plugin.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SANCTIONED_EXCEPTION = "mellea/stdlib/session.py"


def _package_for(path: Path) -> str:
    """Return the dotted package containing `path`, for relative-import resolution."""
    rel = path.relative_to(REPO_ROOT)
    parts = list(rel.parts[:-1])
    if rel.stem != "__init__":
        pass  # module's package is its containing directory either way
    return ".".join(parts)


def _resolve_relative_import(package: str, level: int, module: str | None) -> str:
    """Mirror importlib's `_resolve_name` for `from . import x`-style imports."""
    bits = package.rsplit(".", level - 1)
    base = bits[0]
    return f"{base}.{module}" if module else base


def _imports_telemetry_tracing(path: Path) -> bool:
    """Return True if `path` imports the `mellea.telemetry.tracing` module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = _package_for(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "mellea.telemetry.tracing" for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                resolved = node.module or ""
                if resolved == "mellea.telemetry.tracing":
                    return True
            else:
                resolved = _resolve_relative_import(package, node.level, node.module)
                if resolved == "mellea.telemetry.tracing":
                    return True
                # `from ..telemetry import tracing` imports the submodule by name.
                if resolved == "mellea.telemetry" and any(
                    alias.name == "tracing" for alias in node.names
                ):
                    return True
    return False


def _source_files(*subdirs: str) -> list[Path]:
    files: list[Path] = []
    for subdir in subdirs:
        files.extend((REPO_ROOT / subdir).rglob("*.py"))
    return files


def test_backends_never_import_telemetry_tracing():
    """`mellea/backends/` must fire hooks, not open spans directly."""
    offenders = [
        str(f.relative_to(REPO_ROOT))
        for f in _source_files("mellea/backends")
        if _imports_telemetry_tracing(f)
    ]
    assert offenders == [], (
        "These backend modules import mellea.telemetry.tracing directly, "
        "which means they open spans instead of firing a hook for a "
        "tracing_plugins.py plugin to open one. Fire the matching hook "
        "instead (see mellea/telemetry/tracing_plugins.py for the "
        f"pattern): {offenders}"
    )


def test_stdlib_tracing_import_is_only_the_sanctioned_exception():
    """`mellea/stdlib/session.py` is the one place allowed to import tracing directly."""
    offenders = [
        str(f.relative_to(REPO_ROOT))
        for f in _source_files("mellea/stdlib", "mellea/backends")
        if _imports_telemetry_tracing(f)
    ]
    assert offenders == [SANCTIONED_EXCEPTION], (
        "Exactly one module (mellea/stdlib/session.py) is allowed to import "
        "mellea.telemetry.tracing directly, because OTel Token attach/detach "
        "there is task-affine and can't be delegated to a hook-driven "
        f"plugin. Found: {offenders}. If this is a new legitimate exception, "
        "document it at the import site the way session.py does, and update "
        "this test's SANCTIONED_EXCEPTION."
    )
