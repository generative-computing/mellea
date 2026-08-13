# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enforce the "library code fires hooks; plugins open spans" import boundary.

See #1464. `mellea/backends/` and the rest of `mellea/stdlib/` must never
import `mellea.telemetry.tracing` directly — `mellea/stdlib/session.py` is
the one sanctioned exception, documented at its import site.

The same invariant should eventually cover `mellea.telemetry.metrics` too
(tracked as a follow-up: `mellea/stdlib/tools/_bash_audit.py` currently
imports `create_counter` directly and would need a hook + plugin to fix).
`TELEMETRY_SUBMODULES` is a tuple so that follow-up is a one-line change.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TELEMETRY_SUBMODULES = ("tracing",)
SANCTIONED_EXCEPTIONS = {"mellea/stdlib/session.py"}


def _package_for(path: Path) -> str:
    """Return the dotted package containing `path`, for relative-import resolution."""
    rel = path.relative_to(REPO_ROOT)
    return ".".join(rel.parts[:-1])


def _resolve_relative_import(package: str, level: int, module: str | None) -> str:
    """Mirror importlib's `_resolve_name` for `from . import x`-style imports."""
    bits = package.rsplit(".", level - 1)
    base = bits[0]
    return f"{base}.{module}" if module else base


def _imported_telemetry_submodules(path: Path) -> set[str]:
    """Return the `mellea.telemetry.*` submodules `path` imports directly."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = _package_for(path)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for submodule in TELEMETRY_SUBMODULES:
                    if alias.name == f"mellea.telemetry.{submodule}":
                        found.add(submodule)
        elif isinstance(node, ast.ImportFrom):
            resolved = (
                node.module or ""
                if node.level == 0
                else _resolve_relative_import(package, node.level, node.module)
            )
            for submodule in TELEMETRY_SUBMODULES:
                if resolved == f"mellea.telemetry.{submodule}":
                    found.add(submodule)
                # `from ...telemetry import tracing` imports the submodule by name,
                # whether written as an absolute or a relative import.
                elif resolved == "mellea.telemetry" and any(
                    alias.name == submodule for alias in node.names
                ):
                    found.add(submodule)
    return found


def _source_files(*subdirs: str) -> list[Path]:
    files: list[Path] = []
    for subdir in subdirs:
        files.extend((REPO_ROOT / subdir).rglob("*.py"))
    return files


def test_backends_and_stdlib_never_import_telemetry_tracing():
    """Only the sanctioned exceptions may import `mellea.telemetry.tracing`."""
    offenders = sorted(
        str(f.relative_to(REPO_ROOT))
        for f in _source_files("mellea/backends", "mellea/stdlib")
        if _imported_telemetry_submodules(f)
    )
    assert set(offenders) == SANCTIONED_EXCEPTIONS, (
        "Only mellea/stdlib/session.py is allowed to import "
        "mellea.telemetry.tracing directly (OTel Token attach/detach there "
        "is task-affine and can't be delegated to a hook-driven plugin). A "
        "direct import elsewhere means a module is opening a span itself "
        "instead of firing a hook for a tracing_plugins.py plugin to "
        f"handle. Found: {offenders}. If this is a new legitimate exception, "
        "document it at the import site the way session.py does, and add it "
        "to SANCTIONED_EXCEPTIONS."
    )
