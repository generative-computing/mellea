"""Line-based detection and rewriting of old genslot imports and class names.

Targets:
- ``from mellea.stdlib.components.genslot import ...`` → ``genstub``
- ``import mellea.stdlib.components.genslot [as ...]`` → ``genstub``
- ``from mellea.stdlib.components import genslot [as ...]`` → ``genstub``
- ``from .genslot import ...`` (relative imports) → ``genstub``
- ``GenerativeSlot`` → ``GenerativeStub`` (and Sync/Async variants)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Directories to skip during traversal.
SKIP_DIRS = {"__pycache__", ".git", ".venv", "node_modules"}

# Ordered longest-first so ``SyncGenerativeSlot`` is replaced before ``GenerativeSlot``.
_CLASS_RENAMES: list[tuple[str, str]] = [
    ("AsyncGenerativeSlot", "AsyncGenerativeStub"),
    ("SyncGenerativeSlot", "SyncGenerativeStub"),
    ("GenerativeSlot", "GenerativeStub"),
]

# --- Module-path patterns ---

# Fully-qualified module path (handles both `from … import` and `import …`).
_MODULE_OLD = "mellea.stdlib.components.genslot"
_MODULE_NEW = "mellea.stdlib.components.genstub"
_MODULE_RE = re.compile(re.escape(_MODULE_OLD))

# `from mellea.stdlib.components import genslot` (with optional ` as …`).
_FROM_PARENT_RE = re.compile(
    r"(\bfrom\s+mellea\.stdlib\.components\s+import\s+)"  # prefix
    r"(\bgenslot\b)"  # the name to replace
)

# Relative imports: `from .genslot import …` or `from ..components.genslot import …`
# Matches any leading dots followed by an optional dotted path ending in `.genslot`.
_RELATIVE_RE = re.compile(
    r"(\bfrom\s+\.[\w.]*?)"  # `from .` or `from ..foo.bar`
    r"(\bgenslot\b)"  # the segment to replace
)

# Patterns for old class names — word-boundary aware to avoid false positives.
_CLASS_RES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(rf"\b{old}\b"), new) for old, new in _CLASS_RENAMES
]


@dataclass
class GenStubFixLocation:
    """A single replacement within a file.

    Args:
        filepath: Path to the source file.
        line: One-based line number.
        description: Human-readable description of the replacement.
    """

    filepath: Path
    line: int
    description: str


@dataclass
class GenStubFixResult:
    """Aggregated results across all scanned files.

    Args:
        locations: Individual fix locations.
        total_fixes: Total replacements made (or found in dry-run).
        files_affected: Number of distinct files modified.
    """

    locations: list[GenStubFixLocation]
    total_fixes: int
    files_affected: int


def _fix_line(line: str) -> tuple[str, list[str]]:
    """Apply all genslot→genstub replacements to a single line.

    Returns:
        A (new_line, descriptions) tuple.  *descriptions* is empty when the
        line was not changed.
    """
    descriptions: list[str] = []

    # Fully-qualified module path.
    if _MODULE_RE.search(line):
        line = _MODULE_RE.sub(_MODULE_NEW, line)
        descriptions.append(f"{_MODULE_OLD} → {_MODULE_NEW}")

    # `from mellea.stdlib.components import genslot`
    if _FROM_PARENT_RE.search(line):
        line = _FROM_PARENT_RE.sub(r"\1genstub", line)
        descriptions.append("import genslot → import genstub")

    # Relative imports: `from .genslot import …`
    if _RELATIVE_RE.search(line):
        line = _RELATIVE_RE.sub(r"\1genstub", line)
        descriptions.append(".genslot → .genstub")

    for pattern, replacement in _CLASS_RES:
        if pattern.search(line):
            line = pattern.sub(replacement, line)
            old = pattern.pattern.replace(r"\b", "")
            descriptions.append(f"{old} → {replacement}")

    return line, descriptions


def find_genslot_refs(source: str, filepath: Path) -> list[GenStubFixLocation]:
    """Scan *source* for old genslot references and return their locations.

    Args:
        source: Python source text.
        filepath: Used for the ``filepath`` field in returned locations.

    Returns:
        List of locations that would be changed.
    """
    locations: list[GenStubFixLocation] = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        _, descriptions = _fix_line(line)
        for desc in descriptions:
            locations.append(
                GenStubFixLocation(filepath=filepath, line=lineno, description=desc)
            )
    return locations


def fix_genslot_file(
    filepath: Path, *, dry_run: bool = False
) -> list[GenStubFixLocation]:
    """Fix a single file.

    Args:
        filepath: Path to the Python file to fix.
        dry_run: If ``True``, return locations without modifying the file.

    Returns:
        List of locations found (and optionally fixed).
    """
    source = filepath.read_text()
    locations = find_genslot_refs(source, filepath)

    if not locations or dry_run:
        return locations

    new_lines: list[str] = []
    for line in source.splitlines(keepends=True):
        fixed, _ = _fix_line(line)
        new_lines.append(fixed)

    filepath.write_text("".join(new_lines))
    return locations


def fix_genslot_path(path: Path, *, dry_run: bool = False) -> GenStubFixResult:
    """Fix a file or directory recursively.

    Args:
        path: File or directory to process.
        dry_run: If ``True``, report locations without modifying files.

    Returns:
        Aggregated result with all fix locations and summary counts.
    """
    all_locations: list[GenStubFixLocation] = []
    files_affected = 0

    if path.is_file():
        files = [path]
    else:
        files = sorted(path.rglob("*.py"))

    for f in files:
        parts = f.relative_to(path).parts if path.is_dir() else ()
        if any(part in SKIP_DIRS for part in parts):
            continue

        locs = fix_genslot_file(f, dry_run=dry_run)
        if locs:
            all_locations.extend(locs)
            files_affected += 1

    return GenStubFixResult(
        locations=all_locations,
        total_fixes=len(all_locations),
        files_affected=files_affected,
    )
