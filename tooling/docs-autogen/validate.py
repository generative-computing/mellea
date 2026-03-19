#!/usr/bin/env python3
"""Validate generated API documentation.

Performs comprehensive validation checks on generated MDX files:
- GitHub source links point to correct repository and version
- API coverage meets minimum threshold
- MDX syntax is valid
- No broken internal links
- All required frontmatter present
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

_IN_GHA = os.environ.get("GITHUB_ACTIONS") == "true"

_ROOT_CONTRIB = (
    "https://github.com/generative-computing/mellea/blob/main/CONTRIBUTING.md"
)
_GUIDE_CONTRIB = (
    "https://github.com/generative-computing/mellea/blob/main"
    "/docs/docs/guide/CONTRIBUTING.md"
)

# Per-check fix hints: label (as passed to _print_check_errors) -> (fix text, ref URL)
_CHECK_FIX_HINTS: dict[str, tuple[str, str]] = {
    "Source links": (
        "Regenerate API docs to update version tags in source links: "
        "uv run python tooling/docs-autogen/generate-ast.py",
        f"{_ROOT_CONTRIB}#validating-docstrings",
    ),
    "MDX syntax": (
        "Check for unclosed ``` fences, missing frontmatter, or unescaped {{ }} "
        "in code blocks. Each error includes the file and line number.",
        f"{_GUIDE_CONTRIB}#frontmatter-required-on-every-page",
    ),
    "Internal links": (
        "Verify the relative link path resolves to an existing .mdx file. "
        "Each error shows the file, line, and broken link target.",
        f"{_GUIDE_CONTRIB}#links",
    ),
    "Anchor collisions": (
        "Rename one of the conflicting headings so they produce unique anchors. "
        "Each error shows both heading texts and their line numbers.",
        f"{_GUIDE_CONTRIB}#headings",
    ),
    "RST docstrings": (
        "Replace RST double-backtick notation (``Symbol``) with single backticks "
        "(`Symbol`) in the Python source docstring.",
        f"{_ROOT_CONTRIB}#docstrings",
    ),
    "Stale files": (
        "Delete the listed files — they are review artifacts or superseded content "
        "that should not ship.",
        f"{_GUIDE_CONTRIB}#missing-content",
    ),
    "Doc imports": (
        "Update the import path in the documentation code block to match the "
        "current module/symbol location.",
        f"{_GUIDE_CONTRIB}#code-and-fragment-consistency",
    ),
    "Examples catalogue": (
        "Add a row for the new example directory to docs/docs/examples/index.md.",
        f"{_GUIDE_CONTRIB}#keeping-the-examples-catalogue-up-to-date",
    ),
}

# GitHub Actions silently drops inline diff annotations beyond ~10 per step.
# We cap at 20 total across all validate.py checks so the most important issues
# from each category remain visible in the PR diff.  The complete list is always
# in the job log and (when --output is used) the JSON artifact.
_GHA_ANNOTATION_CAP = 20


# ---------------------------------------------------------------------------
# Error dict helpers
# ---------------------------------------------------------------------------

# Each check function returns list[dict] where every dict has:
#   file    — repo-relative file path (str); empty string for project-level issues
#   line    — 1-based line number (int); 0 for file- or project-level issues
#   message — human-readable description (str)


def _err(file: str, line: int, message: str) -> dict:
    return {"file": file, "line": line, "message": message}


def _line_of(content: str, match_start: int) -> int:
    """Return the 1-based line number for a byte offset in *content*."""
    return content[:match_start].count("\n") + 1


# ---------------------------------------------------------------------------
# GHA annotation helpers
# ---------------------------------------------------------------------------


def _gha_annotation(level: str, title: str, message: str, file: str = "", line: int = 0) -> None:
    """Emit a GitHub Actions workflow command annotation.

    When *file* and *line* are provided the annotation is anchored to that
    source location so it appears inline in the PR diff view.
    """
    for attr in ("message", "title", "file"):
        locals()[attr]  # reference to avoid lint unused-var
    message = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    title   = title  .replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    file    = file   .replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    if file and line:
        print(f"::{level} file={file},line={line},title={title}::{message}")
    else:
        print(f"::{level} title={title}::{message}")


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------


def validate_source_links(docs_dir: Path, version: str) -> tuple[int, list[dict]]:
    """Validate GitHub source links in generated MDX files.

    Args:
        docs_dir: Directory containing MDX files.
        version: Expected version string in links (e.g., ``"0.5.0"``).

    Returns:
        Tuple of (error_count, errors) where each error dict has keys
        ``file``, ``line``, and ``message``.
    """
    errors: list[dict] = []
    expected_repo = "ibm-granite/mellea"
    expected_pattern = f"https://github.com/{expected_repo}/blob/v{version}/"
    link_re = re.compile(r"\[View source\]\((https://github\.com/[^)]+)\)")

    for mdx_file in sorted(docs_dir.rglob("*.mdx")):
        content = mdx_file.read_text()
        rel = str(mdx_file.relative_to(docs_dir))
        for line_num, line in enumerate(content.splitlines(), 1):
            for match in link_re.finditer(line):
                link = match.group(1)
                if not link.startswith(expected_pattern):
                    errors.append(_err(
                        rel, line_num,
                        f"Invalid source link: {link}  (expected prefix: {expected_pattern})",
                    ))

    return len(errors), errors


def validate_coverage(docs_dir: Path, threshold: float) -> tuple[bool, dict]:
    """Validate API coverage meets threshold.

    Args:
        docs_dir: Directory containing MDX files.
        threshold: Minimum coverage percentage (0–100).

    Returns:
        Tuple of (passed, coverage_report).
    """
    sys.path.insert(0, str(Path(__file__).parent))

    try:
        from audit_coverage import (
            discover_cli_commands,
            discover_public_symbols,
            find_documented_symbols,
            generate_coverage_report,
        )
    except ImportError:
        return False, {"error": "audit_coverage.py not found"}

    source_dir = docs_dir.parent.parent.parent
    mellea_symbols = discover_public_symbols(source_dir / "mellea", "mellea")
    cli_symbols = discover_public_symbols(source_dir / "cli", "cli")
    cli_commands = discover_cli_commands(source_dir / "cli")
    documented = find_documented_symbols(docs_dir)

    all_symbols = {**mellea_symbols, **cli_symbols}
    report = generate_coverage_report(all_symbols, documented, cli_commands)
    return report["coverage_percentage"] >= threshold, report


def validate_mdx_syntax(docs_dir: Path) -> tuple[int, list[dict]]:
    """Validate MDX syntax in generated documentation files.

    Checks for unclosed code fences, unescaped curly braces inside code
    blocks, missing frontmatter, and a missing ``title`` field.

    Args:
        docs_dir: Directory containing MDX files.

    Returns:
        Tuple of (error_count, errors).
    """
    errors: list[dict] = []
    fence_re = re.compile(r"^```")

    for mdx_file in sorted(docs_dir.rglob("*.mdx")):
        content = mdx_file.read_text()
        rel = str(mdx_file.relative_to(docs_dir))
        lines = content.splitlines()

        # Unclosed code fence (odd number of ``` in the whole file)
        if content.count("```") % 2 != 0:
            errors.append(_err(rel, 0, "Unclosed code block (odd number of ```)"))

        # Unescaped curly braces inside code blocks
        in_code_block = False
        for line_num, line in enumerate(lines, 1):
            if fence_re.match(line):
                in_code_block = not in_code_block
                continue
            if not in_code_block:
                continue
            for pattern, brace in ((r"\{+", "{"), (r"\}+", "}")):
                for match in re.finditer(pattern, line):
                    seq = match.group()
                    if len(seq) % 2 != 0:
                        errors.append(_err(
                            rel, line_num,
                            f"Unescaped '{brace}' in code block: "
                            f"{len(seq)} consecutive '{brace}' (must be even) — "
                            f"line: {line.strip()[:80]}",
                        ))
                        break

        # Frontmatter presence and structure
        if not content.startswith("---\n"):
            errors.append(_err(rel, 1, "Missing frontmatter (file must start with ---)"))
        else:
            end = content.find("\n---\n", 4)
            if end == -1:
                errors.append(_err(rel, 1, "Malformed frontmatter (no closing ---)"))
            elif "title:" not in content[4:end]:
                errors.append(_err(rel, 1, "Missing 'title' field in frontmatter"))

    return len(errors), errors


def validate_internal_links(docs_dir: Path) -> tuple[int, list[dict]]:
    """Validate that relative internal links resolve to existing files.

    Args:
        docs_dir: Directory containing MDX files.

    Returns:
        Tuple of (error_count, errors).
    """
    errors: list[dict] = []
    link_re = re.compile(r"\[([^\]]+)\]\(([^)]+)\)", re.DOTALL)

    for mdx_file in sorted(docs_dir.rglob("*.mdx")):
        content = mdx_file.read_text()
        rel = str(mdx_file.relative_to(docs_dir))

        for match in link_re.finditer(content):
            link_text, link_url = match.groups()
            link_url = link_url.strip()

            if link_url.startswith(("http://", "https://", "#")):
                continue

            file_part = link_url.split("#", 1)[0] if "#" in link_url else link_url
            if file_part and not file_part.endswith(".mdx"):
                file_part = f"{file_part}.mdx"

            target = (mdx_file.parent / file_part).resolve()
            if not target.exists():
                errors.append(_err(
                    rel, _line_of(content, match.start()),
                    f"Broken link to '{link_url}' (text: '{link_text}')",
                ))

    return len(errors), errors


def validate_anchor_collisions(docs_dir: Path) -> tuple[int, list[dict]]:
    """Check for heading anchor collisions within MDX files.

    Args:
        docs_dir: Directory containing MDX files.

    Returns:
        Tuple of (error_count, errors).
    """
    errors: list[dict] = []

    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from test_mintlify_anchors import mintlify_anchor
    except ImportError:
        def mintlify_anchor(heading: str) -> str:  # type: ignore[misc]
            anchor = heading.lower().replace(" ", "-")
            anchor = re.sub(r"[^a-z0-9-]", "", anchor)
            return re.sub(r"-+", "-", anchor).strip("-")

    heading_re = re.compile(r"^#+\s+(.+)$", re.MULTILINE)

    for mdx_file in sorted(docs_dir.rglob("*.mdx")):
        content = mdx_file.read_text()
        rel = str(mdx_file.relative_to(docs_dir))

        # anchors maps anchor -> (heading_text, line_num) for the first occurrence
        seen: dict[str, tuple[str, int]] = {}
        for match in heading_re.finditer(content):
            heading = match.group(1)
            line_num = _line_of(content, match.start())
            anchor = mintlify_anchor(heading)
            if anchor in seen:
                prev_heading, prev_line = seen[anchor]
                errors.append(_err(
                    rel, line_num,
                    f"Anchor collision '{anchor}': "
                    f"'{prev_heading}' (line {prev_line}) and "
                    f"'{heading}' (line {line_num}) produce the same anchor",
                ))
            else:
                seen[anchor] = (heading, line_num)

    return len(errors), errors


def validate_rst_docstrings(source_dir: Path) -> tuple[int, list[dict]]:
    """Scan Python source files for RST double-backtick notation in docstrings.

    RST-style ``Symbol`` double-backtick markup interacts badly with the
    add_cross_references step: the regex matches the inner single-backtick
    boundary and generates a broken link wrapped in an extra code span, e.g.
    ``Backend`` → `[`Backend`](url)` which Mintlify renders as raw text
    rather than a clickable link.

    Args:
        source_dir: Root of the Python source tree to scan (e.g. repo/mellea).

    Returns:
        Tuple of (error_count, errors).
    """
    errors: list[dict] = []
    pattern = re.compile(r"``([A-Za-z][^`]*)``")

    for py_file in sorted(source_dir.rglob("*.py")):
        try:
            content = py_file.read_text(encoding="utf-8")
        except Exception:
            continue
        rel = str(py_file.relative_to(source_dir.parent))
        for line_num, line in enumerate(content.splitlines(), 1):
            if pattern.search(line):
                errors.append(_err(
                    rel, line_num,
                    f"RST double-backtick notation — use single backticks for "
                    f"Markdown/MDX compatibility: {line.strip()[:100]}",
                ))

    return len(errors), errors


def validate_stale_files(docs_root: Path) -> tuple[int, list[dict]]:
    """Check for stale files that should have been cleaned up.

    Detects review artifacts, superseded files, and other content that
    accumulates during doc rewrites and should not ship in a release.

    Args:
        docs_root: The ``docs/`` directory (parent of ``docs/docs/``).

    Returns:
        Tuple of (error_count, errors).
    """
    errors: list[dict] = []

    for f in docs_root.glob("*REVIEW*"):
        errors.append(_err(str(f.relative_to(docs_root)), 0, "Stale review artifact"))

    old_index = docs_root / "index.md"
    new_index = docs_root / "docs" / "index.mdx"
    if old_index.exists() and new_index.exists():
        errors.append(_err("index.md", 0, "Stale file — superseded by docs/index.mdx"))

    old_tutorial = docs_root / "tutorial.md"
    tutorials_dir = docs_root / "docs" / "tutorials"
    if old_tutorial.exists() and tutorials_dir.is_dir():
        errors.append(_err("tutorial.md", 0, "Stale file — superseded by docs/tutorials/"))

    return len(errors), errors


def validate_examples_catalogue(docs_root: Path) -> tuple[int, list[dict]]:
    """Check that every example directory is listed in the examples index page.

    Scans ``docs/examples/`` for subdirectories that contain at least one
    ``.py`` file and verifies each directory name appears in the catalogue
    table in ``docs/docs/examples/index.md``.

    Args:
        docs_root: The ``docs/`` directory (parent of ``docs/docs/``).

    Returns:
        Tuple of (error_count, errors).
    """
    errors: list[dict] = []
    examples_dir = docs_root / "examples"
    index_file = docs_root / "docs" / "examples" / "index.md"

    if not examples_dir.is_dir():
        return 0, []
    if not index_file.exists():
        errors.append(_err("docs/docs/examples/index.md", 0, "Examples index page not found"))
        return len(errors), errors

    index_content = index_file.read_text()
    skip = {"__pycache__", "helper"}

    for child in sorted(examples_dir.iterdir()):
        if not child.is_dir() or child.name in skip or child.name.startswith("."):
            continue
        if not any(child.rglob("*.py")):
            continue
        if f"`{child.name}/" not in index_content and f"`{child.name}`" not in index_content:
            errors.append(_err(
                f"docs/examples/{child.name}/", 0,
                f"Example directory '{child.name}/' is not listed in "
                f"docs/docs/examples/index.md",
            ))

    return len(errors), errors


def validate_doc_imports(docs_dir: Path) -> tuple[int, list[dict]]:
    """Verify that mellea imports in documentation code blocks still resolve.

    Parses fenced Python code blocks in static docs for ``from mellea.X import Y``
    and ``import mellea.X`` statements, then checks whether each module and symbol
    exists at import time.  Optional-dependency failures (``ImportError`` whose
    message mentions a third-party package) are silently skipped.

    Args:
        docs_dir: The ``docs/docs/`` directory containing static documentation.

    Returns:
        Tuple of (error_count, errors).
    """
    import importlib

    errors: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for doc_file in sorted(list(docs_dir.rglob("*.md")) + list(docs_dir.rglob("*.mdx"))):
        content = doc_file.read_text()
        rel = str(doc_file.relative_to(docs_dir))
        in_python = False

        for line_num, line in enumerate(content.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("```python"):
                in_python = True
                continue
            if stripped.startswith("```") and in_python:
                in_python = False
                continue
            if not in_python:
                continue

            # from mellea.X import Y, Z
            m = re.match(r"\s*from\s+(mellea[\w.]*)\s+import\s+(.+)", line)
            if m:
                module = m.group(1)
                names = [
                    n.strip().split(" as ")[0].strip().rstrip(",")
                    for n in m.group(2).split(",")
                ]
                for name in names:
                    if not name or not name.isidentifier():
                        continue
                    key = (module, name)
                    if key in seen:
                        continue
                    seen.add(key)
                    try:
                        mod = importlib.import_module(module)
                    except ImportError:
                        continue
                    if not hasattr(mod, name):
                        try:
                            importlib.import_module(f"{module}.{name}")
                        except ImportError:
                            errors.append(_err(
                                rel, line_num,
                                f"from {module} import {name} — symbol not found in module",
                            ))
                continue

            # import mellea.X
            m2 = re.match(r"\s*import\s+(mellea[\w.]+)", line)
            if m2:
                module = m2.group(1)
                key = (module, "")
                if key in seen:
                    continue
                seen.add(key)
                try:
                    importlib.import_module(module)
                except ImportError:
                    continue

    return len(errors), errors


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def generate_report(
    source_link_errors: list[dict],
    coverage_passed: bool,
    coverage_report: dict,
    mdx_errors: list[dict],
    link_errors: list[dict],
    anchor_errors: list[dict],
    rst_docstring_errors: list[dict] | None = None,
    stale_errors: list[dict] | None = None,
    import_errors: list[dict] | None = None,
    examples_catalogue_errors: list[dict] | None = None,
) -> dict:
    """Assemble the full validation report.

    Args:
        source_link_errors: Errors from :func:`validate_source_links`.
        coverage_passed: Whether coverage met the threshold.
        coverage_report: Raw coverage report dict.
        mdx_errors: Errors from :func:`validate_mdx_syntax`.
        link_errors: Errors from :func:`validate_internal_links`.
        anchor_errors: Errors from :func:`validate_anchor_collisions`.
        rst_docstring_errors: Errors from :func:`validate_rst_docstrings`.
        stale_errors: Errors from :func:`validate_stale_files`.
        import_errors: Errors from :func:`validate_doc_imports`.
        examples_catalogue_errors: Errors from :func:`validate_examples_catalogue`.

    Returns:
        Report dictionary with all validation results.
    """
    rst_docstring_errors = rst_docstring_errors or []
    stale_errors = stale_errors or []
    import_errors = import_errors or []
    examples_catalogue_errors = examples_catalogue_errors or []

    return {
        "source_links": {
            "passed": len(source_link_errors) == 0,
            "error_count": len(source_link_errors),
            "errors": source_link_errors,
        },
        "coverage": {
            "passed": coverage_passed,
            "percentage": coverage_report.get("coverage_percentage", 0),
            "total_symbols": coverage_report.get("total_symbols", 0),
            "documented_symbols": coverage_report.get("documented_symbols", 0),
            "missing_symbols": coverage_report.get("missing_symbols", {}),
        },
        "mdx_syntax": {
            "passed": len(mdx_errors) == 0,
            "error_count": len(mdx_errors),
            "errors": mdx_errors,
        },
        "internal_links": {
            "passed": len(link_errors) == 0,
            "error_count": len(link_errors),
            "errors": link_errors,
        },
        "anchor_collisions": {
            "passed": len(anchor_errors) == 0,
            "error_count": len(anchor_errors),
            "errors": anchor_errors,
        },
        "rst_docstrings": {
            "passed": len(rst_docstring_errors) == 0,
            "error_count": len(rst_docstring_errors),
            "errors": rst_docstring_errors,
        },
        "stale_files": {
            "passed": len(stale_errors) == 0,
            "error_count": len(stale_errors),
            "errors": stale_errors,
        },
        "doc_imports": {
            "passed": len(import_errors) == 0,
            "error_count": len(import_errors),
            "errors": import_errors,
        },
        "examples_catalogue": {
            "passed": len(examples_catalogue_errors) == 0,
            "error_count": len(examples_catalogue_errors),
            "errors": examples_catalogue_errors,
        },
        "overall_passed": (
            len(source_link_errors) == 0
            and coverage_passed
            and len(mdx_errors) == 0
            and len(link_errors) == 0
            and len(anchor_errors) == 0
            and len(stale_errors) == 0
            and len(import_errors) == 0
            and len(examples_catalogue_errors) == 0
            # rst_docstrings is warning-only — does not fail the build
        ),
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _icon(passed: bool) -> str:
    return "✅" if passed else "❌"


def _print_check_errors(
    label: str, errors: list[dict], gha_budget: list[int]
) -> None:
    """Print a grouped error block for one check to stdout and emit GHA annotations.

    GHA annotations are emitted until the shared *gha_budget* counter (a
    single-element list used as a mutable int) reaches :data:`_GHA_ANNOTATION_CAP`.
    This prevents later check categories from being invisible because an earlier
    category consumed the entire annotation budget.  The full error list is
    always printed to stdout regardless of the cap.

    Args:
        label: Human-readable check name shown in the section header.
        errors: List of error dicts with keys ``file``, ``line``, ``message``.
        gha_budget: Single-element list ``[remaining_annotations]``; decremented
            in-place as annotations are emitted.
    """
    if not errors:
        return
    fix_hint, ref_url = _CHECK_FIX_HINTS.get(label, ("", ""))
    print(f"\n{'─' * 50}")
    print(f"  {label} ({len(errors)} error(s))")
    if fix_hint:
        print(f"  Fix: {fix_hint}")
    if ref_url:
        print(f"  Ref: {ref_url}")
    print(f"{'─' * 50}")
    capped = 0
    for e in errors:
        print(f"  {e['file'] or '—'}{':' + str(e['line']) if e.get('line') else ''}")
        print(f"    {e['message']}")
        if _IN_GHA and gha_budget[0] > 0:
            _gha_annotation(
                "error",
                label,
                e["message"],
                file=e.get("file", ""),
                line=e.get("line", 0),
            )
            gha_budget[0] -= 1
        elif _IN_GHA and gha_budget[0] == 0:
            capped += 1
    if capped:
        print(
            f"  (GHA annotations exhausted — {capped} more error(s) in job log "
            f"and JSON artifact; total cap is {_GHA_ANNOTATION_CAP})"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Validate API documentation")
    parser.add_argument("docs_dir", help="Directory containing generated MDX files")
    parser.add_argument("--version", help="Expected version in GitHub links (e.g., 0.5.0)")
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=80.0,
        help="Minimum API coverage percentage (default: 80)",
    )
    parser.add_argument("--output", help="Output JSON report file")
    parser.add_argument("--skip-coverage", action="store_true", help="Skip coverage validation")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Python source root to scan for RST double-backtick notation (e.g. mellea/)",
    )
    parser.add_argument(
        "--docs-root",
        help="Root docs/ directory for stale-file checks (default: parent of docs_dir)",
    )
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists():
        print(f"ERROR: Directory not found: {docs_dir}", file=sys.stderr)
        sys.exit(1)

    print("🔍 Validating API documentation...")
    print(f"   Directory: {docs_dir}")
    if args.version:
        print(f"   Version: {args.version}")
    print(f"   Coverage threshold: {args.coverage_threshold}%")
    print()

    # Run all checks
    print("Checking GitHub source links...")
    _, source_link_errors = (
        validate_source_links(docs_dir, args.version) if args.version else (0, [])
    )

    print("Checking API coverage...")
    coverage_passed, coverage_report = (
        (True, {})
        if args.skip_coverage
        else validate_coverage(docs_dir, args.coverage_threshold)
    )

    print("Checking MDX syntax...")
    _, mdx_errors = validate_mdx_syntax(docs_dir)

    print("Checking internal links...")
    _, link_errors = validate_internal_links(docs_dir)

    print("Checking anchor collisions...")
    _, anchor_errors = validate_anchor_collisions(docs_dir)

    rst_docstring_errors: list[dict] = []
    if args.source_dir:
        print("Checking source docstrings for RST double-backtick notation...")
        _, rst_docstring_errors = validate_rst_docstrings(args.source_dir)

    print("Checking for stale files...")
    docs_root = Path(args.docs_root) if args.docs_root else docs_dir.parent
    _, stale_errors = validate_stale_files(docs_root)

    print("Checking doc imports...")
    static_docs_dir = docs_root / "docs" if docs_root else docs_dir.parent
    _, import_errors = validate_doc_imports(static_docs_dir)

    print("Checking examples catalogue...")
    _, examples_catalogue_errors = validate_examples_catalogue(docs_root)

    # Assemble report
    report = generate_report(
        source_link_errors,
        coverage_passed,
        coverage_report,
        mdx_errors,
        link_errors,
        anchor_errors,
        rst_docstring_errors,
        stale_errors,
        import_errors,
        examples_catalogue_errors,
    )

    # Summary table
    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)
    checks = [
        ("Source links",        report["source_links"]["passed"]),
        ("Coverage",            report["coverage"]["passed"]),
        ("MDX syntax",          report["mdx_syntax"]["passed"]),
        ("Internal links",      report["internal_links"]["passed"]),
        ("Anchor collisions",   report["anchor_collisions"]["passed"]),
        ("Stale files",         report["stale_files"]["passed"]),
        ("Doc imports",         report["doc_imports"]["passed"]),
        ("Examples catalogue",  report["examples_catalogue"]["passed"]),
    ]
    if args.source_dir:
        checks.append(("RST docstrings (warning)", report["rst_docstrings"]["passed"]))

    for label, passed in checks:
        print(f"{_icon(passed)} {label}: {'PASS' if passed else 'FAIL'}")

    if not args.skip_coverage:
        cov = report["coverage"]
        print(
            f"   Coverage: {cov['percentage']}%"
            f" ({cov['documented_symbols']}/{cov['total_symbols']} symbols)"
        )

    print("\n" + "=" * 60)
    print(f"Overall: {_icon(report['overall_passed'])} {'PASS' if report['overall_passed'] else 'FAIL'}")
    print("=" * 60)

    # Detailed errors — grouped by check
    error_sections = [
        ("Source links",        source_link_errors),
        ("MDX syntax",          mdx_errors),
        ("Internal links",      link_errors),
        ("Anchor collisions",   anchor_errors),
        ("RST docstrings",      rst_docstring_errors),
        ("Stale files",         stale_errors),
        ("Doc imports",         import_errors),
        ("Examples catalogue",  examples_catalogue_errors),
    ]
    any_errors = any(errs for _, errs in error_sections)
    if any_errors:
        print("\nDetailed errors:")
        gha_budget = [_GHA_ANNOTATION_CAP]
        for label, errs in error_sections:
            _print_check_errors(label, errs, gha_budget)

    # GHA summary annotation
    if _IN_GHA:
        if report["overall_passed"]:
            _gha_annotation("notice", "MDX Validation", "All checks passed")
        else:
            total_errors = sum(len(e) for _, e in error_sections)
            _gha_annotation(
                "error",
                "MDX Validation",
                f"{total_errors} error(s) found across {sum(1 for _, e in error_sections if e)} check(s) — see job summary for details",
            )

    # Save JSON report
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
        print(f"\n📄 Report saved to {output_path}")

    sys.exit(0 if report["overall_passed"] else 1)


if __name__ == "__main__":
    main()
