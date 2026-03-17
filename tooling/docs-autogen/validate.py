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
import re
import sys
from pathlib import Path


def validate_source_links(docs_dir: Path, version: str) -> tuple[int, list[str]]:
    """Validate GitHub source links.

    Args:
        docs_dir: Directory containing MDX files
        version: Expected version in links (e.g., "0.5.0")

    Returns:
        Tuple of (error_count, error_messages)
    """
    errors = []
    expected_repo = "ibm-granite/mellea"
    expected_pattern = f"https://github.com/{expected_repo}/blob/v{version}/"

    for mdx_file in docs_dir.rglob("*.mdx"):
        content = mdx_file.read_text()

        # Find all GitHub links
        link_pattern = r"\[View source\]\((https://github\.com/[^)]+)\)"
        for match in re.finditer(link_pattern, content):
            link = match.group(1)
            if not link.startswith(expected_pattern):
                rel_path = mdx_file.relative_to(docs_dir)
                errors.append(
                    f"{rel_path}: Invalid source link: {link}\n"
                    f"  Expected to start with: {expected_pattern}"
                )

    return len(errors), errors


def validate_coverage(docs_dir: Path, threshold: float) -> tuple[bool, dict]:
    """Validate API coverage meets threshold.

    Args:
        docs_dir: Directory containing MDX files
        threshold: Minimum coverage percentage (0-100)

    Returns:
        Tuple of (passed, coverage_report)
    """
    # Import audit_coverage functionality
    import sys

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

    # Run coverage audit
    source_dir = docs_dir.parent.parent.parent  # Go up to project root
    mellea_symbols = discover_public_symbols(source_dir / "mellea", "mellea")
    cli_symbols = discover_public_symbols(source_dir / "cli", "cli")
    cli_commands = discover_cli_commands(source_dir / "cli")
    documented = find_documented_symbols(docs_dir)

    all_symbols = {**mellea_symbols, **cli_symbols}
    report = generate_coverage_report(all_symbols, documented, cli_commands)

    passed = report["coverage_percentage"] >= threshold
    return passed, report


def validate_mdx_syntax(docs_dir: Path) -> tuple[int, list[str]]:
    """Validate MDX syntax.

    Args:
        docs_dir: Directory containing MDX files

    Returns:
        Tuple of (error_count, error_messages)
    """
    errors = []

    for mdx_file in docs_dir.rglob("*.mdx"):
        content = mdx_file.read_text()
        rel_path = mdx_file.relative_to(docs_dir)

        # Check for unclosed code blocks
        code_block_count = content.count("```")
        if code_block_count % 2 != 0:
            errors.append(f"{rel_path}: Unclosed code block (odd number of ```)")

        # Check for unescaped curly braces in code blocks
        lines = content.splitlines()
        in_code_block = False
        code_fence_pattern = re.compile(r"^```")

        for line_num, line in enumerate(lines, 1):
            if code_fence_pattern.match(line):
                in_code_block = not in_code_block
                continue

            if in_code_block:
                # Check for sequences of { or } that have odd length (indicating unescaped)
                # Properly escaped: {{ or }} (even length)
                # Unescaped: { or } or {{{ or }}} (odd length)

                # Find all sequences of consecutive open braces
                for match in re.finditer(r"\{+", line):
                    brace_seq = match.group()
                    if len(brace_seq) % 2 != 0:  # Odd number = unescaped
                        errors.append(
                            f"{rel_path}:{line_num}: Unescaped curly brace in code block (found {len(brace_seq)} consecutive '{{' - should be even)\n"
                            f"  Line: {line.strip()[:80]}"
                        )
                        break  # Only report once per line

                # Find all sequences of consecutive close braces
                for match in re.finditer(r"\}+", line):
                    brace_seq = match.group()
                    if len(brace_seq) % 2 != 0:  # Odd number = unescaped
                        errors.append(
                            f"{rel_path}:{line_num}: Unescaped curly brace in code block (found {len(brace_seq)} consecutive '}}' - should be even)\n"
                            f"  Line: {line.strip()[:80]}"
                        )
                        break  # Only report once per line

        # Check for frontmatter
        if not content.startswith("---\n"):
            errors.append(f"{rel_path}: Missing frontmatter")

        # Check for required frontmatter fields
        if content.startswith("---\n"):
            frontmatter_end = content.find("\n---\n", 4)
            if frontmatter_end == -1:
                errors.append(f"{rel_path}: Malformed frontmatter (no closing ---)")
            else:
                frontmatter = content[4:frontmatter_end]
                if "title:" not in frontmatter:
                    errors.append(f"{rel_path}: Missing 'title' in frontmatter")

    return len(errors), errors


def validate_internal_links(docs_dir: Path) -> tuple[int, list[str]]:
    """Validate internal links point to existing files.

    Args:
        docs_dir: Directory containing MDX files

    Returns:
        Tuple of (error_count, error_messages)
    """
    errors = []

    for mdx_file in docs_dir.rglob("*.mdx"):
        content = mdx_file.read_text()
        rel_path = mdx_file.relative_to(docs_dir)

        # Find relative links (not starting with http)
        # Use DOTALL to handle multiline links
        link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
        for match in re.finditer(link_pattern, content, re.DOTALL):
            link_text, link_url = match.groups()

            # Strip whitespace from URL (handles multiline links)
            link_url = link_url.strip()

            # Skip external links
            if link_url.startswith(("http://", "https://", "#")):
                continue

            # Split anchor from path (e.g., "base#class-component" -> "base", "class-component")
            if "#" in link_url:
                file_path, _ = link_url.split("#", 1)  # anchor not used
            else:
                file_path = link_url

            # Resolve relative link - add .mdx extension if not present
            if file_path and not file_path.endswith(".mdx"):
                file_path = f"{file_path}.mdx"

            target = (mdx_file.parent / file_path).resolve()

            if not target.exists():
                errors.append(
                    f"{rel_path}: Broken link to '{link_url}' (text: '{link_text}')"
                )

    return len(errors), errors


def validate_anchor_collisions(docs_dir: Path) -> tuple[int, list[str]]:
    """Check for anchor collisions within files.

    Args:
        docs_dir: Directory containing MDX files

    Returns:
        Tuple of (error_count, error_messages)
    """
    errors = []

    # Import mintlify_anchor function
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from test_mintlify_anchors import mintlify_anchor
    except ImportError:
        # Fallback implementation
        def mintlify_anchor(heading: str) -> str:
            anchor = heading.lower().replace(" ", "-")
            anchor = re.sub(r"[^a-z0-9-]", "", anchor)
            anchor = re.sub(r"-+", "-", anchor)
            return anchor.strip("-")

    for mdx_file in docs_dir.rglob("*.mdx"):
        content = mdx_file.read_text()
        rel_path = mdx_file.relative_to(docs_dir)

        # Extract all headings
        heading_pattern = r"^#+\s+(.+)$"
        headings = re.findall(heading_pattern, content, re.MULTILINE)

        # Generate anchors
        anchors: dict[str, list[str]] = {}
        for heading in headings:
            anchor = mintlify_anchor(heading)
            if anchor in anchors:
                errors.append(
                    f"{rel_path}: Anchor collision '{anchor}' from headings:\n"
                    f"  1. {anchors[anchor]}\n"
                    f"  2. {heading}"
                )
            else:
                anchors[anchor] = heading

    return len(errors), errors


def validate_rst_docstrings(source_dir: Path) -> tuple[int, list[str]]:
    """Scan Python source files for RST double-backtick notation in docstrings.

    RST-style ``Symbol`` double-backtick markup interacts badly with the
    add_cross_references step: the regex matches the inner single-backtick
    boundary and generates a broken link wrapped in an extra code span, e.g.
    ``Backend`` → `[`Backend`](url)` which Mintlify renders as raw text
    rather than a clickable link.

    Args:
        source_dir: Root of the Python source tree to scan (e.g. repo/mellea)

    Returns:
        Tuple of (error_count, error_messages)
    """
    errors = []
    # Match ``Word`` where Word starts with a letter (RST inline literal)
    pattern = re.compile(r"``([A-Za-z][^`]*)``")

    for py_file in source_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
        except Exception:
            continue

        for line_num, line in enumerate(content.splitlines(), 1):
            if pattern.search(line):
                rel = py_file.relative_to(source_dir.parent)
                errors.append(
                    f"{rel}:{line_num}: RST double-backtick notation — "
                    f"use single backticks for Markdown/MDX compatibility\n"
                    f"  {line.strip()[:100]}"
                )

    return len(errors), errors


def validate_stale_files(docs_root: Path) -> tuple[int, list[str]]:
    """Check for stale files that should have been cleaned up.

    Detects review artifacts, superseded files, and other content that
    accumulates during doc rewrites and should not ship in a release.

    Args:
        docs_root: The ``docs/`` directory (parent of ``docs/docs/``).

    Returns:
        Tuple of (error_count, error_messages).
    """
    errors = []

    # Review tracker artifacts (e.g. PR601-REVIEW.md)
    for f in docs_root.glob("*REVIEW*"):
        errors.append(f"Stale review artifact: {f.relative_to(docs_root)}")

    # Superseded landing page: docs/index.md replaced by docs/docs/index.mdx
    old_index = docs_root / "index.md"
    new_index = docs_root / "docs" / "index.mdx"
    if old_index.exists() and new_index.exists():
        errors.append("Stale file: index.md — superseded by docs/index.mdx")

    # Legacy tutorial: docs/tutorial.md replaced by docs/docs/tutorials/
    old_tutorial = docs_root / "tutorial.md"
    tutorials_dir = docs_root / "docs" / "tutorials"
    if old_tutorial.exists() and tutorials_dir.is_dir():
        errors.append("Stale file: tutorial.md — superseded by docs/tutorials/")

    return len(errors), errors


def validate_doc_imports(docs_dir: Path) -> tuple[int, list[str]]:
    """Verify that mellea imports in documentation code blocks still resolve.

    Parses fenced Python code blocks in static docs for ``from mellea.X import Y``
    and ``import mellea.X`` statements, then checks whether each module and symbol
    exists at import time.  Optional-dependency failures (``ImportError`` whose
    message mentions a third-party package) are silently skipped.

    Args:
        docs_dir: The ``docs/docs/`` directory containing static documentation.

    Returns:
        Tuple of (error_count, error_messages).
    """
    import importlib

    errors: list[str] = []
    seen: set[tuple[str, str]] = set()

    for doc_file in sorted(
        list(docs_dir.rglob("*.md")) + list(docs_dir.rglob("*.mdx"))
    ):
        content = doc_file.read_text()
        rel = doc_file.relative_to(docs_dir)
        in_python = False

        for line in content.splitlines():
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
                        # Optional dependency not installed — skip
                        continue
                    if not hasattr(mod, name):
                        # Could be a submodule (e.g. from pkg import submod)
                        try:
                            importlib.import_module(f"{module}.{name}")
                        except ImportError:
                            errors.append(
                                f"{rel}: from {module} import {name}"
                                f" — symbol not found in module"
                            )
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


def generate_report(
    source_link_errors: list[str],
    coverage_passed: bool,
    coverage_report: dict,
    mdx_errors: list[str],
    link_errors: list[str],
    anchor_errors: list[str],
    rst_docstring_errors: list[str] | None = None,
    stale_errors: list[str] | None = None,
    import_errors: list[str] | None = None,
) -> dict:
    """Generate validation report.

    Returns:
        Report dictionary with all validation results.
    """
    if stale_errors is None:
        stale_errors = []
    if import_errors is None:
        import_errors = []

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
            "passed": len(rst_docstring_errors or []) == 0,
            "error_count": len(rst_docstring_errors or []),
            "errors": rst_docstring_errors or [],
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
        "overall_passed": (
            len(source_link_errors) == 0
            and coverage_passed
            and len(mdx_errors) == 0
            and len(link_errors) == 0
            and len(anchor_errors) == 0
            and len(stale_errors) == 0
            and len(import_errors) == 0
            # rst_docstrings is a warning only — does not fail the build
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate API documentation")
    parser.add_argument("docs_dir", help="Directory containing generated MDX files")
    parser.add_argument(
        "--version", help="Expected version in GitHub links (e.g., 0.5.0)"
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=80.0,
        help="Minimum API coverage percentage (default: 80)",
    )
    parser.add_argument("--output", help="Output JSON report file")
    parser.add_argument(
        "--skip-coverage", action="store_true", help="Skip coverage validation"
    )
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

    # Run validations
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

    rst_docstring_errors: list[str] = []
    if args.source_dir:
        print("Checking source docstrings for RST double-backtick notation...")
        _, rst_docstring_errors = validate_rst_docstrings(args.source_dir)

    print("Checking for stale files...")
    docs_root = Path(args.docs_root) if args.docs_root else docs_dir.parent
    _, stale_errors = validate_stale_files(docs_root)

    print("Checking doc imports...")
    static_docs_dir = docs_root / "docs" if docs_root else docs_dir.parent
    _, import_errors = validate_doc_imports(static_docs_dir)

    # Generate report
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
    )

    # Print results
    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)

    print(f"✅ Source links: {'PASS' if report['source_links']['passed'] else 'FAIL'}")
    if not report["source_links"]["passed"]:
        print(f"   {report['source_links']['error_count']} errors found")

    print(f"✅ Coverage: {'PASS' if report['coverage']['passed'] else 'FAIL'}")
    if not args.skip_coverage:
        print(
            f"   {report['coverage']['percentage']}% "
            f"({report['coverage']['documented_symbols']}/{report['coverage']['total_symbols']} symbols)"
        )

    print(f"✅ MDX syntax: {'PASS' if report['mdx_syntax']['passed'] else 'FAIL'}")
    if not report["mdx_syntax"]["passed"]:
        print(f"   {report['mdx_syntax']['error_count']} errors found")

    print(
        f"✅ Internal links: {'PASS' if report['internal_links']['passed'] else 'FAIL'}"
    )
    if not report["internal_links"]["passed"]:
        print(f"   {report['internal_links']['error_count']} errors found")

    print(
        f"✅ Anchor collisions: {'PASS' if report['anchor_collisions']['passed'] else 'FAIL'}"
    )
    if not report["anchor_collisions"]["passed"]:
        print(f"   {report['anchor_collisions']['error_count']} errors found")

    if args.source_dir:
        print(
            f"✅ RST docstrings: {'PASS' if report['rst_docstrings']['passed'] else 'FAIL'}"
        )
        if not report["rst_docstrings"]["passed"]:
            print(f"   {report['rst_docstrings']['error_count']} occurrences found")

    print(f"✅ Stale files: {'PASS' if report['stale_files']['passed'] else 'FAIL'}")
    if not report["stale_files"]["passed"]:
        print(f"   {report['stale_files']['error_count']} errors found")

    print(f"✅ Doc imports: {'PASS' if report['doc_imports']['passed'] else 'FAIL'}")
    if not report["doc_imports"]["passed"]:
        print(f"   {report['doc_imports']['error_count']} errors found")

    print("\n" + "=" * 60)
    print(f"Overall: {'✅ PASS' if report['overall_passed'] else '❌ FAIL'}")
    print("=" * 60)

    # Print detailed errors
    if not report["overall_passed"]:
        print("\nDetailed Errors:")
        all_errors = (
            source_link_errors
            + mdx_errors
            + link_errors
            + anchor_errors
            + rst_docstring_errors
            + stale_errors
            + import_errors
        )
        for error in all_errors:
            print(f"  • {error}")

    # Save report
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
        print(f"\n📄 Report saved to {output_path}")

    # Exit with appropriate code
    sys.exit(0 if report["overall_passed"] else 1)


if __name__ == "__main__":
    main()
