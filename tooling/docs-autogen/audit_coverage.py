#!/usr/bin/env python3
"""Audit API documentation coverage and docstring quality.

Discovers all public classes and functions in mellea/ and cli/ using Griffe,
then checks which ones have generated MDX documentation. Constants and module
attributes are excluded from the count — they are not expected to have
standalone documentation.

With --quality, also audits docstring quality: flags missing docstrings,
very short docstrings, and functions whose Args/Returns sections are absent.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

try:
    import griffe
except ImportError:
    print("ERROR: griffe not installed. Run: uv pip install griffe", file=sys.stderr)
    sys.exit(1)


def _load_package(source_dir: Path, package_name: str):
    """Load a package with Griffe. Returns the package object or None on failure."""
    try:
        # try_relative_path=False ensures Griffe only searches the explicit
        # search_paths and does not fall back to CWD, which avoids loading a
        # same-named package from the project root when --source-dir points
        # elsewhere (e.g. auditing mellea-b while running from mellea-d).
        search_path = str(source_dir.parent.resolve())
        return griffe.load(
            source_dir.name, search_paths=[search_path], try_relative_path=False
        )
    except Exception as e:
        print(f"WARNING: Failed to load {source_dir}: {e}", file=sys.stderr)
        return None


def discover_public_symbols(
    source_dir: Path, package_name: str
) -> dict[str, list[str]]:
    """Discover all public symbols using Griffe.

    Args:
        source_dir: Root directory to scan (e.g., mellea/ or cli/)
        package_name: Package name to prepend (e.g., "mellea" or "cli")

    Returns:
        Dict mapping full symbol paths to empty lists (for compatibility)
        Example: {"mellea.core.base.Component": [], "mellea.core.base.generative": []}
    """
    symbols: dict[str, list[str]] = {}
    package = _load_package(source_dir, package_name)
    if package is None:
        return symbols

    def walk_module(module, module_path: str):
        """Recursively walk through module and submodules."""
        # Skip internal modules (starting with _)
        if any(part.startswith("_") for part in module_path.split(".")):
            return

        # Get public classes and functions (not starting with _).
        # Constants/attributes are excluded — they are not expected to have
        # standalone documentation and would skew the coverage metric.
        # Aliases (re-exports from other modules) are also excluded — they are
        # documented at their canonical definition, not at each re-export site.
        for name, member in module.members.items():
            if not name.startswith("_"):
                try:
                    if getattr(member, "is_alias", False):
                        continue
                    if member.is_class or member.is_function:
                        full_path = f"{module_path}.{name}"
                        symbols[full_path] = []
                except Exception:
                    # Skip members that can't be resolved (e.g., aliases to stdlib)
                    pass

        # Recursively walk submodules (but skip internal ones)
        if hasattr(module, "modules"):
            for submodule_name, submodule in module.modules.items():
                if not submodule_name.startswith("_"):
                    submodule_path = f"{module_path}.{submodule_name}"
                    walk_module(submodule, submodule_path)

    # Walk through all top-level modules
    for module_name, module in package.modules.items():
        if not module_name.startswith("_"):
            module_path = f"{package_name}.{module_name}"
            walk_module(module, module_path)

    return symbols


# ---------------------------------------------------------------------------
# Docstring quality audit
# ---------------------------------------------------------------------------

_ARGS_RE = re.compile(r"^\s*(Args|Arguments|Parameters)\s*:", re.MULTILINE)
_RETURNS_RE = re.compile(r"^\s*Returns\s*:", re.MULTILINE)
_YIELDS_RE = re.compile(r"^\s*Yields\s*:", re.MULTILINE)
_RAISES_RE = re.compile(r"^\s*Raises\s*:", re.MULTILINE)
_ATTRIBUTES_RE = re.compile(r"^\s*Attributes\s*:", re.MULTILINE)
# Matches an indented param entry inside an Args block: "    param_name:" or "    param_name (type):"
# The colon must be followed by whitespace to avoid matching Sphinx cross-reference
# continuation lines like "        through :func:`...`".
_ARGS_ENTRY_RE = re.compile(r"^\s{4,}(\w+)\s*(?:\([^)]*\))?\s*:\s", re.MULTILINE)
# Return annotations that need no Returns section
_TRIVIAL_RETURNS = {"None", "NoReturn", "Never", "never", ""}

# Return annotations that indicate a generator (should use Yields, not Returns)
_GENERATOR_RETURN_PATTERNS = re.compile(
    r"Generator|Iterator|AsyncGenerator|AsyncIterator"
)


def _check_member(member, full_path: str, short_threshold: int) -> list[dict]:
    """Return quality issues for a single class or function member."""
    issues: list[dict] = []

    doc = getattr(member, "docstring", None)
    doc_text = doc.value.strip() if (doc and doc.value) else ""

    if not doc_text:
        issues.append({"path": full_path, "kind": "missing", "detail": "no docstring"})
        return issues  # no further checks without a docstring

    word_count = len(doc_text.split())
    if word_count < short_threshold:
        preview = doc_text[:70].replace("\n", " ")
        issues.append(
            {
                "path": full_path,
                "kind": "short",
                "detail": f'{word_count} word(s): "{preview}"',
            }
        )

    if getattr(member, "is_function", False):
        # Args section check: only flag when there are meaningful parameters.
        # Use Griffe ParameterKind to correctly exclude *args / **kwargs — their
        # names are stored without the leading '*', so a startswith("*") check
        # would not filter them.
        params = getattr(member, "parameters", None)
        _variadic_kinds = {
            griffe.ParameterKind.var_positional,
            griffe.ParameterKind.var_keyword,
        }
        concrete = [
            p.name
            for p in (params or [])
            if p.name not in ("self", "cls")
            and getattr(p, "kind", None) not in _variadic_kinds
        ]
        # A function whose only non-self params are *args/**kwargs is a variadic
        # forwarder (e.g. def f(*args, **kwargs)). Its docstring Args: section
        # documents accepted kwargs by convention, not a concrete signature —
        # skip both no_args and param_mismatch for these.
        is_variadic_forwarder = (not concrete) and any(
            getattr(p, "kind", None) in _variadic_kinds for p in (params or [])
        )
        if concrete and not _ARGS_RE.search(doc_text):
            sample = ", ".join(concrete[:3]) + ("..." if len(concrete) > 3 else "")
            issues.append(
                {
                    "path": full_path,
                    "kind": "no_args",
                    "detail": f"params [{sample}] have no Args section",
                }
            )
        elif concrete and not is_variadic_forwarder and _ARGS_RE.search(doc_text):
            # Param name mismatch: documented names that don't exist in the signature
            args_block = re.search(
                r"(?:Args|Arguments|Parameters)\s*:(.*?)(?:\n\s*\n|\Z)",
                doc_text,
                re.DOTALL,
            )
            if args_block:
                doc_param_names = set(_ARGS_ENTRY_RE.findall(args_block.group(1)))
                actual_names = set(concrete)
                phantom = doc_param_names - actual_names
                if phantom:
                    issues.append(
                        {
                            "path": full_path,
                            "kind": "param_mismatch",
                            "detail": f"documented params {sorted(phantom)} not in signature",
                        }
                    )

        # Returns/Yields section check: only flag when there is an explicit non-trivial annotation.
        # Generator return types (Generator, Iterator, etc.) require Yields:, not Returns:.
        returns = getattr(member, "returns", None)
        ret_str = str(returns).strip() if returns else ""
        if ret_str and ret_str not in _TRIVIAL_RETURNS:
            is_generator = bool(_GENERATOR_RETURN_PATTERNS.search(ret_str))
            if is_generator and not _YIELDS_RE.search(doc_text):
                issues.append(
                    {
                        "path": full_path,
                        "kind": "no_yields",
                        "detail": f"return type {ret_str!r} is a generator — needs Yields section, not Returns",
                    }
                )
            elif not is_generator and not _RETURNS_RE.search(doc_text):
                issues.append(
                    {
                        "path": full_path,
                        "kind": "no_returns",
                        "detail": f"return type {ret_str!r} has no Returns section",
                    }
                )

        # Raises section check: only flag when the source contains explicit raise statements
        source = getattr(member, "source", None) or ""
        if "raise " in source and not _RAISES_RE.search(doc_text):
            issues.append(
                {
                    "path": full_path,
                    "kind": "no_raises",
                    "detail": "function raises but has no Raises section",
                }
            )

    if getattr(member, "is_class", False):
        # Args section check for classes: look at __init__ typed parameters.
        # Variadic *args/**kwargs are excluded by kind (same logic as function check).
        _variadic_kinds = {
            griffe.ParameterKind.var_positional,
            griffe.ParameterKind.var_keyword,
        }
        init = member.members.get("__init__")
        if init:
            init_params = getattr(init, "parameters", None)
            typed_params = [
                p.name
                for p in (init_params or [])
                if p.name not in ("self", "cls")
                and getattr(p, "kind", None) not in _variadic_kinds
                and p.annotation is not None
            ]
            if typed_params and not _ARGS_RE.search(doc_text):
                sample = ", ".join(typed_params[:3]) + (
                    "..." if len(typed_params) > 3 else ""
                )
                issues.append(
                    {
                        "path": full_path,
                        "kind": "no_class_args",
                        "detail": f"__init__ params [{sample}] have no Args section",
                    }
                )

            # Duplicate Args check: Option C requires Args: on the class docstring only.
            # Flag when __init__ has its own Args: section in addition to the class's.
            init_doc = getattr(init, "docstring", None)
            init_doc_text = (
                init_doc.value.strip() if (init_doc and init_doc.value) else ""
            )
            if _ARGS_RE.search(init_doc_text) and _ARGS_RE.search(doc_text):
                issues.append(
                    {
                        "path": full_path,
                        "kind": "duplicate_init_args",
                        "detail": (
                            "Args: in both class and __init__ docstrings "
                            "(Option C: place Args: on class docstring only)"
                        ),
                    }
                )

    return issues


def audit_docstring_quality(
    source_dir: Path,
    package_name: str,
    short_threshold: int = 5,
    include_methods: bool = True,
    documented: set[str] | None = None,
) -> list[dict]:
    """Audit docstring quality for all public classes and functions.

    Checks each public symbol for:
    - missing: no docstring at all
    - short: docstring below short_threshold words
    - no_args: function with parameters but no Args/Parameters section
    - no_returns: function with a non-trivial return annotation but no Returns section
    - no_yields: generator function (Generator/Iterator return type) but no Yields section
    - no_raises: function whose source contains raise but has no Raises section
    - no_class_args: class whose __init__ has typed params but no Args section on the class
    - duplicate_init_args: Args: present in both class docstring and __init__ (Option C violation)
    - param_mismatch: Args section documents names absent from the real signature

    Note: Attributes: sections are intentionally not enforced. Under the Option C
    convention, Attributes: is only used when stored values differ in type or
    behaviour from the constructor inputs (e.g. type transforms, computed values,
    class constants). Pure-echo entries that repeat Args: verbatim are omitted.

    Only symbols (and methods whose parent class) present in `documented` are
    checked when that set is provided — ensuring the audit is scoped to what is
    actually surfaced in the API reference.

    Args:
        source_dir: Root directory to scan (e.g., mellea/ or cli/)
        package_name: Package name (e.g., "mellea" or "cli")
        short_threshold: Word count below which a docstring is flagged as short
        include_methods: Whether to audit public methods on classes in addition
            to top-level functions and classes
        documented: Set of symbol paths present in the generated MDX docs (from
            find_documented_symbols()). When provided, only documented symbols
            are audited. Pass None to audit all public symbols.

    Returns:
        List of issue dicts, each with keys: path, kind, detail
    """
    issues: list[dict] = []
    package = _load_package(source_dir, package_name)
    if package is None:
        return issues

    def walk_module(module, module_path: str) -> None:
        if any(part.startswith("_") for part in module_path.split(".")):
            return

        for name, member in module.members.items():
            if name.startswith("_"):
                continue
            try:
                if getattr(member, "is_alias", False):
                    continue
                if not (member.is_class or member.is_function):
                    continue
            except Exception:
                continue

            full_path = f"{module_path}.{name}"

            # Skip symbols not in the API reference when a filter is provided
            if documented is not None and full_path not in documented:
                continue

            issues.extend(_check_member(member, full_path, short_threshold))

            if include_methods and getattr(member, "is_class", False):
                for mname, method in member.members.items():
                    if mname.startswith("_"):
                        continue
                    try:
                        if getattr(method, "is_alias", False):
                            continue
                        if not getattr(method, "is_function", False):
                            continue
                    except Exception:
                        continue
                    issues.extend(
                        _check_member(method, f"{full_path}.{mname}", short_threshold)
                    )

        if hasattr(module, "modules"):
            for submodule_name, submodule in module.modules.items():
                if not submodule_name.startswith("_"):
                    walk_module(submodule, f"{module_path}.{submodule_name}")

    for module_name, module in package.modules.items():
        if not module_name.startswith("_"):
            walk_module(module, f"{package_name}.{module_name}")

    return issues


_IN_GHA = os.environ.get("GITHUB_ACTIONS") == "true"


def _gha_cmd(level: str, title: str, message: str) -> None:
    """Emit a GitHub Actions workflow command annotation."""
    # Escape special characters required by the GHA annotation format
    message = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    title = title.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::{level} title={title}::{message}")


def _print_quality_report(issues: list[dict]) -> None:
    """Print a grouped quality report to stdout."""
    by_kind: dict[str, list[dict]] = {}
    for issue in issues:
        by_kind.setdefault(issue["kind"], []).append(issue)

    kind_labels = {
        "missing": "Missing docstrings",
        "short": "Short docstrings",
        "no_args": "Missing Args section",
        "no_returns": "Missing Returns section",
        "no_yields": "Missing Yields section (generator)",
        "no_raises": "Missing Raises section",
        "no_class_args": "Missing class Args section",
        "duplicate_init_args": "Duplicate Args: in class + __init__ (Option C violation)",
        "param_mismatch": "Param name mismatches (documented but not in signature)",
    }

    total = len(issues)
    print(f"\n{'=' * 60}")
    print("Docstring Quality Report")
    print(f"{'=' * 60}")
    print(f"Total issues found: {total}")

    for kind in (
        "missing",
        "short",
        "no_args",
        "no_returns",
        "no_yields",
        "no_raises",
        "no_class_args",
        "duplicate_init_args",
        "param_mismatch",
    ):
        items = by_kind.get(kind, [])
        if not items:
            continue
        label = kind_labels.get(kind, kind)
        print(f"\n{'─' * 50}")
        print(f"  {label} ({len(items)})")
        print(f"{'─' * 50}")
        for item in sorted(items, key=lambda x: x["path"]):
            print(f"  {item['path']}")
            print(f"    {item['detail']}")


def audit_nav_orphans(docs_dir: Path, source_dir: Path) -> list[str]:
    """Find MDX files that exist on disk but are not linked in mint.json navigation.

    An orphaned module has a generated MDX file but no entry in the Mintlify
    navigation tree, so it is unreachable from the docs site.

    Args:
        docs_dir: Directory containing generated MDX files (e.g. docs/docs/api)
        source_dir: Project root, used to locate docs/mint.json

    Returns:
        Sorted list of orphaned module paths relative to docs_dir (no extension)
    """
    mint_json = source_dir / "docs" / "mint.json"

    mdx_files: set[str] = set()
    for mdx_file in docs_dir.rglob("*.mdx"):
        mdx_files.add(str(mdx_file.relative_to(docs_dir).with_suffix("")))

    nav_refs: set[str] = set()
    if mint_json.exists():
        config = json.loads(mint_json.read_text())

        def _extract(obj: object) -> None:
            if isinstance(obj, dict):
                if "page" in obj:
                    page = obj["page"]
                    if isinstance(page, str) and page.startswith("api/"):
                        nav_refs.add(page[len("api/") :])
                for v in obj.values():
                    _extract(v)
            elif isinstance(obj, list):
                for item in obj:
                    _extract(item)

        _extract(config)

    return sorted(mdx_files - nav_refs)


def discover_cli_commands(cli_dir: Path) -> list[str]:
    """Discover CLI commands from Typer applications.

    Args:
        cli_dir: Path to cli/ directory

    Returns:
        List of command names (e.g., ["m serve", "m alora", "m decompose"])
    """
    commands = []

    # Look for Typer app definitions
    # This is a simplified version - full implementation would parse the CLI structure
    main_file = cli_dir / "m.py"
    if main_file.exists():
        content = main_file.read_text()

        # Simple heuristic: look for @app.command() decorators or add_typer() calls

        # Find command decorators
        command_pattern = r'@app\.command\(["\']([^"\']+)["\']\)'
        commands.extend(re.findall(command_pattern, content))

        # Find subcommand additions
        typer_pattern = r'app\.add_typer\([^,]+,\s*name=["\']([^"\']+)["\']\)'
        commands.extend(re.findall(typer_pattern, content))

    return sorted(set(commands))


def find_documented_symbols(docs_dir: Path) -> set[str]:
    """Find which symbols have MDX documentation.

    Args:
        docs_dir: Path to docs/docs/api/ directory

    Returns:
        Set of documented symbol paths (e.g., {"mellea.core.base.Base"})
    """
    documented: set[str] = set()

    if not docs_dir.exists():
        return documented

    # Walk through all .mdx files
    for mdx_file in docs_dir.rglob("*.mdx"):
        # Convert file path to module path
        # e.g., mellea/core/base.mdx -> mellea.core.base
        rel_path = mdx_file.relative_to(docs_dir)
        module_path = str(rel_path.with_suffix("")).replace("/", ".")

        # Read file to find documented symbols
        content = mdx_file.read_text()

        # Look for heading patterns that indicate symbol documentation.
        # mdxify output format (0.2.37+): ### `SymbolName` <sup>...</sup>
        # Legacy formats kept for compatibility with older generated files.
        patterns = [
            r"^##\s+(?:class|function|attribute)\s+(\w+)",  # very old
            r"###\s+<span[^>]*>(?:FUNC|CLASS|ATTR)</span>\s+`(\w+)`",  # intermediate
            r"^###\s+`(\w+)`",  # current mdxify 0.2.37+
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                symbol_name = match.group(1)
                documented.add(f"{module_path}.{symbol_name}")

    return documented


def generate_coverage_report(
    discovered: dict[str, list[str]], documented: set[str], cli_commands: list[str]
) -> dict:
    """Generate coverage report.

    Args:
        discovered: Dict of full symbol paths (keys are "module.symbol", values are empty lists)
        documented: Set of documented symbols (full paths like "module.symbol")
        cli_commands: List of CLI commands

    Returns:
        Coverage report dict with statistics and missing symbols
    """
    # discovered is now a dict where keys are full paths like "mellea.core.base.Component"
    total_symbols = len(discovered)

    # Count how many discovered symbols are documented
    documented_count = len(discovered.keys() & documented)

    # Find missing symbols grouped by module
    missing: dict[str, list[str]] = {}
    for full_path in discovered.keys():
        if full_path not in documented:
            # Extract module and symbol name
            parts = full_path.rsplit(".", 1)
            if len(parts) == 2:
                module_path, symbol_name = parts
                if module_path not in missing:
                    missing[module_path] = []
                missing[module_path].append(symbol_name)

    coverage_pct = (documented_count / total_symbols * 100) if total_symbols > 0 else 0

    return {
        "total_symbols": total_symbols,
        "documented_symbols": documented_count,
        "coverage_percentage": round(coverage_pct, 2),
        "missing_symbols": missing,
        "cli_commands": cli_commands,
        "cli_documented": [],  # TODO: check CLI documentation
    }


def main():
    parser = argparse.ArgumentParser(description="Audit API documentation coverage")
    parser.add_argument("--source-dir", default=".", help="Project root directory")
    parser.add_argument(
        "--docs-dir",
        default=None,
        help="Generated docs directory (default: <source-dir>/docs/docs/api)",
    )
    parser.add_argument("--output", help="Output JSON file for report")
    parser.add_argument(
        "--threshold", type=float, default=80.0, help="Minimum coverage threshold"
    )
    parser.add_argument(
        "--quality",
        action="store_true",
        help="Run docstring quality audit (missing, short, no Args/Returns sections)",
    )
    parser.add_argument(
        "--short-threshold",
        type=int,
        default=5,
        metavar="N",
        help="Flag docstrings with fewer than N words as short (default: 5)",
    )
    parser.add_argument(
        "--no-methods",
        action="store_true",
        help="Exclude class methods from quality audit (check top-level symbols only)",
    )
    parser.add_argument(
        "--orphans",
        action="store_true",
        help="Check for MDX files not linked in docs/mint.json navigation",
    )
    parser.add_argument(
        "--fail-on-quality",
        action="store_true",
        help="Exit 1 if any quality issues are found (for CI/pre-commit use)",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    docs_dir = Path(args.docs_dir) if args.docs_dir else source_dir / "docs/docs/api"

    print("🔍 Discovering public symbols...")
    mellea_symbols = discover_public_symbols(source_dir / "mellea", "mellea")
    cli_symbols = discover_public_symbols(source_dir / "cli", "cli")

    print("🔍 Discovering CLI commands...")
    cli_commands = discover_cli_commands(source_dir / "cli")

    print("📚 Finding documented symbols...")
    documented = find_documented_symbols(docs_dir)

    print("📊 Generating coverage report...")
    all_symbols = {**mellea_symbols, **cli_symbols}
    report = generate_coverage_report(all_symbols, documented, cli_commands)

    # Print coverage report
    print(f"\n{'=' * 60}")
    print("API Documentation Coverage Report")
    print(f"{'=' * 60}")
    print(f"Total classes + functions: {report['total_symbols']}")
    print(f"Documented: {report['documented_symbols']}")
    print(f"Coverage: {report['coverage_percentage']}%")
    print(f"CLI commands: {len(report['cli_commands'])}")

    if report["missing_symbols"]:
        print(
            f"\n⚠️  Missing documentation for {len(report['missing_symbols'])} modules:"
        )
        for module, symbols in sorted(report["missing_symbols"].items()):
            print(f"  {module}: {', '.join(symbols)}")

    # Quality audit — scoped to documented (API reference) symbols only
    quality_issues: list[dict] = []
    if args.quality:
        print("\n🔬 Running docstring quality audit (documented symbols only)...")
        include_methods = not args.no_methods
        for pkg, pkg_name in [("mellea", "mellea"), ("cli", "cli")]:
            pkg_dir = source_dir / pkg
            if pkg_dir.exists():
                quality_issues.extend(
                    audit_docstring_quality(
                        pkg_dir,
                        pkg_name,
                        short_threshold=args.short_threshold,
                        include_methods=include_methods,
                        documented=documented,
                    )
                )
        _print_quality_report(quality_issues)

    # Nav orphan check — MDX files not referenced in mint.json navigation
    orphans: list[str] = []
    if args.orphans:
        print("\n🔗 Checking navigation orphans...")
        orphans = audit_nav_orphans(docs_dir, source_dir)
        print(f"\n{'=' * 60}")
        print("Navigation Orphans Report")
        print(f"{'=' * 60}")
        if orphans:
            print(f"⚠️  {len(orphans)} MDX file(s) not linked in navigation:")
            for orphan in orphans:
                print(f"  • {orphan}")
        else:
            print("✅ All MDX files are linked in navigation.")

    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        full_report = {**report}
        if args.quality:
            full_report["quality_issues"] = quality_issues
        if args.orphans:
            full_report["nav_orphans"] = orphans
        output_path.write_text(json.dumps(full_report, indent=2))
        print(f"\n✅ Report saved to {output_path}")

    # Check threshold
    failed = False
    if report["coverage_percentage"] < args.threshold:
        print(
            f"\n❌ Coverage {report['coverage_percentage']}% below threshold {args.threshold}%"
        )
        failed = True
    else:
        print(f"\n✅ Coverage meets threshold {args.threshold}%")

    if args.fail_on_quality and quality_issues:
        print(
            f"\n❌ {len(quality_issues)} quality issue(s) found (--fail-on-quality set)"
        )
        failed = True

    if _IN_GHA:
        if not args.quality:
            _gha_cmd(
                "notice", "Docstring quality", "skipped (pass --quality to enable)"
            )
        elif quality_issues:
            _gha_cmd(
                "error" if args.fail_on_quality else "warning",
                "Docstring quality",
                f"{len(quality_issues)} issue(s) found — see job summary for details",
            )
        else:
            _gha_cmd(
                "notice",
                "Docstring quality",
                "All documented symbols pass quality checks",
            )

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
