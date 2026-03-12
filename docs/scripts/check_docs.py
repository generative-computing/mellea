#!/usr/bin/env python3
"""Validate Mellea documentation: links and Python code snippets.

Standalone script — no dependencies beyond Python 3.10+ stdlib.
Idempotent: read-only, reports problems to stdout, exits non-zero
if any hard errors are found.

Usage
-----
    python docs/scripts/check_docs.py                # run all checks
    python docs/scripts/check_docs.py links           # links only
    python docs/scripts/check_docs.py code            # code only
    python docs/scripts/check_docs.py shell           # shell quoting only
    python docs/scripts/check_docs.py --verbose       # show every item checked

Link checks
-----------
* Internal doc-to-doc links (relative paths within docs/docs/).
* Mintlify absolute paths (/getting-started/installation etc.) resolved
  against docs/docs/ and docs.json navigation.
* Mintlify Card href="..." attributes (JSX).
* Links that escape docs/docs/ (e.g. ../../examples/) — these resolve
  on the local filesystem but NOT on the published Mintlify site.  They
  are flagged as errors: use a full GitHub URL instead.
* External URLs (https://) — checked with a lightweight HEAD request.
  Failures are reported as warnings (network-dependent).
* docs.json navbar links and nav page slugs.

Code checks
-----------
* Syntax — every ```python block is compiled with compile().
  Snippets that fail only because of `await` outside a function or
  leading indentation are classified as *fragments* (warning, not error).
* Import analysis — top-level imports are checked for availability.
  mellea.* imports are checked against the repo source tree.
  Third-party imports that can't be found produce warnings.
* Missing-import heuristic — flags known mellea names used but never
  imported.
* Duplicate detection — code blocks of 4+ non-blank lines that appear
  identically in different files are flagged for consolidation.

Shell checks
------------
* Scans ```bash / ```shell blocks for `pip install X[extras]` or
  `uv pip install X[extras]` without shell quoting.  Unquoted square
  brackets break in zsh.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import re
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent  # docs/scripts/../../
DOCS_ROOT = REPO_ROOT / "docs" / "docs"  # Mintlify content root

# Skip API reference pages (separate PR)
SKIP_PREFIXES = ("api/",)

# GitHub base for converting escaped relative links
GITHUB_BASE = "https://github.com/generative-computing/mellea/blob/main"

# Timeout for external URL checks (seconds)
HTTP_TIMEOUT = 10

# ---------------------------------------------------------------------------
# Shared: collect doc files
# ---------------------------------------------------------------------------


def collect_doc_files() -> list[Path]:
    """Return all .md and .mdx files under DOCS_ROOT, skipping API ref."""
    files: list[Path] = []
    for ext in ("*.md", "*.mdx"):
        for p in sorted(DOCS_ROOT.rglob(ext)):
            rel = p.relative_to(DOCS_ROOT).as_posix()
            if any(rel.startswith(pfx) for pfx in SKIP_PREFIXES):
                continue
            files.append(p)
    return files


# ===================================================================
# LINK CHECKING
# ===================================================================

# Markdown link: [text](target) — but not images ![alt](src)
MD_LINK_RE = re.compile(r"(?<!!)\[(?:[^\]]*)\]\(([^)]+)\)")

# Mintlify Card href="..." (JSX)
HREF_RE = re.compile(r'href="([^"]+)"')


def extract_links(filepath: Path) -> list[tuple[int, str]]:
    """Return (line_number, raw_target) pairs from a file."""
    links: list[tuple[int, str]] = []
    text = filepath.read_text(encoding="utf-8", errors="replace")
    for lineno, line in enumerate(text.splitlines(), start=1):
        for m in MD_LINK_RE.finditer(line):
            links.append((lineno, m.group(1)))
        for m in HREF_RE.finditer(line):
            links.append((lineno, m.group(1)))
    return links


def is_external(target: str) -> bool:
    return target.startswith(("http://", "https://", "mailto:"))


def is_anchor_only(target: str) -> bool:
    return target.startswith("#")


def strip_anchor(target: str) -> str:
    return target.split("#", 1)[0]


def file_exists_mintlify(resolved: Path) -> bool:
    """Check whether the resolved target exists, trying Mintlify
    extension conventions (.md, .mdx, index files)."""
    if resolved.exists():
        return True
    if resolved.with_suffix(".md").exists():
        return True
    if resolved.with_suffix(".mdx").exists():
        return True
    if resolved.is_dir():
        if (resolved / "index.md").exists():
            return True
        if (resolved / "index.mdx").exists():
            return True
    return False


def check_external_url(url: str, cache: dict[str, int | str]) -> int | str:
    """HEAD-check an external URL.  Returns HTTP status code or error string.
    Results are cached for the session."""
    if url in cache:
        return cache[url]
    # Create an SSL context that doesn't verify (avoids cert issues in CI)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": "mellea-doc-checker/1"}
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT, context=ctx) as resp:
            cache[url] = resp.status
            return resp.status
    except urllib.error.HTTPError as exc:
        cache[url] = exc.code
        return exc.code
    except Exception as exc:
        result = f"error: {exc}"
        cache[url] = result
        return result


def load_nav_pages() -> set[str]:
    """Return the set of page slugs declared in docs.json navigation."""
    docs_json = DOCS_ROOT / "docs.json"
    if not docs_json.exists():
        return set()
    with open(docs_json, encoding="utf-8") as f:
        data = json.load(f)
    pages: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, str):
            pages.add(node)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            for key in ("pages", "groups", "tabs"):
                if key in node:
                    walk(node[key])

    walk(data.get("navigation", {}))
    return pages


def load_navbar_links() -> list[tuple[str, str]]:
    """Return (label, href) for links in docs.json navbar."""
    docs_json = DOCS_ROOT / "docs.json"
    if not docs_json.exists():
        return []
    with open(docs_json, encoding="utf-8") as f:
        data = json.load(f)
    links: list[tuple[str, str]] = []
    navbar = data.get("navbar", {})
    primary = navbar.get("primary", {})
    if "href" in primary:
        links.append((primary.get("label", "primary"), primary["href"]))
    for item in navbar.get("links", []):
        if "href" in item:
            links.append((item.get("label", ""), item["href"]))
    return links


def run_link_checks(
    doc_files: list[Path], verbose: bool, check_external: bool
) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) from link checking."""
    errors: list[str] = []
    warnings: list[str] = []
    url_cache: dict[str, int | str] = {}
    total_links = 0
    total_external = 0

    for filepath in doc_files:
        rel = filepath.relative_to(DOCS_ROOT)
        links = extract_links(filepath)

        for lineno, raw_target in links:
            total_links += 1

            # Pure anchor — skip
            if is_anchor_only(raw_target):
                if verbose:
                    print(f"  [skip] {rel}:{lineno} -> {raw_target} (anchor)")
                continue

            # External URL
            if is_external(raw_target):
                total_external += 1
                if check_external:
                    result = check_external_url(raw_target, url_cache)
                    if isinstance(result, int) and 200 <= result < 400:
                        if verbose:
                            print(f"  [ok]   {rel}:{lineno} -> {raw_target} ({result})")
                    elif isinstance(result, int) and result == 404:
                        errors.append(
                            f"  {rel}:{lineno} -> {raw_target}  [HTTP {result}]"
                        )
                    elif isinstance(result, int):
                        warnings.append(
                            f"  {rel}:{lineno} -> {raw_target}  [HTTP {result}]"
                        )
                    else:
                        warnings.append(f"  {rel}:{lineno} -> {raw_target}  [{result}]")
                elif verbose:
                    print(f"  [skip] {rel}:{lineno} -> {raw_target} (external)")
                continue

            # Internal link — resolve
            target_clean = strip_anchor(raw_target)
            if not target_clean:
                continue

            # Absolute Mintlify path
            if target_clean.startswith("/"):
                # Static assets
                if target_clean.startswith(("/images/", "/logo/")):
                    resolved = DOCS_ROOT / target_clean.lstrip("/")
                    if not resolved.exists():
                        errors.append(
                            f"  {rel}:{lineno} -> {raw_target}"
                            f"  [static asset not found]"
                        )
                    elif verbose:
                        print(f"  [ok]   {rel}:{lineno} -> {raw_target}")
                    continue

                resolved = DOCS_ROOT / target_clean.lstrip("/")
                if file_exists_mintlify(resolved):
                    if verbose:
                        print(f"  [ok]   {rel}:{lineno} -> {raw_target}")
                else:
                    errors.append(
                        f"  {rel}:{lineno} -> {raw_target}"
                        f"  [page not found under docs/docs/]"
                    )
                continue

            # Relative path
            source_dir = filepath.parent
            resolved = (source_dir / target_clean).resolve()

            # Check if the resolved path escapes DOCS_ROOT
            try:
                resolved.relative_to(DOCS_ROOT)
                inside_docs = True
            except ValueError:
                inside_docs = False

            if not inside_docs:
                # It might still exist in the repo...
                if resolved.exists() or Path(str(resolved)).exists():
                    # File exists in repo but won't work on Mintlify site
                    # Suggest the GitHub URL
                    try:
                        repo_rel = resolved.relative_to(REPO_ROOT)
                        suggested = f"{GITHUB_BASE}/{repo_rel}"
                    except ValueError:
                        suggested = "(could not compute GitHub URL)"
                    errors.append(
                        f"  {rel}:{lineno} -> {raw_target}"
                        f"  [escapes docs/ — won't work on Mintlify."
                        f" Suggest: {suggested}]"
                    )
                else:
                    errors.append(
                        f"  {rel}:{lineno} -> {raw_target}"
                        f"  [file not found, and escapes docs/]"
                    )
                if verbose:
                    print(f"  [ESC]  {rel}:{lineno} -> {raw_target}")
                continue

            # Normal internal link
            if file_exists_mintlify(resolved):
                if verbose:
                    print(f"  [ok]   {rel}:{lineno} -> {raw_target}")
            else:
                errors.append(f"  {rel}:{lineno} -> {raw_target}  [file not found]")

    # docs.json nav page slugs
    nav_pages = load_nav_pages()
    for slug in sorted(nav_pages):
        if any(slug.startswith(pfx) for pfx in SKIP_PREFIXES):
            continue
        resolved = DOCS_ROOT / slug
        if not file_exists_mintlify(resolved):
            errors.append(f"  docs.json nav: '{slug}' — file not found")
        elif verbose:
            print(f"  [ok]   docs.json nav: {slug}")

    # docs.json navbar links (external URLs)
    if check_external:
        for label, href in load_navbar_links():
            if is_external(href):
                result = check_external_url(href, url_cache)
                if isinstance(result, int) and result == 404:
                    errors.append(f"  docs.json navbar '{label}': {href}  [HTTP 404]")
                elif isinstance(result, int) and result >= 400:
                    warnings.append(
                        f"  docs.json navbar '{label}': {href}  [HTTP {result}]"
                    )
                elif isinstance(result, str):
                    warnings.append(f"  docs.json navbar '{label}': {href}  [{result}]")
                elif verbose:
                    print(f"  [ok]   docs.json navbar '{label}': {href}")

    print(
        f"\nLinks: scanned {len(doc_files)} files, "
        f"{total_links} links ({total_external} external)"
    )

    return errors, warnings


# ===================================================================
# CODE CHECKING
# ===================================================================

FENCE_OPEN_RE = re.compile(r"^```(?:python|py)\b.*$")
FENCE_CLOSE_RE = re.compile(r"^```\s*$")

# Known mellea names that should be imported when used
MELLEA_NAMES = {
    "mellea",
    "generative",
    "mify",
    "MelleaTool",
    "SimpleContext",
    "instruct",
    "start_session",
    "act",
    "aact",
    "GenSlot",
    "Requirement",
    "PydanticRequirement",
    "RegexRequirement",
    "ChatFormatter",
    "TemplateFormatter",
    "ModelOptions",
    "GuardianCheck",
    "MObject",
}


def extract_python_blocks(filepath: Path) -> list[tuple[int, str]]:
    """Return (start_line, code_text) for each Python fenced block."""
    blocks: list[tuple[int, str]] = []
    text = filepath.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    in_block = False
    block_start = 0
    block_lines: list[str] = []

    for i, line in enumerate(lines):
        if not in_block:
            if FENCE_OPEN_RE.match(line.strip()):
                in_block = True
                block_start = i + 2  # 1-indexed, next line
                block_lines = []
        else:
            if FENCE_CLOSE_RE.match(line.strip()):
                in_block = False
                blocks.append((block_start, "\n".join(block_lines)))
            else:
                block_lines.append(line)
    return blocks


# SyntaxError messages that indicate a code *fragment* rather than a
# genuinely broken snippet.  These are downgraded to warnings.
_FRAGMENT_PATTERNS = (
    "'await' outside function",
    "'await' outside async function",
    "asynchronous comprehension outside of an asynchronous function",
    "unexpected indent",
    "'yield' outside function",
)


def check_syntax(code: str, filename: str) -> tuple[str | None, bool]:
    """Try to compile; return (error_message, is_fragment).

    is_fragment is True when the error is due to the snippet being an
    incomplete fragment (e.g. bare ``await`` or leading indentation)
    rather than genuinely broken syntax.
    """
    try:
        compile(code, filename, "exec")
        return None, False
    except SyntaxError as exc:
        detail = f"line {exc.lineno}: {exc.msg}" if exc.lineno else str(exc)
        msg = exc.msg or ""
        is_frag = any(pat in msg for pat in _FRAGMENT_PATTERNS)
        return f"SyntaxError: {detail}", is_frag


def extract_imports(code: str) -> list[tuple[str, int | None]]:
    """Return (module_name, lineno) for each import statement."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    imports: list[tuple[str, int | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append((node.module, node.lineno))
    return imports


def module_importable(module_name: str) -> bool:
    """Check if module_name can be resolved without actually importing."""
    top = module_name.split(".")[0]
    return importlib.util.find_spec(top) is not None


def classify_module(name: str) -> str:
    if name.startswith("mellea"):
        return "mellea"
    if name.split(".")[0] in sys.stdlib_module_names:
        return "stdlib"
    return "third-party"


def check_missing_mellea_imports(code: str) -> list[str]:
    """Flag mellea names used but never imported."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.asname or alias.name.split(".")[-1])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported.add(alias.asname or alias.name)
            if node.module:
                imported.add(node.module.split(".")[0])

    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)

    return sorted(used & MELLEA_NAMES - imported)


# Minimum lines for a code block to be considered for duplicate detection.
# Short snippets (imports, one-liners) are expected to repeat.
_DUPE_MIN_LINES = 4


def _code_hash(code: str) -> str:
    """Normalize and hash a code block for duplicate detection."""
    # Strip trailing whitespace per line, collapse blank lines
    normalized = "\n".join(line.rstrip() for line in code.splitlines()).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def run_code_checks(
    doc_files: list[Path], verbose: bool
) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) from code block checking."""
    errors: list[str] = []
    warnings: list[str] = []
    total_blocks = 0

    # Duplicate tracking: hash -> list of "file:line" labels
    seen_blocks: dict[str, list[str]] = {}

    for filepath in doc_files:
        rel = filepath.relative_to(DOCS_ROOT)
        blocks = extract_python_blocks(filepath)

        for start_line, code in blocks:
            total_blocks += 1
            label = f"{rel}:{start_line}"

            if verbose:
                preview = code.split("\n", 1)[0][:60]
                print(f"  [{total_blocks:3d}] {label}  {preview!r}")

            # Track duplicates (only for non-trivial blocks)
            line_count = len([ln for ln in code.splitlines() if ln.strip()])
            if line_count >= _DUPE_MIN_LINES:
                h = _code_hash(code)
                seen_blocks.setdefault(h, []).append(label)

            # 1. Syntax
            err, is_fragment = check_syntax(code, str(rel))
            if err:
                if is_fragment:
                    warnings.append(f"  {label} — {err} (fragment)")
                else:
                    errors.append(f"  {label} — {err}")
                continue

            # 2. Imports
            for mod_name, mod_line in extract_imports(code):
                cls = classify_module(mod_name)
                loc = f"{label}+{mod_line}" if mod_line else label
                if cls == "mellea" and not module_importable(mod_name):
                    warnings.append(
                        f"  {loc}: import {mod_name} — mellea submodule not found"
                    )
                elif cls == "third-party" and not module_importable(mod_name):
                    warnings.append(
                        f"  {loc}: import {mod_name}"
                        f" — third-party not installed (add install note?)"
                    )

            # 3. Missing mellea imports
            missing = check_missing_mellea_imports(code)
            if missing:
                warnings.append(
                    f"  {label}: uses {', '.join(missing)} without importing"
                )

    # 4. Duplicate code blocks (across different files)
    for h, locations in seen_blocks.items():
        if len(locations) < 2:
            continue
        # Only flag if the duplicates span different files
        files = {loc.rsplit(":", 1)[0] for loc in locations}
        if len(files) >= 2:
            locs = ", ".join(locations)
            warnings.append(
                f"  duplicate code block in {len(locations)} places: {locs}"
            )

    print(f"\nCode: scanned {len(doc_files)} files, {total_blocks} Python block(s)")

    return errors, warnings


# ===================================================================
# SHELL CHECKING
# ===================================================================

BASH_FENCE_RE = re.compile(r"^```(?:bash|shell|sh|zsh)\b.*$")

# Matches pip/uv install with [extras] — e.g. pip install mellea[litellm]
# Captures the full token including any surrounding quotes so we can check.
INSTALL_EXTRAS_RE = re.compile(
    r"""(?:pip|uv)\s+(?:install|pip\s+install)\s+  # pip install / uv install / uv pip install
        (?:(?:-\S+\s+)*)                            # optional flags like -U
        (['"]?)                                      # optional opening quote
        (\S+\[[^\]]+\])                              # package[extras]
        (['"]?)                                      # optional closing quote
    """,
    re.VERBOSE,
)


def extract_bash_blocks(filepath: Path) -> list[tuple[int, str]]:
    """Return (start_line, code_text) for each bash fenced block."""
    blocks: list[tuple[int, str]] = []
    text = filepath.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    in_block = False
    block_start = 0
    block_lines: list[str] = []

    for i, line in enumerate(lines):
        if not in_block:
            if BASH_FENCE_RE.match(line.strip()):
                in_block = True
                block_start = i + 2
                block_lines = []
        else:
            if FENCE_CLOSE_RE.match(line.strip()):
                in_block = False
                blocks.append((block_start, "\n".join(block_lines)))
            else:
                block_lines.append(line)
    return blocks


def run_shell_checks(
    doc_files: list[Path], verbose: bool
) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) from shell block checking."""
    errors: list[str] = []
    warnings: list[str] = []
    total_blocks = 0

    for filepath in doc_files:
        rel = filepath.relative_to(DOCS_ROOT)

        # Check bash code blocks
        blocks = extract_bash_blocks(filepath)
        for start_line, code in blocks:
            total_blocks += 1
            for i, line in enumerate(code.splitlines()):
                m = INSTALL_EXTRAS_RE.search(line)
                if m:
                    open_q, pkg, close_q = m.group(1), m.group(2), m.group(3)
                    quoted = open_q and close_q  # has matching quotes
                    if not quoted:
                        lineno = start_line + i
                        errors.append(
                            f"  {rel}:{lineno} — unquoted extras: {pkg}"
                            f'  [breaks in zsh — use "{pkg}"]'
                        )
                    elif verbose:
                        print(f"  [ok]   {rel}:{start_line + i} — quoted: {pkg}")

        # Also check inline code in markdown text for install commands
        text = filepath.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            # Look for inline backtick commands: `pip install foo[bar]`
            for tick_m in re.finditer(r"`([^`]+)`", line):
                content = tick_m.group(1)
                extras_m = INSTALL_EXTRAS_RE.search(content)
                if extras_m:
                    open_q = extras_m.group(1)
                    pkg = extras_m.group(2)
                    close_q = extras_m.group(3)
                    quoted = open_q and close_q
                    if not quoted:
                        errors.append(
                            f"  {rel}:{lineno} — unquoted extras in"
                            f" inline code: {pkg}"
                            f'  [breaks in zsh — use "{pkg}"]'
                        )

    print(f"\nShell: scanned {len(doc_files)} files, {total_blocks} bash block(s)")

    return errors, warnings


# ===================================================================
# Main
# ===================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Mellea docs: links and code snippets"
    )
    all_checks = ["links", "code", "shell"]
    parser.add_argument(
        "checks",
        nargs="*",
        default=all_checks,
        metavar="CHECK",
        help="Which checks to run: links, code, shell (default: all)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show every item checked"
    )
    parser.add_argument(
        "--skip-external", action="store_true", help="Skip HTTP checks on external URLs"
    )
    args = parser.parse_args()

    if not DOCS_ROOT.is_dir():
        print(f"ERROR: docs root not found at {DOCS_ROOT}", file=sys.stderr)
        return 2

    doc_files = collect_doc_files()
    all_errors: list[str] = []
    all_warnings: list[str] = []

    if "links" in args.checks:
        print("=" * 60)
        print("LINK CHECKS")
        print("=" * 60)
        errs, warns = run_link_checks(
            doc_files, args.verbose, check_external=not args.skip_external
        )
        all_errors.extend(errs)
        all_warnings.extend(warns)

    if "code" in args.checks:
        print("\n" + "=" * 60)
        print("CODE CHECKS")
        print("=" * 60)
        errs, warns = run_code_checks(doc_files, args.verbose)
        all_errors.extend(errs)
        all_warnings.extend(warns)

    if "shell" in args.checks:
        print("\n" + "=" * 60)
        print("SHELL CHECKS")
        print("=" * 60)
        errs, warns = run_shell_checks(doc_files, args.verbose)
        all_errors.extend(errs)
        all_warnings.extend(warns)

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if all_errors:
        print(f"\n{len(all_errors)} ERROR(s):\n")
        for e in all_errors:
            print(e)

    if all_warnings:
        print(f"\n{len(all_warnings)} WARNING(s):\n")
        for w in all_warnings:
            print(w)

    if not all_errors and not all_warnings:
        print("\nAll checks passed.")
    elif not all_errors:
        print(f"\nNo errors. {len(all_warnings)} warning(s) to review.")

    return 1 if all_errors else 0


if __name__ == "__main__":
    sys.exit(main())
