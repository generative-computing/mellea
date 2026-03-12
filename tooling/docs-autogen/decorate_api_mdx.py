#!/usr/bin/env python3
"""decorate_api_mdx.py — Step 2 of the API docs pipeline.

Applies a series of decoration passes to the raw MDX files produced by
generate-ast.py (step 1).  Run via build.py or directly:

    uv run python tooling/docs-autogen/decorate_api_mdx.py \\
        --api-dir docs/docs/api \\
        --version 0.5.0 \\
        --source-dir mellea

Decoration passes (applied in order per file):
1. fix_source_links      — correct GitHub blob URLs to versioned tags
2. inject_preamble       — add per-module introductory text
3. inject_sidebar_fix    — insert SidebarFix Mintlify component
4. escape_mdx_syntax     — escape {{ }} in code blocks so MDX doesn't treat them as JSX
5. add_cross_references  — linkify type names to their definition pages
6. decorate_mdx_body     — add CLASS/FUNC pills and visual dividers to headings

WARNING: Not idempotent.  Each pass appends/wraps without checking for prior runs.
Always run the full pipeline from scratch via `uv run poe apidocs`.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# =========================
# SidebarFix injection
# =========================

FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.S)

# Canonical Mintlify docs-root absolute import (this is what worked for you)
SIDEBAR_IMPORT_LINE = 'import { SidebarFix } from "/snippets/SidebarFix.mdx";'
SIDEBAR_RENDER_LINE = "<SidebarFix />"

IMPORT_RE = re.compile(
    r'(?m)^\s*import\s+\{\s*SidebarFix\s*\}\s+from\s+["\']\/snippets\/SidebarFix\.mdx["\']\s*;?\s*$'
)
RENDER_RE = re.compile(
    r"(?m)^\s*<\s*SidebarFix\s*\/\s*>\s*$|^\s*<\s*SidebarFix\s*>\s*<\/\s*SidebarFix\s*>\s*$"
)

SIDEBAR_BLOCK = SIDEBAR_IMPORT_LINE + "\n\n" + SIDEBAR_RENDER_LINE + "\n\n"


def inject_sidebar_fix(mdx_text: str) -> str:
    """Insert SidebarFix import + render right after frontmatter (or at top if none)."""
    has_import = bool(IMPORT_RE.search(mdx_text))
    has_render = bool(RENDER_RE.search(mdx_text))
    if has_import and has_render:
        return mdx_text

    # Normalize: remove any partial remnants and add canonical block
    mdx_text = IMPORT_RE.sub("", mdx_text)
    mdx_text = RENDER_RE.sub("", mdx_text)

    m = FRONTMATTER_RE.match(mdx_text)
    if m:
        insert_at = m.end()
        return (
            mdx_text[:insert_at]
            + "\n"
            + SIDEBAR_BLOCK
            + mdx_text[insert_at:].lstrip("\n")
        )

    return SIDEBAR_BLOCK + mdx_text.lstrip("\n")


# =========================
# GitHub source link fixing
# =========================


def fix_source_links(content: str, version: str) -> str:
    """Fix GitHub source links to use a versioned tag instead of blob/main.

    Handles both output formats from mdxify:
    - HTML anchor: <a href="https://github.com/OWNER/REPO/blob/main/PATH">
    - Markdown link: [View source](https://github.com/OWNER/REPO/blob/main/PATH)

    Args:
        content: MDX file content
        version: Package version (e.g., "0.5.0")

    Returns:
        Content with blob/main replaced by blob/v{version}
    """
    # HTML href format (used by current mdxify output):
    # <a href="https://github.com/OWNER/REPO/blob/BRANCH/PATH" ...>
    html_pattern = r'href="(https://github\.com/([^/]+)/([^/]+)/blob/)[^/]+/([^"]+)"'

    def replace_html(match):
        base = match.group(1)  # https://github.com/OWNER/REPO/blob/
        path = match.group(4)
        return f'href="{base}v{version}/{path}"'

    content = re.sub(html_pattern, replace_html, content)

    # Markdown link format (kept for backwards compatibility):
    # [View source](https://github.com/OWNER/REPO/blob/BRANCH/PATH)
    md_pattern = (
        r"\[View source\]\((https://github\.com/([^/]+)/([^/]+)/blob/)[^/]+/([^)]+)\)"
    )

    def replace_md(match):
        base = match.group(1)
        path = match.group(4)
        return f"[View source]({base}v{version}/{path})"

    content = re.sub(md_pattern, replace_md, content)

    return content


# =========================
# MDX escaping
# =========================


_DOCTEST_PROMPT_RE = re.compile(r"^\s*>>>\s")


def wrap_doctest_blocks(content: str) -> str:
    """Wrap bare doctest (>>>) blocks in ```python code fences.

    mdxify renders Python doctest examples from docstrings as raw MDX prose.
    Lines starting with >>> are parsed by MDX as nested Markdown blockquotes,
    and subsequent output lines without a > prefix trigger a lazy-line parse
    error in MDX v2.

    This pass wraps each contiguous doctest block in a ```python...``` fence.
    Must be called before escape_mdx_syntax() so the fence escaping applies.

    Args:
        content: MDX file content

    Returns:
        Content with doctest blocks wrapped in fenced code blocks
    """
    lines = content.splitlines(keepends=True)
    result: list[str] = []
    in_fence = False
    i = 0

    while i < len(lines):
        raw = lines[i]
        stripped = raw.rstrip("\n").rstrip("\r")

        if stripped.lstrip().startswith("```"):
            in_fence = not in_fence
            result.append(raw)
            i += 1
            continue

        if in_fence:
            result.append(raw)
            i += 1
            continue

        if _DOCTEST_PROMPT_RE.match(stripped):
            # Collect all consecutive non-blank lines as one doctest block
            block: list[str] = []
            while i < len(lines):
                raw_l = lines[i]
                sl = raw_l.rstrip("\n").rstrip("\r")
                if not sl.strip():
                    break
                block.append(sl.lstrip())
                i += 1

            if block:
                result.append("```python\n")
                for bl in block:
                    result.append(bl + "\n")
                result.append("```\n")
            continue

        result.append(raw)
        i += 1

    return "".join(result)


def escape_mdx_syntax(content: str) -> str:
    """Escape MDX-sensitive characters in code blocks and prose.

    MDX interprets curly braces as JSX expressions, which breaks
    Python dict literals and JSON in two contexts:

    1. Inside fenced code blocks (```): escaped as {{ / }} so that the
       MDX compiler does not strip them.
    2. In prose outside code blocks: escaped as \\{ / \\} (CommonMark
       backslash escapes) so acorn never sees a bare { as an expression.
       Import lines and JSX/HTML tag lines are left untouched.

    Args:
        content: MDX file content

    Returns:
        Content with escaped MDX syntax
    """
    lines = content.splitlines(keepends=True)
    result = []
    in_code_block = False
    code_fence_pattern = re.compile(r"^```")

    for line in lines:
        # Track code block boundaries
        if code_fence_pattern.match(line):
            in_code_block = not in_code_block
            result.append(line)
            continue

        if in_code_block:
            # Inside fenced code blocks: escape { and } as {{ / }} so that the
            # MDX compiler does not strip them as JSX expression delimiters.
            result.append(line.replace("{", "{{").replace("}", "}}"))
        else:
            # Outside fenced code blocks: escape bare { and } with backslash
            # so MDX does not try to evaluate them as JavaScript expressions.
            # Skip import lines (e.g. import { SidebarFix } from "...") and
            # JSX/HTML tag lines (e.g. <div ...>) which require real brace syntax.
            if "{" in line or "}" in line:
                s = line.lstrip()
                if not s.startswith("<") and not s.startswith("import "):
                    line = line.replace("{", r"\{").replace("}", r"\}")
            result.append(line)

    return "".join(result)


# =========================
# Preamble injection
# =========================


def inject_preamble(
    content: str, module_path: str, docstring_cache: dict[str, str] | None = None
) -> str:
    """Inject preamble text after frontmatter.

    Uses the module docstring from source as the preamble, walking up the
    dotted path until a docstring is found (e.g. cli.alora.commands falls back
    to cli.alora). If nothing is found the file is returned unchanged.

    Args:
        content: MDX file content
        module_path: Module path derived from file path (e.g. "mellea.core.base")
        docstring_cache: Optional module-path→docstring dict from
            build_module_docstring_cache(). When provided, module docstrings are
            used as preambles.

    Returns:
        Content with preamble injected
    """
    # Module docstring from source — walk up the dotted path so that
    # cli.alora.commands falls back to cli.alora when the leaf has no docstring.
    preamble_text: str | None = None
    if docstring_cache:
        candidate = module_path
        while candidate:
            doc = docstring_cache.get(candidate)
            if doc:
                preamble_text = doc + "\n\n"
                break
            candidate = candidate.rsplit(".", 1)[0] if "." in candidate else ""

    if not preamble_text:
        return content

    # Find end of frontmatter
    if not content.startswith("---\n"):
        return content

    frontmatter_end = content.find("\n---\n", 4)
    if frontmatter_end == -1:
        return content

    # Inject preamble after frontmatter
    before = content[: frontmatter_end + 5]  # Include closing ---\n
    after = content[frontmatter_end + 5 :]

    return f"{before}\n{preamble_text}{after}"


# =========================
# Heading decoration (pills/dividers)
# =========================

CLASS_SPAN = (
    '<span className="ml-2 inline-flex items-center rounded-full '
    "px-2 py-1 text-[0.7rem] font-bold tracking-wide "
    'bg-[#4ADE8033]/20 text-[#15803D]">CLASS</span>'
)

FUNC_SPAN = (
    '<span className="ml-2 inline-flex items-center rounded-full '
    "px-2 py-1 text-[0.7rem] font-bold tracking-wide "
    'bg-[#3064E3]/20 text-[#1D4ED8]">FUNC</span>'
)

SPAN_RE = re.compile(r'\s*<span className="[^"]*rounded-full[^"]*">.*?</span>\s*')
LABEL_RE = re.compile(r"^\[(class|func|Class|funct)\]\s+")
# DIVIDER_LINE = '---'
DIVIDER_LINE = '<div className="w-full h-px bg-gray-200 dark:bg-gray-700 my-4" />'
SPACER_BLOCK = '<div className="h-8" />'


def pick_kind(name: str, level: int, current_section: str | None) -> str | None:
    if level >= 4:
        return "func"
    if level == 3:
        if current_section == "functions":
            return "func"
        if current_section == "classes":
            return "class"
        return "class" if name and name[0].isupper() else "func"
    return None


def label_heading(line: str, current_section: str | None) -> str:
    head_match = re.match(r"^(#{2,6})\s+(.*)$", line)
    if not head_match:
        return line

    hashes = head_match.group(1)
    body = LABEL_RE.sub("", head_match.group(2)).strip()
    body = SPAN_RE.sub(" ", body).strip()

    # Only decorate headings that look like: ## `Something`
    m = re.match(r"^`([^`]+)`(.*)$", body)
    if not m:
        return line

    name = m.group(1)
    rest = m.group(2).rstrip()
    level = len(hashes)

    kind = pick_kind(name, level, current_section)
    if not kind:
        return f"{hashes} `{name}`{rest}"

    span = CLASS_SPAN if kind == "class" else FUNC_SPAN

    if rest:
        return f"{hashes} {span} `{name}`{rest}"
    return f"{hashes} {span} `{name}`"


def line_has_pill(line: str) -> bool:
    return line.strip().startswith("#") and (
        "bg-[#3064E3]/20" in line or "bg-[#4ADE8033]/20" in line
    )


def last_non_empty_is_divider(lines: list[str]) -> bool:
    for line in reversed(lines):
        if line.strip() != "":
            return line.strip() == DIVIDER_LINE
    return False


def decorate_mdx_body(full_text: str) -> str:
    """Applies the pill/divider logic with tight HR placement, WITHOUT triggering
    Setext headings.
    """
    lines = full_text.splitlines()
    out: list[str] = []

    current_section: str | None = None  # None | "classes" | "functions"
    in_item: bool = False  # inside a class/function block?

    # Detect an "item heading" inside a section (class/function entry)
    item_heading_re = re.compile(r"^(#{3,6})\s+.*`[^`]+`")

    def last_non_empty_is_divider_local() -> bool:
        for ln in reversed(out):
            if ln.strip() != "":
                return ln.strip() == DIVIDER_LINE
        return False

    def append_divider():
        # Ensure previous line is blank to prevent Setext headings.
        if out and out[-1].strip() != "":
            out.append("")
        if not last_non_empty_is_divider_local():
            out.append(DIVIDER_LINE)
        out.append("")

    def close_item_if_open():
        nonlocal in_item
        if in_item:
            append_divider()
            in_item = False

    for line in lines:
        stripped = line.strip()

        # Skip <br />
        if stripped == "<br />":
            continue

        # --- NEW LOGIC START ---
        # Inject spacer before "**Methods:**"
        if stripped == "**Methods:**":
            out.append("")  # Ensure we are on a new line
            out.append(SPACER_BLOCK)  # Insert the height div
            out.append(line)
            continue
        # --- NEW LOGIC END ---

        # Section headers
        if stripped == "## Classes":
            close_item_if_open()
            current_section = "classes"
            out.append(line)
            append_divider()
            continue

        if stripped == "## Functions":
            close_item_if_open()
            current_section = "functions"
            out.append(line)
            append_divider()
            continue

        # Any other H2 header exits the section
        if stripped.startswith("## ") and stripped not in (
            "## Classes",
            "## Functions",
        ):
            close_item_if_open()
            current_section = None
            in_item = False
            out.append(line)
            continue

        # Decorate headings (but do not add HR here)
        labeled = label_heading(line, current_section)
        stripped_labeled = labeled.strip()

        # Within Classes/Functions: a new item heading closes the prior item
        if current_section in ("classes", "functions") and item_heading_re.match(
            stripped_labeled
        ):
            close_item_if_open()
            in_item = True
            out.append(labeled)
            continue

        out.append(labeled)

    close_item_if_open()

    while out and out[-1].strip() == "":
        out.pop()

    return "\n".join(out) + "\n"


# =========================
# Cross-reference functions
# =========================


def extract_type_references(content: str) -> set[str]:
    """Extract type references from docstrings and signatures.

    Looks for:
    - Type annotations: Backend, Optional[Backend], List[str]
    - Docstring references: See `Backend` for details
    - Return types: Returns Backend instance

    Args:
        content: MDX file content

    Returns:
        Set of type names referenced (e.g., {"Backend", "Session"})
    """
    refs = set()

    # Pattern 1: Type annotations in code blocks
    # Example: def foo(backend: Backend) -> Session:
    type_pattern = r":\s*([A-Z][a-zA-Z0-9_]*)"
    refs.update(re.findall(type_pattern, content))

    # Pattern 2: Backtick references in text
    # Example: See `Backend` for details
    backtick_pattern = r"`([A-Z][a-zA-Z0-9_]*)`"
    refs.update(re.findall(backtick_pattern, content))

    # Pattern 3: Generic types
    # Example: Optional[Backend], List[Session]
    generic_pattern = r"\[([A-Z][a-zA-Z0-9_]*)\]"
    refs.update(re.findall(generic_pattern, content))

    return refs


def build_module_docstring_cache(source_dir: Path) -> dict[str, str]:
    """Build a module-path→docstring lookup table from the source packages.

    Loads mellea and cli (if present) with Griffe and returns a dict mapping
    each public module's dotted path to the first paragraph of its docstring,
    e.g. {"cli.alora": "alora command group for training and uploading adapters."}.

    This is used by inject_preamble() to source per-module descriptions directly
    from the code rather than maintaining a hardcoded list — new modules are
    covered automatically.

    Args:
        source_dir: The mellea/ source directory (e.g. <repo-root>/mellea)

    Returns:
        Dict mapping module path → docstring text, or empty dict on failure
    """
    try:
        import griffe
    except ImportError:
        return {}

    cache: dict[str, str] = {}
    search_root = source_dir.parent

    def _walk(module: object, path: str) -> None:
        if any(part.startswith("_") for part in path.split(".")):
            return
        doc = getattr(module, "docstring", None)
        if doc:
            text = doc.value.strip()
            # Take only the first paragraph
            first_para = text.split("\n\n")[0].replace("\n", " ").strip()
            if first_para:
                cache[path] = first_para
        if hasattr(module, "modules"):
            for name, sub in module.modules.items():
                if not name.startswith("_"):
                    _walk(sub, f"{path}.{name}")

    for pkg_name in ("mellea", "cli"):
        pkg_dir = search_root / pkg_name
        if not pkg_dir.exists():
            continue
        try:
            pkg = griffe.load(pkg_name, search_paths=[str(search_root)])
            _walk(pkg, pkg_name)
        except Exception:
            pass

    return cache


def build_symbol_cache(source_dir: Path) -> dict[str, str]:
    """Build a symbol→module-path lookup table from the mellea package.

    Loads the package once with Griffe and returns a dict mapping each
    exported symbol name to its canonical module path, e.g.:
        {"Backend": "mellea.core.backend", "Session": "mellea.stdlib.session", ...}

    This is intentionally called once per process (in main()) and the result
    is passed through to add_cross_references(), avoiding the O(files x symbols)
    repeated Griffe loads that made the old resolve_symbol_path() so expensive.

    Args:
        source_dir: The mellea/ source directory (e.g. <repo-root>/mellea)

    Returns:
        Dict mapping symbol name → module path, or empty dict on failure
    """
    try:
        import griffe
    except ImportError:
        return {}

    try:
        package = griffe.load("mellea", search_paths=[str(source_dir.parent)])
    except Exception:
        return {}

    cache: dict[str, str] = {}
    for _mod_key, module in package.modules.items():
        for symbol_name, member in module.members.items():
            if symbol_name in cache:
                continue  # Keep first (shallowest) match
            try:
                canonical = member.canonical_path
            except Exception:
                # Griffe raises AliasResolutionError for re-exports that point
                # outside the package (e.g. `from dataclasses import dataclass`).
                # Skip these — they're not mellea symbols.
                continue
            parts = canonical.split(".")
            if len(parts) > 1:
                cache[symbol_name] = ".".join(parts[:-1])
    return cache


def add_cross_references(
    content: str,
    module_path: str,
    source_dir: Path,
    symbol_cache: dict[str, str] | None = None,
) -> str:
    """Add cross-reference links to type mentions.

    Args:
        content: MDX file content
        module_path: Current module path (e.g., "mellea.core.base")
        source_dir: Project source directory (used only if symbol_cache is None)
        symbol_cache: Pre-built symbol→module-path mapping from build_symbol_cache().
            Pass this in from main() to avoid reloading Griffe for every file.

    Returns:
        Content with cross-reference links added
    """
    # Import mintlify_anchor from test file
    import sys

    test_anchors_path = Path(__file__).parent / "test_mintlify_anchors.py"
    if test_anchors_path.exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "test_mintlify_anchors", test_anchors_path
        )
        if spec and spec.loader:
            test_anchors = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test_anchors)
            mintlify_anchor = test_anchors.mintlify_anchor
        else:
            # Fallback if import fails
            def mintlify_anchor(heading: str) -> str:
                anchor = heading.lower().replace(" ", "-")
                anchor = re.sub(r"[^a-z0-9-]", "", anchor)
                anchor = re.sub(r"-+", "-", anchor)
                return anchor.strip("-")
    else:
        # Fallback if file doesn't exist
        def mintlify_anchor(heading: str) -> str:
            anchor = heading.lower().replace(" ", "-")
            anchor = re.sub(r"[^a-z0-9-]", "", anchor)
            anchor = re.sub(r"-+", "-", anchor)
            return anchor.strip("-")

    # Use provided cache or build a one-off one (slow path, kept for backwards compat)
    if symbol_cache is None:
        symbol_cache = build_symbol_cache(source_dir)

    # Extract type references
    refs = extract_type_references(content)

    # Resolve each reference to a module path using the cache
    resolved = {}
    for ref in refs:
        target_module = symbol_cache.get(ref)
        if target_module and target_module != module_path:
            resolved[ref] = target_module

    # Replace backtick references with links
    # Example: `Backend` -> [`Backend`](../core/backend#class-backend)
    def replace_ref(match):
        symbol = match.group(1)
        if symbol in resolved:
            target_module = resolved[symbol]
            # Calculate relative path
            # module_path is like "mellea.core.formatter" (the current file)
            # target_module is like "mellea.core.base" (the target file)
            current_parts = module_path.split(".")
            target_parts = target_module.split(".")

            # Find common prefix
            common = 0
            for i in range(min(len(current_parts), len(target_parts))):
                if current_parts[i] == target_parts[i]:
                    common += 1
                else:
                    break

            # Build relative path
            # We need to go up from the current file's directory, not from the file itself
            # current_parts[-1] is the file name, so we exclude it
            up_levels = (
                len(current_parts) - common - 1
            )  # -1 because we're in a file, not a dir

            # If we're in the same directory (e.g., both in mellea.core), up_levels will be 0
            # and we just need the target file name
            if up_levels == 0:
                rel_path = target_parts[-1]  # Just the filename in same directory
            else:
                rel_path = "../" * up_levels + "/".join(target_parts[common:])

            # Generate anchor
            anchor = mintlify_anchor(f"class {symbol}")

            return f"[`{symbol}`]({rel_path}#{anchor})"
        return match.group(0)

    # First, fix existing markdown links that may have wrong paths
    # Pattern: [`Symbol`](path#anchor)
    def fix_existing_link(match):
        symbol = match.group(1)
        # old_path = match.group(2)  # Not needed, we recalculate
        anchor = match.group(3)

        if symbol in resolved:
            target_module = resolved[symbol]
            current_parts = module_path.split(".")
            target_parts = target_module.split(".")

            # Find common prefix
            common = 0
            for i in range(min(len(current_parts), len(target_parts))):
                if current_parts[i] == target_parts[i]:
                    common += 1
                else:
                    break

            # Build relative path
            up_levels = len(current_parts) - common - 1

            if up_levels == 0:
                rel_path = target_parts[-1]
            else:
                rel_path = "../" * up_levels + "/".join(target_parts[common:])

            # Keep the existing anchor
            return f"[`{symbol}`]({rel_path}#{anchor})"
        return match.group(0)

    # Fix existing links first
    existing_link_pattern = r"\[`([A-Z][a-zA-Z0-9_]*)`\]\(([^)#]+)#([^)]+)\)"
    content = re.sub(existing_link_pattern, fix_existing_link, content)

    # Then add links to plain backtick references
    pattern = r"`([A-Z][a-zA-Z0-9_]*)`"
    content = re.sub(pattern, replace_ref, content)

    return content


# =========================
# Path resolution + processing
# =========================


def resolve_api_dir(docs_root: Path | None, api_dir: Path | None) -> Path:
    if api_dir is not None:
        return api_dir
    if docs_root is not None:
        return docs_root / "api"

    cwd = Path.cwd()
    candidates = [cwd / "docs" / "api", cwd / "docs" / "docs" / "api"]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def process_mdx_file(
    path: Path,
    version: str,
    api_dir: Path | None = None,
    source_dir: Path | None = None,
    symbol_cache: dict[str, str] | None = None,
    docstring_cache: dict[str, str] | None = None,
) -> bool:
    """Process a single MDX file.

    Args:
        path: Path to MDX file
        version: Package version for GitHub links
        api_dir: API directory (for extracting module path)
        source_dir: Project source directory (for cross-reference resolution)
        symbol_cache: Pre-built symbol→module-path cache from build_symbol_cache().
            When provided, cross-references are resolved without re-loading Griffe.
        docstring_cache: Pre-built module-path→docstring cache from
            build_module_docstring_cache(). When provided, module docstrings are
            used as preambles for modules not covered by hardcoded overrides.

    Returns:
        True if file was modified, False otherwise
    """
    original = path.read_text(encoding="utf-8")

    # Extract module path from filepath
    # e.g., docs/docs/api/mellea/core/base.mdx -> mellea.core.base
    if api_dir:
        try:
            rel_path = path.relative_to(api_dir)
            module_path = str(rel_path.with_suffix("")).replace("/", ".")
        except ValueError:
            # If path is not relative to api_dir, use filename
            module_path = path.stem
    else:
        module_path = path.stem

    # Step 1: Fix GitHub source links
    text = fix_source_links(original, version)

    # Step 2: Inject preamble
    text = inject_preamble(text, module_path, docstring_cache)

    # Step 3: inject SidebarFix
    text = inject_sidebar_fix(text)

    # Step 3.5: Wrap bare doctest (>>>) blocks in fenced code blocks.
    # Must run before escape_mdx_syntax so the new fences are processed.
    text = wrap_doctest_blocks(text)

    # Step 4: Escape MDX syntax in code blocks and bare prose braces
    text = escape_mdx_syntax(text)

    # Step 5: Add cross-references (if source_dir provided)
    if source_dir:
        text = add_cross_references(text, module_path, source_dir, symbol_cache)

    # Step 6: decorate headings/dividers
    text = decorate_mdx_body(text)

    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Decorate API MDX files")
    parser.add_argument(
        "--docs-root", type=Path, default=None, help="Mintlify docs root directory"
    )
    parser.add_argument(
        "--api-dir", type=Path, default=None, help="API directory containing MDX files"
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Package version for GitHub links (e.g., 0.5.0)",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Project source directory for cross-reference resolution",
    )
    args = parser.parse_args()

    api_dir = resolve_api_dir(args.docs_root, args.api_dir)
    if not api_dir.exists():
        raise SystemExit(
            f"❌ API directory not found: {api_dir}\n"
            "Try:\n"
            "  uv run python tooling/docs-autogen/decorate_api_mdx.py --docs-root docs/docs --version 0.5.0\n"
            "or\n"
            "  uv run python tooling/docs-autogen/decorate_api_mdx.py --api-dir docs/docs/api --version 0.5.0\n"
        )

    # Resolve source directory (default to mellea/ in current directory)
    source_dir = args.source_dir
    if source_dir is None:
        source_dir = Path.cwd() / "mellea"
        if not source_dir.exists():
            source_dir = None  # Disable cross-references if source not found

    mdx_files = list(api_dir.rglob("*.mdx"))
    if not mdx_files:
        print(f"⚠️ No .mdx files found under: {api_dir}")
        return

    # Build the Griffe caches once for all files (key perf fix:
    # the old code called griffe.load() once per symbol per file — O(files x symbols)).
    symbol_cache: dict[str, str] | None = None
    docstring_cache: dict[str, str] | None = None
    if source_dir:
        print(f"Building symbol cache from {source_dir}...")
        symbol_cache = build_symbol_cache(source_dir)
        print(f"  Cached {len(symbol_cache)} symbols.")
        docstring_cache = build_module_docstring_cache(source_dir)
        print(f"  Cached {len(docstring_cache)} module docstrings.")

    changed = 0
    for f in mdx_files:
        if process_mdx_file(
            f, args.version, api_dir, source_dir, symbol_cache, docstring_cache
        ):
            changed += 1

    print(
        f"✅ Done. Processed {len(mdx_files)} MDX files under {api_dir}. Updated {changed}."
    )


if __name__ == "__main__":
    main()
