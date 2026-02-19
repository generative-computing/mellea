#!/usr/bin/env python3
"""
decorate_api_mdx_merged.py

Merges the two behaviors you were running back-to-back:

1) decorate_api_mdx.py behavior:
   - Adds CLASS/FUNC pills to headings
   - Inserts divider lines
   - Skips <br />

2) v3 behavior:
   - Injects SidebarFix import + render so Mintlify sidebar badges/icons work:
       import { SidebarFix } from "/snippets/SidebarFix.mdx";
       <SidebarFix />

Usage examples:

# (Recommended) Point at the Mintlify docs root (the folder that contains api/ and snippets/)
python3 decorate_api_mdx_merged.py --docs-root /path/to/docs/docs

# Or point directly at api directory
python3 decorate_api_mdx_merged.py --api-dir /path/to/docs/docs/api
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
    r'(?m)^\s*<\s*SidebarFix\s*\/\s*>\s*$|^\s*<\s*SidebarFix\s*>\s*<\/\s*SidebarFix\s*>\s*$'
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
        return mdx_text[:insert_at] + "\n" + SIDEBAR_BLOCK + mdx_text[insert_at:].lstrip("\n")

    return SIDEBAR_BLOCK + mdx_text.lstrip("\n")


# =========================
# Heading decoration (pills/dividers)
# =========================

CLASS_SPAN = (
    '<span className="ml-2 inline-flex items-center rounded-full '
    'px-2 py-1 text-[0.7rem] font-bold tracking-wide '
    'bg-[#4ADE8033]/20 text-[#15803D]">CLASS</span>'
)

FUNC_SPAN = (
    '<span className="ml-2 inline-flex items-center rounded-full '
    'px-2 py-1 text-[0.7rem] font-bold tracking-wide '
    'bg-[#3064E3]/20 text-[#1D4ED8]">FUNC</span>'
)

SPAN_RE = re.compile(r'\s*<span className="[^"]*rounded-full[^"]*">.*?</span>\s*')
LABEL_RE = re.compile(r'^\[(class|func|Class|funct)\]\s+')
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
    head_match = re.match(r'^(#{2,6})\s+(.*)$', line)
    if not head_match:
        return line

    hashes = head_match.group(1)
    body = LABEL_RE.sub("", head_match.group(2)).strip()
    body = SPAN_RE.sub(" ", body).strip()

    # Only decorate headings that look like: ## `Something`
    m = re.match(r'^`([^`]+)`(.*)$', body)
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
    return (
        line.strip().startswith("#")
        and ("bg-[#3064E3]/20" in line or "bg-[#4ADE8033]/20" in line)
    )


def last_non_empty_is_divider(lines: list[str]) -> bool:
    for line in reversed(lines):
        if line.strip() != "":
            return line.strip() == DIVIDER_LINE
    return False


def decorate_mdx_body(full_text: str) -> str:
    """
    Applies the pill/divider logic with tight HR placement, WITHOUT triggering
    Setext headings.
    """
    lines = full_text.splitlines()
    out: list[str] = []

    current_section: str | None = None  # None | "classes" | "functions"
    in_item: bool = False               # inside a class/function block?

    # Detect an "item heading" inside a section (class/function entry)
    item_heading_re = re.compile(r'^(#{3,6})\s+.*`[^`]+`')

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
            out.append("") # Ensure we are on a new line
            out.append(SPACER_BLOCK) # Insert the height div
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
        if stripped.startswith("## ") and stripped not in ("## Classes", "## Functions"):
            close_item_if_open()
            current_section = None
            in_item = False
            out.append(line)
            continue

        # Decorate headings (but do not add HR here)
        labeled = label_heading(line, current_section)
        stripped_labeled = labeled.strip()

        # Within Classes/Functions: a new item heading closes the prior item
        if current_section in ("classes", "functions") and item_heading_re.match(stripped_labeled):
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
# Path resolution + processing
# =========================

def resolve_api_dir(docs_root: Path | None, api_dir: Path | None) -> Path:
    if api_dir is not None:
        return api_dir
    if docs_root is not None:
        return docs_root / "api"

    cwd = Path.cwd()
    candidates = [
        cwd / "docs" / "api",
        cwd / "docs" / "docs" / "api",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def process_mdx_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")

    # Step 1: inject SidebarFix
    text = inject_sidebar_fix(original)

    # Step 2: decorate headings/dividers
    text = decorate_mdx_body(text)

    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-root", type=Path, default=None)
    parser.add_argument("--api-dir", type=Path, default=None)
    args = parser.parse_args()

    api_dir = resolve_api_dir(args.docs_root, args.api_dir)
    if not api_dir.exists():
        raise SystemExit(
            f"❌ API directory not found: {api_dir}\n"
            "Try:\n"
            "  python3 decorate_api_mdx_merged.py --docs-root /path/to/docs/docs\n"
            "or\n"
            "  python3 decorate_api_mdx_merged.py --api-dir /path/to/docs/docs/api\n"
        )

    mdx_files = list(api_dir.rglob("*.mdx"))
    if not mdx_files:
        print(f"⚠️ No .mdx files found under: {api_dir}")
        return

    changed = 0
    for f in mdx_files:
        if process_mdx_file(f):
            changed += 1

    print(f"✅ Done. Processed {len(mdx_files)} MDX files under {api_dir}. Updated {changed}.")


if __name__ == "__main__":
    main()
