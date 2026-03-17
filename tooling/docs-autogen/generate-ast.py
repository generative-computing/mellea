#!/usr/bin/env python3
"""generate-ast.py — mdxify + postprocess docs pipeline.

Runs mdxify against the project's mellea and cli packages then postprocesses
the generated MDX files into the Mintlify docs tree.

Requires mdxify to be installed in the current Python environment.  Run via::

    uv run python tooling/docs-autogen/generate-ast.py

Pipeline:
  1) Run mdxify --all for root modules: mellea, cli into STAGING: <repo-root>/docs/api/<pkg>
  2) Reorganize flat mdxify output into nested folders
  3) Rename __init__.mdx -> <foldername>.mdx (dedupe if identical)
  4) Update frontmatter (title/sidebarTitle/description) from H1 + first paragraph
  5) Remove truly-empty MDX files
  6) Move generated docs to <docs-root>/api (replace existing)
  7) Build Mintlify API Reference nav (NO .mdx suffix)
  8) Merge by replacing ONLY: { "tab": "API Reference", ... } in docs.json
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

NAV_TAB = "API Reference"
PACKAGES = ["mellea", "cli"]

# Script is in tooling/docs-autogen/generate-ast.py -> repo root is 2 parents up
REPO_ROOT = Path(__file__).resolve().parents[2]

# Staging output (inside repo root; later moved into docs-root/api)
STAGING_DOCS_ROOT = REPO_ROOT / "docs"
STAGING_API_DIR = STAGING_DOCS_ROOT / "api"

# Clean run dir for mdxify (outside the repo root so repo-local packages
# cannot shadow site-packages via directory-level import precedence)
MDXIFY_CWD = REPO_ROOT / ".mdxify-run-cwd"

# If you want explicit link backing, keep repo-url. If you want mdxify auto-detection, omit.
REPO_URL = "https://github.com/generative-computing/mellea"


# -----------------------------
# Helpers
# -----------------------------


def yaml_quote(value: str | None) -> str:
    if value is None:
        return '""'
    v = str(value).replace("\\", "\\\\").replace('"', '\\"')
    v = v.replace("\r\n", "\n").replace("\n", "\\n")
    return f'"{v}"'


def safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def safe_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def strip_frontmatter(lines: list[str]) -> list[str]:
    if lines and lines[0].strip() == "---":
        try:
            end_idx = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
            return lines[end_idx + 1 :]
        except StopIteration:
            return []
    return lines


def is_meaningful_body_line(line: str) -> bool:
    # headings count as meaningful so we don't delete index pages.
    s = line.strip()
    if not s:
        return False
    if s.startswith("<!--") and s.endswith("-->"):
        return False
    return True


def find_docs_json(cli_path: str | None, search_root: Path | None = None) -> Path:
    if cli_path:
        p = Path(cli_path)
        if not p.is_absolute():
            root = search_root if search_root is not None else REPO_ROOT
            p = (root / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"--docs-json path not found: {p}")
        return p

    root = search_root if search_root is not None else REPO_ROOT
    candidates = [
        root / "docs.json",
        root / "docs" / "docs.json",
        root / "docs" / "docs" / "docs.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Could not locate docs.json under {root}. "
        "Pass --docs-json explicitly, e.g. --docs-json docs/docs/docs.json"
    )


def merge_api_reference_into_docs_json(
    docs_json_path: Path, api_tab_obj: dict[str, Any]
) -> None:
    data = json.loads(docs_json_path.read_text(encoding="utf-8"))
    nav = data.get("navigation") or {}
    tabs = nav.get("tabs") or []

    if not isinstance(tabs, list) or not tabs:
        raise RuntimeError("docs.json has no navigation.tabs (or it's empty)")

    replaced = False
    for i, tab in enumerate(tabs):
        if isinstance(tab, dict) and tab.get("tab") == NAV_TAB:
            tabs[i] = api_tab_obj
            replaced = True
            break

    if not replaced:
        raise RuntimeError(f'No tab named "{NAV_TAB}" found in docs.json')

    nav["tabs"] = tabs
    data["navigation"] = nav
    docs_json_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"✅ Merged API Reference tab into: {docs_json_path}", flush=True)


# -----------------------------
# mdxify generation
# -----------------------------
def run_mdxify_generation(root_module: str, source_root: Path | None = None) -> None:
    output_dir = STAGING_API_DIR / root_module
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run mdxify from a clean cwd so repo-local source directories don't take
    # precedence over site-packages via directory-level import shadowing.
    if MDXIFY_CWD.exists():
        shutil.rmtree(MDXIFY_CWD)
    MDXIFY_CWD.mkdir(parents=True, exist_ok=True)

    print(
        f"➡️ Generating documentation for root module: {root_module} into {output_dir}",
        flush=True,
    )

    cmd = [
        sys.executable,
        "-m",
        "mdxify",
        "--all",
        "--root-module",
        root_module,
        "--output-dir",
        str(output_dir),
        "--no-update-nav",
        "-v",
        "--repo-url",
        REPO_URL,
    ]

    env = os.environ.copy()
    if source_root is not None:
        # Point mdxify at an external source tree (--source-dir).
        # Prepend it to PYTHONPATH so it takes precedence over any installed copy.
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{source_root}:{existing}" if existing else str(source_root)
        )
    else:
        # Default: drop PYTHONPATH so no repo-local directory shadows site-packages.
        env.pop("PYTHONPATH", None)

    subprocess.run(cmd, check=True, text=True, cwd=str(MDXIFY_CWD), env=env)
    print(f"✅ Successfully generated docs for {root_module}", flush=True)


# -----------------------------
# Restructure + cleanup
# -----------------------------
def reorganize_to_nested_structure() -> None:
    print("-" * 30, flush=True)
    print("📁 Reorganizing MDX files into nested folder structure...", flush=True)

    all_mdx = glob.glob(str(STAGING_API_DIR / "**" / "*.mdx"), recursive=True)

    for old in all_mdx:
        old_path = Path(old)
        pkg = old_path.parent.name
        parent_dir = old_path.parent

        if pkg not in PACKAGES:
            continue

        # Only reorganize files directly under docs/api/<pkg> (flat)
        if parent_dir != (STAGING_API_DIR / pkg):
            continue

        base = old_path.stem
        prefix = f"{pkg}-"
        if not base.startswith(prefix):
            continue

        module_path_raw = base[len(prefix) :]
        if not module_path_raw:
            continue

        parts = module_path_raw.split("-")
        new_dir = STAGING_API_DIR / pkg / Path(*parts[:-1])
        new_path = new_dir / f"{parts[-1]}.mdx"

        if new_path.resolve() == old_path.resolve():
            continue

        new_dir.mkdir(parents=True, exist_ok=True)
        print(f"   Moving {old_path} -> {new_path}", flush=True)
        old_path.replace(new_path)

    print("✅ Folder reorganization complete.", flush=True)


def rename_init_files_to_parent() -> None:
    print("-" * 30, flush=True)
    print(
        "📛 Renaming __init__.mdx files to folder-name.mdx (dedupe if identical)...",
        flush=True,
    )

    init_files = glob.glob(str(STAGING_API_DIR / "**" / "__init__.mdx"), recursive=True)

    def normalize_text(s: str) -> str:
        return "\n".join(
            line.rstrip() for line in s.replace("\r\n", "\n").split("\n")
        ).strip()

    for old in init_files:
        old_path = Path(old)
        folder = old_path.parent.name
        new_path = old_path.parent / f"{folder}.mdx"

        if not new_path.exists():
            print(f"   Renaming {old_path} -> {new_path}", flush=True)
            old_path.rename(new_path)
            continue

        old_txt = normalize_text(safe_read_text(old_path))
        new_txt = normalize_text(safe_read_text(new_path))

        if old_txt == new_txt:
            print(
                f"   🗑️ Duplicate content: removing {old_path} (same as {new_path})",
                flush=True,
            )
            old_path.unlink(missing_ok=True)
        else:
            print(
                f"   ⚠️ Content differs: keeping {old_path} (leaving existing {new_path})",
                flush=True,
            )

    print("✅ __init__.mdx rename/dedupe pass complete.", flush=True)


def extract_title_and_description(
    body_lines: list[str],
) -> tuple[str | None, str | None, int | None, int | None]:
    h1_pattern = re.compile(r"^#\s+`?(.+?)`?\s*$")

    title_value = None
    h1_idx = None
    for i, line in enumerate(body_lines):
        m = h1_pattern.match(line.strip())
        if m:
            title_value = m.group(1).strip("`").strip()
            h1_idx = i
            break

    if not title_value or h1_idx is None:
        return None, None, None, None

    desc_value = None
    desc_idx = None
    for j in range(h1_idx + 1, len(body_lines)):
        s = body_lines[j].strip()
        if not s:
            continue
        if s.startswith("#"):
            break
        if s.startswith("```"):
            continue
        desc_value = s
        desc_idx = j
        break

    return title_value, desc_value, h1_idx, desc_idx


def update_frontmatter_metadata() -> None:
    print("-" * 30, flush=True)
    print(
        "📝 Updating frontmatter title/description/sidebarTitle from content...",
        flush=True,
    )

    mdx_files = glob.glob(str(STAGING_API_DIR / "**" / "*.mdx"), recursive=True)

    for p in mdx_files:
        path = Path(p)
        text = safe_read_text(path)
        lines = text.splitlines()

        if not lines or lines[0].strip() != "---":
            continue

        try:
            end_idx = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
        except StopIteration:
            continue

        front_lines = lines[1:end_idx]
        body_lines = lines[end_idx + 1 :]

        title_value, desc_value, h1_idx, desc_idx = extract_title_and_description(
            body_lines
        )
        if not title_value:
            continue

        preserved = []
        for line in front_lines:
            k = line.strip()
            if k.startswith(("title:", "sidebarTitle:", "description:")):
                continue
            preserved.append(line)

        cleaned_body: list[str] = []
        for idx, line in enumerate(body_lines):
            if h1_idx is not None and idx == h1_idx:
                continue
            if desc_idx is not None and idx == desc_idx:
                continue
            cleaned_body.append(line)

        new_front = ["---"]
        new_front.append(f"title: {yaml_quote(title_value)}")
        new_front.append(f"sidebarTitle: {yaml_quote(title_value)}")
        if desc_value:
            new_front.append(f"description: {yaml_quote(desc_value)}")
        new_front.extend(preserved)
        new_front.append("---")

        new_text = "\n".join(new_front + cleaned_body).rstrip() + "\n"
        safe_write_text(path, new_text)

    print("✅ Frontmatter update complete.", flush=True)


def remove_empty_mdx_files() -> None:
    print("-" * 30, flush=True)
    print("🧹 Removing empty/no-content MDX files...", flush=True)

    mdx_files = glob.glob(str(STAGING_API_DIR / "**" / "*.mdx"), recursive=True)
    removed = 0

    for p in mdx_files:
        path = Path(p)
        lines = safe_read_text(path).splitlines()
        body = strip_frontmatter(lines)
        meaningful = any(is_meaningful_body_line(line) for line in body)
        if not meaningful:
            print(f"   🗑️ Removing empty file: {path}", flush=True)
            path.unlink(missing_ok=True)
            removed += 1

    print(f"✅ Removed {removed} empty files.", flush=True)


# -----------------------------
# Move + nav generation
# -----------------------------
def move_api_to_docs_root(target_docs_root: Path) -> Path:
    target_docs_root = target_docs_root.resolve()
    target_api_dir = target_docs_root / "api"

    print("-" * 30, flush=True)
    print(f"📦 Moving generated API docs to: {target_api_dir}", flush=True)

    if not STAGING_API_DIR.exists():
        raise RuntimeError(f"Staging API dir not found: {STAGING_API_DIR}")

    if target_api_dir.exists():
        print(f"   🧹 Deleting existing target api dir: {target_api_dir}", flush=True)
        shutil.rmtree(target_api_dir)

    target_docs_root.mkdir(parents=True, exist_ok=True)
    shutil.move(str(STAGING_API_DIR), str(target_api_dir))

    print("✅ Move complete.", flush=True)
    return target_api_dir


def build_tree_from_paths(paths: list[str]) -> dict[str, Any]:
    root: dict[str, Any] = {}

    def insert(node: dict[str, Any], parts: list[str], page_path: str) -> None:
        if not parts:
            node.setdefault("__pages__", []).append(page_path)
            return
        k = parts[0]
        node.setdefault(k, {})
        insert(node[k], parts[1:], page_path)

    for p in paths:
        parts = p.split("/")
        if len(parts) < 3:
            continue
        sub = parts[2:]  # after api/<pkg>
        insert(root, sub[:-1], p)

    return root


def tree_to_mintlify(node: dict[str, Any], group_name: str) -> dict[str, Any]:
    pages: list[Any] = []
    file_pages = node.get("__pages__", [])
    if file_pages:
        pages.extend(sorted(file_pages))

    for k in sorted(x for x in node.keys() if x != "__pages__"):
        pages.append(tree_to_mintlify(node[k], k))

    return {"group": group_name, "pages": pages}


def collect_pages_under(api_dir: Path, pkg: str, docs_root: Path) -> list[str]:
    base = api_dir / pkg
    files = glob.glob(str(base / "**" / "*.mdx"), recursive=True)

    out: list[str] = []
    for f in files:
        fp = Path(f)
        rel = fp.relative_to(docs_root)
        out.append(rel.as_posix().removesuffix(".mdx"))
    return sorted(out)


def _read_frontmatter_field(mdx_path: Path, field: str) -> str:
    """Extract a single YAML frontmatter field value from an MDX file."""
    try:
        content = mdx_path.read_text(encoding="utf-8")
    except OSError:
        return ""
    if not content.startswith("---\n"):
        return ""
    end = content.find("\n---\n", 4)
    if end == -1:
        return ""
    fm = content[4:end]
    for line in fm.splitlines():
        if line.startswith(f"{field}:"):
            val = line[len(field) + 1 :].strip().strip('"').strip("'")
            return val
    return ""


_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def _read_body_preamble(mdx_path: Path) -> str:
    """Extract the first rich prose paragraph from an MDX body.

    Skips frontmatter, import lines, and JSX/HTML tags, then returns the
    first meaningful paragraph of prose text before any heading.

    Markdown links are simplified to their display text, and { / } are
    escaped as HTML entities so the text is safe inside JSX Card children.
    """
    try:
        content = mdx_path.read_text(encoding="utf-8")
    except OSError:
        return ""
    # Strip frontmatter
    if content.startswith("---\n"):
        end = content.find("\n---\n", 4)
        if end != -1:
            content = content[end + 5 :]
    lines = content.splitlines()
    current_para: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            if current_para:
                break
            continue
        # Skip import lines, JSX/HTML tags, and SidebarFix
        if s.startswith(("import ", "<")) or "SidebarFix" in s or s == "---":
            if current_para:
                break
            continue
        # Stop at the first heading
        if s.startswith("#"):
            break
        current_para.append(s)
    text = " ".join(current_para).strip()
    # Simplify markdown links to their display text
    text = _MD_LINK_RE.sub(r"\1", text)
    # Escape { and } for safe rendering inside JSX Card children
    text = text.replace("{", "&#123;").replace("}", "&#125;")
    return text


def _find_index_mdx(child_dir: Path) -> Path | None:
    """Return the best representative MDX file for a module subdirectory."""
    # Prefer a file whose stem matches the directory name
    candidate = child_dir / f"{child_dir.name}.mdx"
    if candidate.exists():
        return candidate
    mdxs = sorted(child_dir.glob("*.mdx"))
    return mdxs[0] if mdxs else None


def _collect_module_entries(
    api_dir: Path, pkg: str, docstring_cache: dict[str, str] | None = None
) -> list[tuple[str, str, str, str]]:
    """Return (module_name, description, preamble, href) for direct children of api/<pkg>/.

    Only includes entries at the immediate subdirectory level (not deep nested),
    to keep the landing page overview concise.

    The preamble for each entry comes from the package's __init__ docstring
    (looked up in docstring_cache via the dotted module path), falling back to
    reading the body of the representative MDX file when no cache is available.
    This follows standard Python practice: __init__.py is the canonical
    description of a package.
    """
    base = api_dir / pkg
    if not base.exists():
        return []
    entries = []
    for child in sorted(base.iterdir()):
        if child.is_dir():
            index_mdx = _find_index_mdx(child)
            if index_mdx is None:
                continue
            module_path = f"{pkg}.{child.name}"
            desc = _read_frontmatter_field(index_mdx, "description")
            if docstring_cache:
                preamble = docstring_cache.get(module_path, "")
            else:
                preamble = _read_body_preamble(index_mdx)
            href = f"api/{pkg}/{child.name}/{index_mdx.stem}"
            entries.append((child.name, desc, preamble, href))
        elif child.suffix == ".mdx":
            module_path = f"{pkg}.{child.stem}"
            desc = _read_frontmatter_field(child, "description")
            if docstring_cache:
                preamble = docstring_cache.get(module_path, "")
            else:
                preamble = _read_body_preamble(child)
            href = f"api/{pkg}/{child.stem}"
            entries.append((child.stem, desc, preamble, href))
    return entries


def generate_landing_page(
    api_dir: Path, docs_root: Path, docstring_cache: dict[str, str] | None = None
) -> None:
    """Generate api-reference.mdx as a landing page for the API Reference tab.

    Scans the generated api/ directory structure to produce an up-to-date
    overview of every top-level module, rendered as a prose list with links.
    The file is written to docs_root so Mintlify serves it when the user
    clicks the API Reference tab.

    When docstring_cache is provided, package descriptions are sourced directly
    from __init__ module docstrings (standard Python practice) rather than
    inferred from the representative MDX file body.
    """
    mellea_entries = _collect_module_entries(api_dir, "mellea", docstring_cache)
    cli_entries = _collect_module_entries(api_dir, "cli", docstring_cache)

    def _entry(name: str, desc: str, preamble: str, href: str, pkg: str) -> str:
        title = f"mellea.{name}" if pkg == "mellea" else f"m {name}"
        body = preamble or desc
        line = f"- **[{title}](/{href})**"
        if body:
            line += f" — {body}"
        return line

    def _list_section(entries: list[tuple[str, str, str, str]], pkg: str) -> str:
        if not entries:
            return "_No modules found._\n"
        return "\n".join(_entry(n, d, p, h, pkg) for n, d, p, h in entries) + "\n"

    content = """\
---
title: "Mellea API Reference"
description: "Complete reference for Mellea's Python API and command-line tools"
---

Mellea is a library for writing generative programs. Generative programming replaces
flaky agents and brittle prompts with structured, maintainable, and efficient AI
workflows.

## Python Library

"""
    content += _list_section(mellea_entries, "mellea")
    content += "\n## CLI (`m`)\n\n"
    content += _list_section(cli_entries, "cli")

    out_path = docs_root / "api-reference.mdx"
    safe_write_text(out_path, content)
    print(f"✅ Generated landing page: {out_path}", flush=True)


def build_api_reference_tab_object(api_dir: Path, docs_root: Path) -> dict[str, Any]:
    cli_pages = collect_pages_under(api_dir, "cli", docs_root)
    mellea_pages = collect_pages_under(api_dir, "mellea", docs_root)

    cli_tree = build_tree_from_paths(cli_pages)
    mellea_tree = build_tree_from_paths(mellea_pages)

    cli_nav = tree_to_mintlify(cli_tree, "cli")
    mellea_nav = tree_to_mintlify(mellea_tree, "mellea")

    return {
        "tab": NAV_TAB,
        "pages": [
            "api-reference",  # landing page, generated by generate_landing_page()
            {"group": "mellea", "pages": mellea_nav["pages"]},
            {"group": "cli", "pages": cli_nav["pages"]},
        ],
    }


def _load_docstring_cache(source_root: Path | None = None) -> dict[str, str]:
    """Load the module-path→docstring cache from the source packages.

    Imports build_module_docstring_cache from decorate_api_mdx (same directory)
    and builds the cache from the given source root (defaults to REPO_ROOT).
    Returns an empty dict on any failure so callers degrade gracefully to the
    _read_body_preamble() fallback.

    Args:
        source_root: Root of the repo whose mellea/ package to read. Defaults
            to REPO_ROOT (this checkout).

    Returns:
        Dict mapping module dotted paths to their first-paragraph docstrings.
    """
    root = source_root if source_root is not None else REPO_ROOT
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from decorate_api_mdx import (
            build_module_docstring_cache,  # type: ignore[import]
        )

        cache = build_module_docstring_cache(root / "mellea")
        print(f"  Loaded {len(cache)} module docstrings from source.", flush=True)
        return cache
    except Exception as exc:
        print(f"  WARNING: could not load docstring cache: {exc}", flush=True)
        return {}


def build_and_merge_navigation(
    docs_json_path: Path,
    api_dir: Path,
    docs_root: Path,
    docstring_cache: dict[str, str] | None = None,
) -> None:
    print("-" * 30, flush=True)
    print(
        "🛠️ Building API Reference navigation and merging into docs.json...", flush=True
    )
    generate_landing_page(api_dir, docs_root, docstring_cache)
    api_tab = build_api_reference_tab_object(api_dir, docs_root)
    merge_api_reference_into_docs_json(docs_json_path, api_tab)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MDX API docs with mdxify, postprocess, and merge into the Mintlify docs tree."
    )
    parser.add_argument(
        "--docs-json", required=False, help="Path to docs.json to update."
    )
    parser.add_argument(
        "--docs-root",
        required=False,
        help="Mintlify docs root (defaults to parent of docs.json).",
    )
    parser.add_argument(
        "--nav-only",
        action="store_true",
        help=(
            "Skip MDX generation; only rebuild the landing page and navigation from "
            "existing files in docs-root/api. Use after decoration so that preamble "
            "text injected by decorate_api_mdx.py appears in landing page cards."
        ),
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Source repo root whose mellea/ and cli/ packages to document "
        "(default: this repo). Overrides REPO_ROOT for package lookup and "
        "docstring cache; does not affect staging/output paths.",
    )

    args = parser.parse_args()

    # source_root: the repo whose mellea/ and cli/ packages to document.
    # Defaults to REPO_ROOT (this checkout). Override with --source-dir when
    # generating docs for a different repo without modifying that repo.
    source_root = Path(args.source_dir).resolve() if args.source_dir else REPO_ROOT

    docs_json_path = find_docs_json(args.docs_json, search_root=source_root)
    docs_root = (
        Path(args.docs_root).resolve()
        if args.docs_root
        else docs_json_path.parent.resolve()
    )

    # --nav-only: skip all MDX generation; just rebuild landing page + nav
    if args.nav_only:
        final_api_dir = (docs_root / "api").resolve()
        if not final_api_dir.exists():
            raise SystemExit(
                f"❌ API dir not found: {final_api_dir}\n"
                "Run the full pipeline first (without --nav-only)."
            )
        print("🔄 --nav-only: rebuilding landing page and navigation...", flush=True)
        docstring_cache = _load_docstring_cache(source_root)
        build_and_merge_navigation(
            docs_json_path, final_api_dir, docs_root, docstring_cache
        )
        print("🎉 Navigation rebuild complete.", flush=True)
        return

    # Prep staging
    if STAGING_API_DIR.exists():
        shutil.rmtree(STAGING_API_DIR)
    STAGING_API_DIR.mkdir(parents=True, exist_ok=True)

    # Generate MDX into staging (clean cwd inside run_mdxify_generation)
    for pkg in PACKAGES:
        run_mdxify_generation(pkg, source_root if source_root != REPO_ROOT else None)

    # Restructure + cleanup in staging
    reorganize_to_nested_structure()
    rename_init_files_to_parent()
    update_frontmatter_metadata()
    remove_empty_mdx_files()

    # Move staging api -> final docs root/api
    final_api_dir = move_api_to_docs_root(docs_root)

    # Merge nav based on final location
    build_and_merge_navigation(
        docs_json_path, final_api_dir, docs_root, _load_docstring_cache(source_root)
    )

    # Cleanup mdxify run cwd (optional)
    if MDXIFY_CWD.exists():
        shutil.rmtree(MDXIFY_CWD)

    print("-" * 30, flush=True)
    print("🎉 All tasks complete!", flush=True)


if __name__ == "__main__":
    main()
