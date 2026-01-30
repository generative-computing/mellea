#!/usr/bin/env python3
"""
generate-ast.py (INSTALL mode, then mdxify + postprocess)

Key fix vs your prior script:
  - mdxify is executed with cwd set to a clean temp directory OUTSIDE the repo root,
    so Python imports the INSTALLED distribution (site-packages) and does not get
    shadowed by repo-local ./mellea or ./cli packages.

Pipeline:
  1) Create/Reuse venv: <repo-root>/.venv-docs-autogen
  2) pip install: mdxify + mellea (optionally pinned)
  3) Run mdxify --all for root modules: mellea, cli into STAGING: <repo-root>/docs/api/<pkg>
  4) Reorganize flat mdxify output into nested folders
  5) Rename __init__.mdx -> <foldername>.mdx (dedupe if identical)
  6) Update frontmatter (title/sidebarTitle/description) from H1 + first paragraph
  7) Remove truly-empty MDX files
  8) Move generated docs to <docs-root>/api (replace existing)
  9) Build Mintlify API Reference nav (NO .mdx suffix)
 10) Merge by replacing ONLY: { "tab": "API Reference", ... } in docs.json

Usage:
  python3 tooling/docs-autogen/generate-ast.py \
    --docs-json docs/docs/docs.json \
    --docs-root docs/docs \
    --pypi-name mellea \
    --pypi-version 0.3.0

Notes:
  - --pypi-version may be "v0.3.0" or "0.3.0". If omitted, installs latest.
"""

import os
import sys
import json
import glob
import re
import subprocess
import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


NAV_TAB = "API Reference"
PACKAGES = ["mellea", "cli"]

# Script is in tooling/docs-autogen/generate-ast.py -> repo root is 2 parents up
REPO_ROOT = Path(__file__).resolve().parents[2]

# Staging output (inside repo root; later moved into docs-root/api)
STAGING_DOCS_ROOT = REPO_ROOT / "docs"
STAGING_API_DIR = STAGING_DOCS_ROOT / "api"

# Venv + clean run dir (outside import shadowing)
VENV_DIR = REPO_ROOT / ".venv-docs-autogen"
MDXIFY_CWD = REPO_ROOT / ".mdxify-run-cwd"  # must NOT contain mellea/ or cli/ packages

# If you want explicit link backing, keep repo-url. If you want mdxify auto-detection, omit.
REPO_URL = "https://github.com/generative-computing/mellea"


# -----------------------------
# Helpers
# -----------------------------
def normalize_version(v: Optional[str]) -> Optional[str]:
    if not v:
        return None
    return v[1:] if v.startswith("v") else v


def yaml_quote(value: Optional[str]) -> str:
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


def strip_frontmatter(lines: List[str]) -> List[str]:
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


def find_docs_json(cli_path: Optional[str]) -> Path:
    if cli_path:
        p = Path(cli_path)
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"--docs-json path not found: {p}")
        return p

    candidates = [
        REPO_ROOT / "docs.json",
        REPO_ROOT / "docs" / "docs.json",
        REPO_ROOT / "docs" / "docs" / "docs.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Could not locate docs.json. Pass --docs-json explicitly, e.g. --docs-json docs/docs/docs.json"
    )


def merge_api_reference_into_docs_json(docs_json_path: Path, api_tab_obj: Dict[str, Any]) -> None:
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
# Venv + installs
# -----------------------------
def ensure_venv() -> Path:
    if not VENV_DIR.exists():
        print(f"🧪 Creating venv: {VENV_DIR}", flush=True)
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)

    py = VENV_DIR / ("Scripts" if os.name == "nt" else "bin") / "python"
    if not py.exists():
        raise RuntimeError(f"Venv python not found at: {py}")
    return py


def pip_install(venv_python: Path, pypi_name: str, pypi_version: Optional[str]) -> None:
    ver = normalize_version(pypi_version)
    spec = pypi_name if not ver else f"{pypi_name}=={ver}"

    print(f"📦 Installing into venv: mdxify + {spec}", flush=True)
    subprocess.run([str(venv_python), "-m", "pip", "install", "-U", "pip"], check=True)
    subprocess.run([str(venv_python), "-m", "pip", "install", "-U", "mdxify", spec], check=True)


# -----------------------------
# mdxify generation
# -----------------------------
def run_mdxify_generation(venv_python: Path, root_module: str) -> None:
    output_dir = STAGING_API_DIR / root_module
    output_dir.mkdir(parents=True, exist_ok=True)

    # Critical: run mdxify from a clean cwd so repo-local packages don't shadow site-packages
    if MDXIFY_CWD.exists():
        shutil.rmtree(MDXIFY_CWD)
    MDXIFY_CWD.mkdir(parents=True, exist_ok=True)

    print(f"➡️ Generating documentation for root module: {root_module} into {output_dir}", flush=True)

    cmd = [
        str(venv_python),
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

    # Ensure PYTHONPATH doesn't accidentally include repo roots
    env = os.environ.copy()
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
    print("📛 Renaming __init__.mdx files to folder-name.mdx (dedupe if identical)...", flush=True)

    init_files = glob.glob(str(STAGING_API_DIR / "**" / "__init__.mdx"), recursive=True)

    def normalize_text(s: str) -> str:
        return "\n".join(line.rstrip() for line in s.replace("\r\n", "\n").split("\n")).strip()

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
            print(f"   🗑️ Duplicate content: removing {old_path} (same as {new_path})", flush=True)
            old_path.unlink(missing_ok=True)
        else:
            print(f"   ⚠️ Content differs: keeping {old_path} (leaving existing {new_path})", flush=True)

    print("✅ __init__.mdx rename/dedupe pass complete.", flush=True)


def extract_title_and_description(body_lines: List[str]) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[int]]:
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
    print("📝 Updating frontmatter title/description/sidebarTitle from content...", flush=True)

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

        title_value, desc_value, h1_idx, desc_idx = extract_title_and_description(body_lines)
        if not title_value:
            continue

        preserved = []
        for line in front_lines:
            k = line.strip()
            if k.startswith("title:") or k.startswith("sidebarTitle:") or k.startswith("description:"):
                continue
            preserved.append(line)

        cleaned_body: List[str] = []
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


def build_tree_from_paths(paths: List[str]) -> Dict[str, Any]:
    root: Dict[str, Any] = {}

    def insert(node: Dict[str, Any], parts: List[str], page_path: str) -> None:
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


def tree_to_mintlify(node: Dict[str, Any], group_name: str) -> Dict[str, Any]:
    pages: List[Any] = []
    file_pages = node.get("__pages__", [])
    if file_pages:
        pages.extend(sorted(file_pages))

    for k in sorted(x for x in node.keys() if x != "__pages__"):
        pages.append(tree_to_mintlify(node[k], k))

    return {"group": group_name, "pages": pages}


def collect_pages_under(api_dir: Path, pkg: str, docs_root: Path) -> List[str]:
    base = api_dir / pkg
    files = glob.glob(str(base / "**" / "*.mdx"), recursive=True)

    out: List[str] = []
    for f in files:
        fp = Path(f)
        rel = fp.relative_to(docs_root)
        out.append(rel.as_posix().removesuffix(".mdx"))
    return sorted(out)


def build_api_reference_tab_object(api_dir: Path, docs_root: Path) -> Dict[str, Any]:
    cli_pages = collect_pages_under(api_dir, "cli", docs_root)
    mellea_pages = collect_pages_under(api_dir, "mellea", docs_root)

    cli_tree = build_tree_from_paths(cli_pages)
    mellea_tree = build_tree_from_paths(mellea_pages)

    cli_nav = tree_to_mintlify(cli_tree, "cli")
    mellea_nav = tree_to_mintlify(mellea_tree, "mellea")

    return {
        "tab": NAV_TAB,
        "pages": [
            {"group": "mellea", "pages": mellea_nav["pages"]},
            {"group": "cli", "pages": cli_nav["pages"]},
        ],
    }


def build_and_merge_navigation(docs_json_path: Path, api_dir: Path, docs_root: Path) -> None:
    print("-" * 30, flush=True)
    print("🛠️ Building API Reference navigation and merging into docs.json...", flush=True)
    api_tab = build_api_reference_tab_object(api_dir, docs_root)
    merge_api_reference_into_docs_json(docs_json_path, api_tab)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install mellea + mdxify in a venv, generate MDX API docs, postprocess, move to docs root, merge nav."
    )
    parser.add_argument("--docs-json", required=False, help="Path to docs.json to update.")
    parser.add_argument("--docs-root", required=False, help="Mintlify docs root (defaults to parent of docs.json).")
    parser.add_argument("--pypi-name", default="mellea", help="PyPI project name to install (default: mellea).")
    parser.add_argument("--pypi-version", required=False, help="Version like v0.3.0 or 0.3.0. Omit for latest.")

    args = parser.parse_args()

    docs_json_path = find_docs_json(args.docs_json)
    docs_root = Path(args.docs_root).resolve() if args.docs_root else docs_json_path.parent.resolve()

    # Prep staging
    if STAGING_API_DIR.exists():
        shutil.rmtree(STAGING_API_DIR)
    STAGING_API_DIR.mkdir(parents=True, exist_ok=True)

    venv_python = ensure_venv()
    pip_install(venv_python, args.pypi_name, args.pypi_version)

    # Generate MDX into staging (critical cwd fix inside run_mdxify_generation)
    for pkg in PACKAGES:
        run_mdxify_generation(venv_python, pkg)

    # Restructure + cleanup in staging
    reorganize_to_nested_structure()
    rename_init_files_to_parent()
    update_frontmatter_metadata()
    remove_empty_mdx_files()

    # Move staging api -> final docs root/api
    final_api_dir = move_api_to_docs_root(docs_root)

    # Merge nav based on final location
    build_and_merge_navigation(docs_json_path, final_api_dir, docs_root)

    # Cleanup mdxify run cwd (optional)
    if MDXIFY_CWD.exists():
        shutil.rmtree(MDXIFY_CWD)

    print("-" * 30, flush=True)
    print("🎉 All tasks complete!", flush=True)


if __name__ == "__main__":
    main()
