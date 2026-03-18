#!/usr/bin/env python3
"""Unified API documentation build script.

Usage:
    python build.py                      # auto-detects version from pyproject.toml
    python build.py --version 0.5.0      # explicit version override
    python build.py --skip-decoration    # generation only
"""

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path


def read_project_version(repo_root: Path) -> str:
    """Read the project version from pyproject.toml."""
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.exists():
        raise SystemExit(f"❌ pyproject.toml not found at {pyproject}")
    with pyproject.open("rb") as f:
        data = tomllib.load(f)
    version = data.get("project", {}).get("version")
    if not version:
        raise SystemExit("❌ No [project].version found in pyproject.toml")
    return version


def normalize_version(version: str) -> str:
    """Strip pre-release suffixes for GitHub source links."""
    return version.split("-")[0]


def main():
    parser = argparse.ArgumentParser(description="Build API documentation")
    parser.add_argument(
        "--version",
        default=None,
        help="Package version (e.g., 0.5.0). Defaults to version in pyproject.toml.",
    )
    parser.add_argument(
        "--output-dir", default="docs/docs/api", help="Output directory"
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Source repo root to document (default: this repo). "
        "Use to generate docs for a different checkout, e.g. --source-dir ../mellea-b",
    )
    parser.add_argument(
        "--skip-generation", action="store_true", help="Skip AST generation"
    )
    parser.add_argument(
        "--skip-decoration", action="store_true", help="Skip MDX decoration"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    repo_root = (
        Path(args.source_dir).resolve() if args.source_dir else script_dir.parent.parent
    )
    output_dir = Path(args.output_dir)

    version = args.version or read_project_version(repo_root)
    normalized_version = normalize_version(version)

    # Step 1: Generate AST
    if not args.skip_generation:
        cmd = [
            sys.executable,
            str(script_dir / "generate-ast.py"),
            "--docs-root",
            str(
                output_dir.parent
            ),  # generate-ast.py expects docs/docs, not docs/docs/api
            "--source-dir",
            str(repo_root),
        ]

        print(f"[build.py] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(
                f"[build.py] ERROR: generate-ast.py failed with code {result.returncode}",
                file=sys.stderr,
            )
            sys.exit(result.returncode)

    # Step 2: Decorate MDX
    if not args.skip_decoration:
        source_pkg_dir = repo_root / "mellea"
        cmd = [
            sys.executable,
            str(script_dir / "decorate_api_mdx.py"),
            "--api-dir",
            str(output_dir),
            "--version",
            normalized_version,
        ]
        if source_pkg_dir.exists():
            cmd += ["--source-dir", str(source_pkg_dir)]

        print(f"[build.py] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(
                f"[build.py] ERROR: decorate_api_mdx.py failed with code {result.returncode}",
                file=sys.stderr,
            )
            sys.exit(result.returncode)

    # Step 3: Rebuild landing page and navigation from decorated files.
    # Decoration (step 2) injects rich preamble text into each module's MDX;
    # this step re-reads those decorated files so the landing page cards show
    # the full module descriptions rather than the short frontmatter one-liners.
    if not args.skip_generation and not args.skip_decoration:
        cmd = [
            sys.executable,
            str(script_dir / "generate-ast.py"),
            "--docs-root",
            str(output_dir.parent),
            "--nav-only",
            "--source-dir",
            str(repo_root),
        ]
        print(f"[build.py] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(
                f"[build.py] ERROR: generate-ast.py (nav-only) failed with code {result.returncode}",
                file=sys.stderr,
            )
            sys.exit(result.returncode)

    print(
        f"[build.py] ✅ Documentation build complete (version={version}, normalized={normalized_version})"
    )


if __name__ == "__main__":
    main()
