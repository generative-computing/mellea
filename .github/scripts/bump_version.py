#!/usr/bin/env python3
# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute and commit the next release version.

Reads the current version from pyproject.toml, computes the next version per
the requested mode, writes it back, refreshes uv.lock, and commits. The
computed version is printed to stdout for callers to capture.

Modes:
  rc           — X.Y.ZrcN -> X.Y.Zrc(N+1)
  final        — X.Y.0rcN -> X.Y.0         (first final of the minor)
  patch-rc     — X.Y.Z    -> X.Y.(Z+1)rc0  | X.Y.(Z+1)rcN -> X.Y.(Z+1)rc(N+1)
  patch-final  — X.Y.ZrcN (Z>0) -> X.Y.Z   (promote patch rc to final)
  dev          — X.Y.Z.devN -> X.Y.Z.dev(N+1)   (main-only)

`dev` mode runs on `main` and iterates its .devN counter. All other modes
run on `release/v*` branches.

With --dry-run the script prints the proposed version and exits without
writing or committing.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"


def read_current_version() -> Version:
    with PYPROJECT.open("rb") as f:
        data = tomllib.load(f)
    raw = data["project"]["version"]
    return Version(raw)


def existing_tags() -> set[str]:
    out = subprocess.run(
        ["git", "tag", "--list", "v*"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return {line.strip() for line in out.stdout.splitlines() if line.strip()}


def current_branch() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def compute_next(current: Version, mode: str) -> Version:
    """Compute the next version per mode. Raises ValueError on disallowed transitions."""
    major, minor, patch = (
        current.release[0],
        current.release[1],
        (current.release[2] if len(current.release) > 2 else 0),
    )

    if mode == "dev":
        if current.dev is None:
            raise ValueError(
                f"mode=dev requires current version to be a .dev release; got {current}."
            )
        if current.pre is not None:
            raise ValueError(
                f"mode=dev does not support .devN combined with a pre-release "
                f"segment; got {current}."
            )
        return Version(f"{major}.{minor}.{patch}.dev{current.dev + 1}")

    if current.dev is not None:
        raise ValueError(
            f"Current version {current} is a .dev release; mode {mode!r} only "
            "operates on release branches (rc/final). Ran on the wrong branch?"
        )

    if mode == "rc":
        if current.pre is None or current.pre[0] != "rc":
            raise ValueError(
                f"mode=rc requires current version to be an rc; got {current}. "
                "If this is a final, use mode=patch-rc to start a patch cycle."
            )
        return Version(f"{major}.{minor}.{patch}rc{current.pre[1] + 1}")

    if mode == "final":
        if current.pre is None or current.pre[0] != "rc":
            raise ValueError(
                f"mode=final requires current version to be an rc; got {current}."
            )
        if patch != 0:
            raise ValueError(
                f"mode=final is for promoting minor rcs (X.Y.0rcN -> X.Y.0); "
                f"got patch version {current}. Use mode=patch-final for patches."
            )
        return Version(f"{major}.{minor}.{patch}")

    if mode == "patch-rc":
        if current.pre is None:
            return Version(f"{major}.{minor}.{patch + 1}rc0")
        if current.pre[0] != "rc":
            raise ValueError(f"Unexpected pre-release segment in {current}")
        if patch == 0:
            raise ValueError(
                f"mode=patch-rc requires an existing final or patch-rc; got "
                f"{current} which is a minor rc. Use mode=rc to iterate minor rcs."
            )
        return Version(f"{major}.{minor}.{patch}rc{current.pre[1] + 1}")

    if mode == "patch-final":
        if current.pre is None or current.pre[0] != "rc":
            raise ValueError(
                f"mode=patch-final requires current to be a patch rc; got {current}."
            )
        if patch == 0:
            raise ValueError(
                f"mode=patch-final is for patches (Z>0); got {current}. "
                "Use mode=final to promote a minor rc."
            )
        return Version(f"{major}.{minor}.{patch}")

    raise ValueError(f"Unknown mode: {mode!r}")


def write_pyproject(new_version: Version) -> None:
    content = PYPROJECT.read_text()
    pattern = re.compile(r'^(version\s*=\s*")[^"]+(")', re.MULTILINE)
    new_content, n = pattern.subn(rf"\g<1>{new_version}\g<2>", content, count=1)
    if n != 1:
        raise RuntimeError("Failed to locate version line in pyproject.toml")
    PYPROJECT.write_text(new_content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["rc", "final", "patch-rc", "patch-final", "dev"],
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the proposed next version and exit without writing or committing.",
    )
    parser.add_argument(
        "--skip-branch-check",
        action="store_true",
        help="Skip the branch assertion. For local testing only.",
    )
    args = parser.parse_args()

    if not args.skip_branch_check:
        branch = current_branch()
        if args.mode == "dev":
            if branch != "main":
                print(
                    f"error: mode=dev must run on main; current branch is {branch!r}",
                    file=sys.stderr,
                )
                return 2
        elif not branch.startswith("release/v"):
            print(
                f"error: mode={args.mode} must run on a release/v* branch; "
                f"current is {branch!r}",
                file=sys.stderr,
            )
            return 2

    current = read_current_version()
    try:
        next_version = compute_next(current, args.mode)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    tag = f"v{next_version}"
    if tag in existing_tags():
        print(
            f"error: tag {tag} already exists; refusing to overwrite", file=sys.stderr
        )
        return 2

    if args.dry_run:
        print(next_version)
        return 0

    write_pyproject(next_version)
    # Override UV_FROZEN inherited from the workflow env: frozen mode rejects
    # lockfile updates, but every bump changes the package entry.
    subprocess.run(
        ["uv", "lock", "--upgrade-package", "mellea"],
        cwd=REPO_ROOT,
        check=True,
        env={**os.environ, "UV_FROZEN": "0"},
    )
    subprocess.run(
        ["git", "add", "pyproject.toml", "uv.lock"], cwd=REPO_ROOT, check=True
    )
    subprocess.run(
        ["git", "commit", "-m", f"release: bump version to {next_version} [skip ci]"],
        cwd=REPO_ROOT,
        check=True,
    )

    print(next_version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
