#!/usr/bin/env python3
# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check (or fix) SPDX license headers on code files.

Every code file must begin with the two-line header (after an optional shebang):

    Copyright IBM Corp. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

rendered in the file's native comment syntax. This script is stdlib-only so it can
run in CI without installing dependencies.

Scope (matches what the headers were originally applied to):
- Included extensions: .py .sh .ts .tsx .js .mjs .css
- Excluded: everything under docs/, all Jinja templates, and non-code file types.

Usage:
    python3 .github/scripts/check_license_headers.py --check   # CI: exit 1 if any missing
    python3 .github/scripts/check_license_headers.py --fix     # insert/upgrade in place

Optional file paths may be passed positionally (as pre-commit does). When given,
only those files are checked/fixed; otherwise all git-tracked code files are used.
In --fix mode the script exits non-zero if it changed any file, so pre-commit
blocks the commit and the user re-stages the headers.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

COPYRIGHT = "Copyright IBM Corp. All Rights Reserved."
SPDX = "SPDX-License-Identifier: Apache-2.0"

# Comment prefix per extension family.
HASH_EXTS = {".py", ".sh"}
SLASH_EXTS = {".ts", ".tsx", ".js", ".mjs"}
CSS_EXTS = {".css"}
ALL_EXTS = HASH_EXTS | SLASH_EXTS | CSS_EXTS

# Paths (relative to repo root) whose subtrees are exempt from the header rule.
EXCLUDED_PREFIXES = ("docs/",)

# How many leading lines to scan for the header (tolerates a shebang + blank line).
SCAN_LINES = 6


def tracked_files() -> list[Path]:
    """Return git-tracked files in scope: target extensions, minus excluded subtrees."""
    patterns = [f"*{ext}" for ext in ALL_EXTS]
    out = subprocess.run(
        ["git", "ls-files", "-z", *patterns],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: list[Path] = []
    for rel in out.split("\0"):
        if not rel:
            continue
        if rel.startswith(EXCLUDED_PREFIXES):
            continue
        result.append(REPO / rel)
    return result


def select_files(paths: list[str]) -> list[Path]:
    """Return in-scope files for the given paths, or all tracked files if none.

    Filters passed paths to the target extensions and excluded subtrees so the
    hook behaves the same on a staged subset as on the whole tree.
    """
    if not paths:
        return tracked_files()
    result: list[Path] = []
    for raw in paths:
        p = Path(raw)
        abs_p = p if p.is_absolute() else REPO / p
        if not abs_p.is_file() or abs_p.suffix not in ALL_EXTS:
            continue
        try:
            rel = abs_p.resolve().relative_to(REPO).as_posix()
        except ValueError:
            continue
        if rel.startswith(EXCLUDED_PREFIXES):
            continue
        result.append(abs_p)
    return result


def header_lines(ext: str) -> list[str]:
    """Build the header lines (no trailing newlines) for a file extension."""
    if ext in CSS_EXTS:
        return [f"/* {COPYRIGHT} */", f"/* {SPDX} */"]
    prefix = "//" if ext in SLASH_EXTS else "#"
    return [f"{prefix} {COPYRIGHT}", f"{prefix} {SPDX}"]


def has_header(lines: list[str]) -> bool:
    """True if both the copyright and SPDX lines appear near the top of the file."""
    head = "\n".join(lines[:SCAN_LINES])
    return COPYRIGHT in head and SPDX in head


def fix_file(path: Path) -> str:
    """Insert or upgrade the header in a single file. Returns an action string."""
    ext = path.suffix
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    had_trailing_newline = text.endswith("\n")
    if had_trailing_newline and lines and lines[-1] == "":
        lines = lines[:-1]

    hdr = header_lines(ext)

    if has_header(lines):
        return "ok"

    # Upgrade: a bare one-line SPDX with no copyright line — replace it with the pair,
    # keeping the comment style already present on that line.
    for i, line in enumerate(lines[:SCAN_LINES]):
        if SPDX in line and COPYRIGHT not in line:
            lines[i : i + 1] = hdr
            new_text = "\n".join(lines) + ("\n" if had_trailing_newline else "")
            path.write_text(new_text, encoding="utf-8")
            return "upgraded"

    # Fresh insert, after a shebang if present.
    insert_at = 1 if lines and lines[0].startswith("#!") else 0
    block = list(hdr)
    following = lines[insert_at] if insert_at < len(lines) else ""
    if following.strip() != "":
        block.append("")
    lines[insert_at:insert_at] = block
    new_text = "\n".join(lines) + ("\n" if had_trailing_newline else "")
    path.write_text(new_text, encoding="utf-8")
    return "inserted"


def run_check(files: list[Path]) -> int:
    """Report files missing the header. Return process exit code."""
    missing = [
        p for p in files if not has_header(p.read_text(encoding="utf-8").split("\n"))
    ]
    if not missing:
        print(f"All {len(files)} code files have SPDX license headers.")
        return 0

    print(
        f"{len(missing)} file(s) are missing the SPDX license header:\n",
        file=sys.stderr,
    )
    for p in missing:
        print(f"  {p.relative_to(REPO)}", file=sys.stderr)
    print(
        "\nEvery code file must begin with (after any shebang):\n\n"
        f"    {COPYRIGHT}\n"
        f"    {SPDX}\n\n"
        "in the file's comment syntax (# for .py/.sh, // for .ts/.tsx/.js/.mjs, /* */ for .css).\n\n"
        "Fix locally with:\n\n"
        "    python3 .github/scripts/check_license_headers.py --fix\n",
        file=sys.stderr,
    )
    return 1


def run_fix(files: list[Path]) -> int:
    """Insert/upgrade headers in place. Exit non-zero if any file was changed."""
    counts: dict[str, int] = {}
    for p in files:
        action = fix_file(p)
        counts[action] = counts.get(action, 0) + 1
        if action != "ok":
            print(f"{action:9} {p.relative_to(REPO)}")
    changed = counts.get("inserted", 0) + counts.get("upgraded", 0)
    print(
        f"\nDone: {changed} file(s) updated "
        f"({counts.get('inserted', 0)} inserted, {counts.get('upgraded', 0)} upgraded), "
        f"{counts.get('ok', 0)} already compliant."
    )
    # As an auto-fixer, exit non-zero when files were modified so pre-commit fails
    # the run and the user re-stages the now-headered files.
    return 1 if changed else 0


def main() -> int:
    """Parse args and dispatch to check or fix."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check",
        action="store_true",
        help="Report files missing the header and exit non-zero (default).",
    )
    group.add_argument(
        "--fix", action="store_true", help="Insert or upgrade headers in place."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional files to check/fix (pre-commit passes these). "
        "Defaults to all git-tracked code files.",
    )
    args = parser.parse_args()

    files = select_files(args.paths)
    if args.fix:
        return run_fix(files)
    return run_check(files)


if __name__ == "__main__":
    sys.exit(main())
