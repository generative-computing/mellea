# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that links in the example documentation point at real files."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit

REPOSITORY_ROOT = Path(__file__).parents[1]
EXAMPLES_ROOT = REPOSITORY_ROOT / "docs" / "examples"

# The CLI page is generated during the documentation build and intentionally is not
# checked into the repository.
GENERATED_DOCS = {REPOSITORY_ROOT / "docs" / "docs" / "reference" / "cli.md"}
LINK_RE = re.compile(r"!?\[[^\]]*\]\(\s*(?:<([^>]+)>|([^\s)]+))")


def _local_links(path: Path):
    in_code_block = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        yield from (first or second for first, second in LINK_RE.findall(line))


def _missing_links() -> list[str]:
    missing = []
    for markdown_file in EXAMPLES_ROOT.rglob("*.md"):
        for target in _local_links(markdown_file):
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc:
                continue
            target_path = (markdown_file.parent / parsed.path).resolve()
            if target_path.exists() or target_path in GENERATED_DOCS:
                continue
            missing.append(f"{markdown_file.relative_to(REPOSITORY_ROOT)}: {target}")
    return missing


def test_example_markdown_links_resolve():
    """Prevent broken relative links from being added outside Docusaurus."""
    assert not (missing := _missing_links()), "Broken example links:\n" + "\n".join(
        missing
    )
