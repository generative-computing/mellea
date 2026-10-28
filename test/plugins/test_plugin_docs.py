# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks that the hook reference in the plugins docs covers every HookType."""

import re
from pathlib import Path

import pytest

from mellea.plugins.types import HookType

PLUGINS_DOC = (
    Path(__file__).parents[2] / "docs" / "docs" / "concepts" / "plugins.mdx"
)


@pytest.fixture(scope="module")
def doc_text() -> str:
    return PLUGINS_DOC.read_text(encoding="utf-8")


@pytest.mark.parametrize("hook", list(HookType), ids=lambda h: h.value)
def test_hook_has_reference_section(hook: HookType, doc_text: str) -> None:
    assert re.search(rf"^#### `{hook.value}`$", doc_text, re.MULTILINE), (
        f"plugins.mdx has no '#### `{hook.value}`' section"
    )


@pytest.mark.parametrize("hook", list(HookType), ids=lambda h: h.value)
def test_hook_has_policy_table_row(hook: HookType, doc_text: str) -> None:
    assert re.search(rf"^\| `{hook.value}` \|", doc_text, re.MULTILINE), (
        f"plugins.mdx policy table has no row for `{hook.value}`"
    )


def test_hook_count_matches_enum(doc_text: str) -> None:
    assert f"Enum with all {len(HookType)} hook types" in doc_text
