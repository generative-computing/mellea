"""This file contains various requirements for Markdown-formatted files."""

import mistletoe

from mellea.stdlib.base import LegacyContext
from mellea.stdlib.requirement import Requirement

# region lists


def as_markdown_list(ctx: LegacyContext) -> list[str] | None:
    xs = list()
    raw_output = ctx.last_output()
    assert raw_output is not None
    try:
        parsed = mistletoe.Document(raw_output.value)
        for child in parsed.children:
            if type(child) is not mistletoe.block_token.List:
                return None
        for item in child.children:
            xs.append(mistletoe.base_renderer.BaseRenderer().render(item))
        return xs
    except Exception:
        return None


def _md_list(ctx: LegacyContext):
    return as_markdown_list(ctx) is not None


is_markdown_list = Requirement(
    description="The response should be formatted as a Markdown list.",
    validation_fn=_md_list,
)


# endregion

# region tables


def _md_table(ctx: LegacyContext):
    raw_output = ctx.last_output()
    assert raw_output is not None
    try:
        parsed = mistletoe.Document(raw_output.value)
        if len(parsed.children) != 1:
            return False
        return type(parsed.children[0]) is mistletoe.block_token.Table
    except Exception:
        return False


is_markdown_table = Requirement(
    description="The output should be formatted as a Markdown table.",
    validation_fn=_md_table,
)


# endregion
