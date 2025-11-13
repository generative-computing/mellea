"""This file contains various requirements for Markdown-formatted files."""

import mistletoe

from mellea.stdlib.base import Context
from mellea.stdlib.requirement import Requirement

# region lists


def as_markdown_list(ctx: Context) -> list[str] | None:
    """Attempts to format the last_output of the given context as a markdown list."""
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
