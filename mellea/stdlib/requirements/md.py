"""This file contains various requirements for Markdown-formatted files."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...core import Context, Requirement

if TYPE_CHECKING:
    import mistletoe

_mistletoe = None


def _get_mistletoe():
    global _mistletoe
    if _mistletoe is None:
        import mistletoe as mt

        _mistletoe = mt
    return _mistletoe


# region lists


def as_markdown_list(ctx: Context) -> list[str] | None:
    """Attempts to format the last_output of the given context as a markdown list."""
    mistletoe = _get_mistletoe()
    xs = list()
    raw_output = ctx.last_output()
    assert raw_output is not None
    try:
        assert raw_output.value is not None
        parsed = mistletoe.Document(raw_output.value)
        assert parsed.children is not None
        children = list(parsed.children)
        for child in children:
            if type(child) is not mistletoe.block_token.List:
                return None
        assert child.children is not None
        for item in child.children:
            xs.append(mistletoe.base_renderer.BaseRenderer().render(item))
        return xs
    except Exception:
        return None


def _md_list(ctx: Context):
    return as_markdown_list(ctx) is not None


is_markdown_list = Requirement(
    description="The response should be formatted as a Markdown list.",
    validation_fn=_md_list,
)


# endregion

# region tables


def _md_table(ctx: Context):
    mistletoe = _get_mistletoe()
    raw_output = ctx.last_output()
    assert raw_output is not None
    try:
        assert raw_output.value is not None
        parsed = mistletoe.Document(raw_output.value)
        assert parsed.children is not None
        children = list(parsed.children)
        if len(children) != 1:
            return False
        return type(children[0]) is mistletoe.block_token.Table
    except Exception:
        return False


is_markdown_table = Requirement(
    description="The output should be formatted as a Markdown table.",
    validation_fn=_md_table,
)


# endregion
