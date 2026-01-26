import pytest

from mellea.core import CBlock, ModelOutputThunk, default_output_to_bool
from mellea.stdlib.context import ChatContext, Context
from mellea.stdlib.requirements import (
    as_markdown_list,
    is_markdown_list,
    is_markdown_table,
)


def from_model(s: str) -> Context:
    ctx = ChatContext()
    ctx = ctx.add(ModelOutputThunk(value=s, meta={"test": True}))
    return ctx


MARKDOWN_LIST_CTX: Context = from_model(
    """
 * This is the first item
 * This is the second item
 * This is the third item
"""
)

MARKDOWN_OL_LIST_CTX: Context = from_model(
    """
1. The first item is the most important.
2. The second item is not as important.
3. Silly things get buried in the middle.
4. But the last item we save for being also important.
"""
)


MARKDOWN_TABLE_CONTEXT: Context = from_model(
    """
| Month    | Revenue |
| -------- | ------- |
| January  | $100    |
| February | $900    |
| March    | $14     |
"""
)


async def test_markdown_list():
    result = await is_markdown_list.validate(None, MARKDOWN_LIST_CTX)  # type: ignore
    assert result
    assert len(as_markdown_list(MARKDOWN_LIST_CTX)) == 3  # type: ignore
    assert len(as_markdown_list(MARKDOWN_OL_LIST_CTX)) == 4  # type: ignore
    assert type(as_markdown_list(MARKDOWN_OL_LIST_CTX)[0]) is str  # type: ignore
    result = await is_markdown_list.validate(None, MARKDOWN_OL_LIST_CTX)  # type: ignore
    assert result


async def test_markdown_table():
    result = await is_markdown_table.validate(None, MARKDOWN_TABLE_CONTEXT)  # type: ignore
    assert result


def test_default_output_to_bool_yes():
    assert default_output_to_bool("yeS") == True


def test_default_output_to_bool_no():
    assert default_output_to_bool("nO") == False


def test_default_output_to_bool_complicated_yes():
    assert (
        default_output_to_bool(
            CBlock("The requirement is met by the output. Therefore, my answer is yes.")
        )
        == True
    )


def test_default_output_to_bool_word_with_yes_in_it():
    assert (
        default_output_to_bool("Here's a word that meets those requirements: ayes.")
        == False
    )


if __name__ == "__main__":
    pytest.main([__file__])
