import pytest

from mellea.stdlib.base import CBlock, ModelOutputThunk, LegacyLinearContext, LegacyContext
from mellea.stdlib.reqlib.md import is_markdown_list, is_markdown_table, as_markdown_list
from mellea.stdlib.requirement import default_output_to_bool


def from_model(s: str) -> LegacyContext:
    ctx = LegacyLinearContext()
    ctx.insert(ModelOutputThunk(value=s, meta={"test": True}))
    return ctx


MARKDOWN_LIST_CTX: LegacyContext = from_model(
    """
 * This is the first item
 * This is the second item
 * This is the third item
"""
)

MARKDOWN_OL_LIST_CTX: LegacyContext = from_model(
    """
1. The first item is the most important.
2. The second item is not as important.
3. Silly things get buried in the middle.
4. But the last item we save for being also important.
"""
)


MARKDOWN_TABLE_CONTEXT: LegacyContext = from_model(
    """
| Month    | Revenue |
| -------- | ------- |
| January  | $100    |
| February | $900    |
| March    | $14     |
"""
)


def test_markdown_list():
    assert is_markdown_list.validate(None, MARKDOWN_LIST_CTX)
    assert len(as_markdown_list(MARKDOWN_LIST_CTX)) == 3
    assert len(as_markdown_list(MARKDOWN_OL_LIST_CTX)) == 4
    assert type(as_markdown_list(MARKDOWN_OL_LIST_CTX)[0]) is str
    assert is_markdown_list.validate(None, MARKDOWN_OL_LIST_CTX)


def test_markdown_table():
    assert is_markdown_table.validate(None, MARKDOWN_TABLE_CONTEXT)

def test_default_output_to_bool_yes():
    assert default_output_to_bool("yeS") == True

def test_default_output_to_bool_no():
    assert default_output_to_bool("nO") == False

def test_default_output_to_bool_complicated_yes():
    assert default_output_to_bool(CBlock("The requirement is met by the output. Therefore, my answer is yes.")) == True

def test_default_output_to_bool_word_with_yes_in_it():
    assert default_output_to_bool("Here's a word that meets those requirements: ayes.") == False

if __name__ == "__main__":
    pytest.main([__file__])
