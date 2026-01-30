import pytest

from mellea.backends import ModelOption
from mellea.backends.ollama import OllamaModelBackend
from mellea.backends.tools import (
    AbstractMelleaTool,
    MelleaTool,
    add_tools_from_context_actions,
    add_tools_from_model_options,
)
from mellea.core import ModelOutputThunk
from mellea.stdlib.components.docs.richdocument import Table
from mellea.stdlib.context import ChatContext
from mellea.stdlib.session import MelleaSession


@pytest.fixture(scope="module")
def m() -> MelleaSession:
    return MelleaSession(backend=OllamaModelBackend(), ctx=ChatContext())


@pytest.fixture(scope="module")
def table() -> Table:
    t = Table.from_markdown(
        """| Month    | Savings |
| -------- | ------- |
| January  | $250    |
| February | $80     |
| March    | $420    |"""
    )
    assert t is not None, "test setup failed: could not create table from markdown"
    return t


def test_tool_called_from_context_action(m: MelleaSession, table: Table):
    """Make sure tools can be called from actions in the context."""
    m.reset()

    # Insert a component with tools into the context.
    m.ctx = m.ctx.add(table)

    # Create fake tools.
    def test1(): ...
    def test2(): ...

    model_opts = {
        ModelOption.TOOLS: [MelleaTool.from_callable(t) for t in [test1, test2]]
    }

    tools: dict[str, AbstractMelleaTool] = {}

    add_tools_from_model_options(tools, model_opts)
    assert "test1" in tools
    assert "test2" in tools

    add_tools_from_context_actions(tools, m.ctx.actions_for_available_tools())
    assert "to_markdown" in tools


def test_tool_called(m: MelleaSession, table: Table):
    """We don't force tools to be called. As a result, this test might unexpectedly fail."""
    r = 10
    m.reset()

    returned_tool = False
    for i in range(r):
        transformed = m.transform(table, "add a new row to this table")
        if isinstance(transformed, Table):
            returned_tool = True
            break

    assert returned_tool, f"did not return a tool after {r} attempts"


def test_tool_not_called(m: MelleaSession, table: Table):
    """Ensure tools aren't always called when provided."""
    r = 10
    m.reset()

    returned_no_tool = False
    for i in range(r):
        transformed = m.transform(table, "output a text description of this table")
        if isinstance(transformed, ModelOutputThunk):
            returned_no_tool = True
            break

    assert returned_no_tool, (
        f"only returned tools after {r} attempts, should've returned a response with no tools"
    )


if __name__ == "__main__":
    pytest.main([__file__])
