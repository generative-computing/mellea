import pytest

from mellea.backends import Backend
from mellea.backends.ollama import OllamaModelBackend
from mellea.backends.tools import add_tools_from_model_options
from mellea.backends.types import ModelOption
from mellea.stdlib.base import ModelOutputThunk
from mellea.stdlib.docs.richdocument import Table
from mellea.stdlib.session import LinearContext, MelleaSession


@pytest.fixture(scope="module")
def m() -> MelleaSession:
    return MelleaSession(
        backend=OllamaModelBackend(),
        ctx=LinearContext(),
    )


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

def test_add_tools_from_model_options(table: Table):
    def get_weather(location: str) -> int:
        """Returns the weather in Celsius."""
        return 21

    model_options = {
        ModelOption.TOOLS: [get_weather, table.content_as_string]
    }

    tools = {}
    add_tools_from_model_options(tools, model_options)

    assert tools["get_weather"] == get_weather

    # Must use `==` for bound methods.
    assert tools["content_as_string"] == table.content_as_string, f"{tools["content_as_string"]} is not {table.content_as_string}"

def test_tool_called(m: MelleaSession, table: Table):
    """We don't force tools to be called. As a result, this test might unexpectedly fail."""
    r = 10

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

    returned_no_tool = False
    for i in range(r):
        transformed = m.transform(table, "output a text description of this table")
        if isinstance(transformed, ModelOutputThunk):
            returned_no_tool = True
            break

    assert (
        returned_no_tool
    ), f"only returned tools after {r} attempts, should've returned a response with no tools"

if __name__ == "__main__":
    pytest.main([__file__])
