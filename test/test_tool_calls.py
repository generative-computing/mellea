import pytest

from mellea.backends import Backend
from mellea.backends.ollama import OllamaModelBackend
from mellea.backends.tools import add_tools_from_context_actions, add_tools_from_model_options
from mellea.backends.types import ModelOption
from mellea.stdlib.base import CBlock, Component, ModelOutputThunk, TemplateRepresentation
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

class FakeToolComponent(Component):
    def __init__(self) -> None:
        super().__init__()
    
    def tool1(self):
        return
    
    def parts(self):
        return []

    def format_for_llm(self) -> TemplateRepresentation:
        return TemplateRepresentation(
            obj=self,
            args={"arg": None},
            tools={
                self.tool1.__name__: self.tool1
            }
        )

class FakeToolComponentWithExtraTool(FakeToolComponent):
    def __init__(self) -> None:
        super().__init__()
    
    def tool2(self):
        return
    
    def format_for_llm(self) -> TemplateRepresentation:
        tr = super().format_for_llm()
        assert tr.tools is not None
        tr.tools[self.tool2.__name__] = self.tool2
        return tr


def test_add_tools_from_model_options_list():
    def get_weather(location: str) -> int:
        """Returns the weather in Celsius."""
        return 21

    ftc = FakeToolComponent()
    model_options = {
        ModelOption.TOOLS: [get_weather, ftc.tool1]
    }

    tools = {}
    add_tools_from_model_options(tools, model_options)

    assert tools["get_weather"] == get_weather

    # Must use `==` for bound methods.
    assert tools["tool1"] == ftc.tool1, f"{tools["tool1"]} should == {ftc.tool1}"


def test_add_tools_from_model_options_map():
    def get_weather(location: str) -> int:
        """Returns the weather in Celsius."""
        return 21

    ftc = FakeToolComponent()
    model_options = {
        ModelOption.TOOLS: {
            get_weather.__name__: get_weather,
            ftc.tool1.__name__: ftc.tool1
        }
    }

    tools = {}
    add_tools_from_model_options(tools, model_options)

    assert tools["get_weather"] == get_weather

    # Must use `==` for bound methods.
    assert tools["tool1"] == ftc.tool1, f"{tools["tool1"]} should == {ftc.tool1}"


def test_add_tools_from_context_actions():

    ftc1 = FakeToolComponentWithExtraTool()
    ftc2 = FakeToolComponent()

    ctx_actions = [CBlock("Hello"), ftc1, ftc2]
    tools = {}
    add_tools_from_context_actions(tools, ctx_actions)

    # Check that tools with the same name get properly overwritten in order of ctx.
    assert tools["tool1"] == ftc2.tool1, f"{tools["tool1"]} should == {ftc2.tool1}"

    # Check that tools that aren't overwritten are still there.
    assert tools["tool2"] == ftc1.tool2, f"{tools["tool2"]} should == {ftc1.tool2}"


def test_tool_called_from_context_action(m: MelleaSession, table: Table):
    """Make sure tools can be called from actions in the context."""
    r = 10
    m.ctx.reset()

    # Insert a component with tools into the context.
    m.ctx.insert(table)

    returned_tool = False
    for i in range(r):
        # Make sure the specific generate call is on a different action with
        # no tools to make sure it's a tool from the context.
        result = m.backend.generate_from_context(
            CBlock("Add a row to the table."),
            m.ctx,
            tool_calls=True
        )
        if result.tool_calls is not None and len(result.tool_calls) > 0:
            returned_tool = True
            break

    assert returned_tool, f"did not return a tool after {r} attempts"


def test_tool_called(m: MelleaSession, table: Table):
    """We don't force tools to be called. As a result, this test might unexpectedly fail."""
    r = 10
    m.ctx.reset()

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
    m.ctx.reset()

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
