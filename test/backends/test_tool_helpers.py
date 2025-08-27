
import pytest
from mellea.backends.tools import add_tools_from_context_actions, add_tools_from_model_options
from mellea.backends.types import ModelOption
from mellea.stdlib.base import CBlock, Component, TemplateRepresentation

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
    tool1 = tools['tool1']
    assert tool1 == ftc.tool1, f"{tool1} should == {ftc.tool1}"


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
    tool1 = tools['tool1']
    assert tool1 == ftc.tool1, f"{tool1} should == {ftc.tool1}"


def test_add_tools_from_context_actions():

    ftc1 = FakeToolComponentWithExtraTool()
    ftc2 = FakeToolComponent()

    ctx_actions = [CBlock("Hello"), ftc1, ftc2]
    tools = {}
    add_tools_from_context_actions(tools, ctx_actions)

    # Check that tools with the same name get properly overwritten in order of ctx.
    tool1 = tools['tool1']
    assert tool1 == ftc2.tool1, f"{tool1} should == {ftc2.tool1}"

    # Check that tools that aren't overwritten are still there.
    tool2 = tools["tool2"]
    assert tool2 == ftc1.tool2, f"{tool2} should == {ftc1.tool2}"

if __name__ == "__main__":
    pytest.main([__file__])
