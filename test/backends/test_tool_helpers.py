# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from mellea.backends import ModelOption
from mellea.backends.tools import (
    MelleaTool,
    add_tools_from_context_actions,
    add_tools_from_model_options,
)
from mellea.core import CBlock, Component, ModelOutputThunk, TemplateRepresentation


class FakeToolComponent(Component[str]):
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
            tools={self.tool1.__name__: MelleaTool.from_callable(self.tool1)},
        )

    def _parse(self, computed: ModelOutputThunk) -> str:
        return ""


class FakeToolComponentWithExtraTool(FakeToolComponent):
    def __init__(self) -> None:
        super().__init__()

    def tool2(self):
        return

    def format_for_llm(self) -> TemplateRepresentation:
        tr = super().format_for_llm()
        assert tr.tools is not None
        tr.tools[self.tool2.__name__] = MelleaTool.from_callable(self.tool2)
        return tr


def test_add_tools_from_model_options_list():
    def get_weather(location: str) -> int:
        """Returns the weather in Celsius."""
        return 21

    ftc = FakeToolComponent()
    model_options = {
        ModelOption.TOOLS: [
            MelleaTool.from_callable(t) for t in [get_weather, ftc.tool1]
        ]
    }

    tools = {}
    add_tools_from_model_options(tools, model_options)

    assert tools["get_weather"]._call_tool == get_weather

    # Must use `==` for bound methods.
    tool1 = tools["tool1"]._call_tool
    assert tool1 == ftc.tool1, f"{tool1} should == {ftc.tool1}"


def test_add_tools_from_model_options_map():
    def get_weather(location: str) -> int:
        """Returns the weather in Celsius."""
        return 21

    ftc = FakeToolComponent()
    model_options = {
        ModelOption.TOOLS: {
            get_weather.__name__: MelleaTool.from_callable(get_weather),
            ftc.tool1.__name__: MelleaTool.from_callable(ftc.tool1),
        }
    }

    tools = {}
    add_tools_from_model_options(tools, model_options)

    assert tools["get_weather"]._call_tool == get_weather

    # Must use `==` for bound methods.
    tool1 = tools["tool1"]._call_tool
    assert tool1 == ftc.tool1, f"{tool1} should == {ftc.tool1}"


def test_add_tools_from_context_actions():
    import re

    ftc1 = FakeToolComponentWithExtraTool()
    ftc2 = FakeToolComponent()

    # Extract component IDs before adding tools (ID is based on Python object identity)
    ftc1_id = hex(id(ftc1))[-8:]
    ftc2_id = hex(id(ftc2))[-8:]

    ctx_actions = [CBlock("Hello"), ftc1, ftc2]
    tools = {}
    add_tools_from_context_actions(tools, ctx_actions)

    # With auto-prefixing using component IDs, tools with the same name no longer collide.
    # Both are preserved with prefixed names: component_{ID}.tool1
    tool1_key_ftc1 = f"component_{ftc1_id}.tool1"
    tool1_key_ftc2 = f"component_{ftc2_id}.tool1"

    assert tool1_key_ftc1 in tools, f"Expected {tool1_key_ftc1} in tools"
    assert tool1_key_ftc2 in tools, f"Expected {tool1_key_ftc2} in tools"

    tool1_from_ftc1 = tools[tool1_key_ftc1]._call_tool
    assert tool1_from_ftc1 == ftc1.tool1, f"{tool1_from_ftc1} should == {ftc1.tool1}"

    tool1_from_ftc2 = tools[tool1_key_ftc2]._call_tool
    assert tool1_from_ftc2 == ftc2.tool1, f"{tool1_from_ftc2} should == {ftc2.tool1}"

    # Check that tools that aren't duplicated are still there with prefixed names.
    tool2_key = f"component_{ftc1_id}.tool2"
    assert tool2_key in tools, f"Expected {tool2_key} in tools"

    tool2 = tools[tool2_key]._call_tool
    assert tool2 == ftc1.tool2, f"{tool2} should == {ftc1.tool2}"

    # Verify that all tool prefixes match the expected ID pattern
    for tool_name in tools:
        if tool_name.startswith("component_"):
            assert re.match(r"component_[0-9a-f]{8}\.", tool_name), (
                f"Tool name {tool_name} does not match ID-based prefix pattern"
            )


if __name__ == "__main__":
    pytest.main([__file__])
