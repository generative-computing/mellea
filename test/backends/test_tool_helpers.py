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
    # Both are preserved with prefixed names: component_{ID}__tool1
    tool1_key_ftc1 = f"component_{ftc1_id}__tool1"
    tool1_key_ftc2 = f"component_{ftc2_id}__tool1"

    assert tool1_key_ftc1 in tools, f"Expected {tool1_key_ftc1} in tools"
    assert tool1_key_ftc2 in tools, f"Expected {tool1_key_ftc2} in tools"

    tool1_from_ftc1 = tools[tool1_key_ftc1]._call_tool
    assert tool1_from_ftc1 == ftc1.tool1, f"{tool1_from_ftc1} should == {ftc1.tool1}"

    tool1_from_ftc2 = tools[tool1_key_ftc2]._call_tool
    assert tool1_from_ftc2 == ftc2.tool1, f"{tool1_from_ftc2} should == {ftc2.tool1}"

    # Check that tools that aren't duplicated are still there with prefixed names.
    tool2_key = f"component_{ftc1_id}__tool2"
    assert tool2_key in tools, f"Expected {tool2_key} in tools"

    tool2 = tools[tool2_key]._call_tool
    assert tool2 == ftc1.tool2, f"{tool2} should == {ftc1.tool2}"

    # Verify that all tool prefixes match the expected ID pattern
    for tool_name in tools:
        if tool_name.startswith("component_"):
            assert re.match(r"component_[0-9a-f]{8}__", tool_name), (
                f"Tool name {tool_name} does not match ID-based prefix pattern"
            )


def test_add_tools_from_context_actions_exceeds_length_limit(caplog):
    """Verify warning when tool name exceeds 64-character provider limit."""
    import logging

    # Create a custom component with a very long tool name
    # The tool name (key in tools dict) is what gets prefixed, so we need a 45+ char name
    # to exceed the 64 char limit (20 char prefix + 45+ char name = 65+ chars)
    class ComponentWithLongToolName(FakeToolComponent):
        def format_for_llm(self) -> TemplateRepresentation:
            long_tool_key = (
                "x" * 45
            )  # Will exceed 64 chars after prefixing (20 + 45 = 65)
            long_tool = MelleaTool.from_callable(lambda: None, name="tool")
            return TemplateRepresentation(
                obj=self, args={"arg": None}, tools={long_tool_key: long_tool}
            )

    component = ComponentWithLongToolName()
    tools = {}
    with caplog.at_level(logging.WARNING, logger="mellea"):
        add_tools_from_context_actions(tools, [component])

    # Verify warning was logged for length constraint
    assert any(
        "exceeds 64-character limit" in record.message for record in caplog.records
    ), f"Expected length warning in logs; got: {[r.message for r in caplog.records]}"


def test_add_tools_from_context_actions_invalid_characters(caplog):
    """Verify warning when tool name contains invalid characters."""
    import logging

    # Create a custom component with a tool name containing invalid characters
    class ComponentWithInvalidToolName(FakeToolComponent):
        def format_for_llm(self) -> TemplateRepresentation:
            # Invalid character: @ (not in [a-zA-Z0-9_-])
            # The key in tools dict is what gets prefixed, so use @ in the key
            invalid_tool_key = "my@tool"
            invalid_tool = MelleaTool.from_callable(lambda: None, name="tool")
            return TemplateRepresentation(
                obj=self, args={"arg": None}, tools={invalid_tool_key: invalid_tool}
            )

    component = ComponentWithInvalidToolName()
    tools = {}
    with caplog.at_level(logging.WARNING, logger="mellea"):
        add_tools_from_context_actions(tools, [component])

    # Verify warning was logged for invalid characters
    assert any("invalid characters" in record.message for record in caplog.records), (
        f"Expected invalid character warning in logs; got: {[r.message for r in caplog.records]}"
    )


if __name__ == "__main__":
    pytest.main([__file__])
