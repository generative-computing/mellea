# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from mellea.backends import ModelOption
from mellea.backends.tools import (
    MelleaTool,
    add_tools_from_context_actions,
    add_tools_from_model_options,
    parse_tools,
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
    ftc1 = FakeToolComponentWithExtraTool()
    ftc2 = FakeToolComponent()

    ctx_actions = [CBlock("Hello"), ftc1, ftc2]
    tools = {}
    add_tools_from_context_actions(tools, ctx_actions)

    # Check that tools with the same name get properly overwritten in order of ctx.
    tool1 = tools["tool1"]._call_tool
    assert tool1 == ftc2.tool1, f"{tool1} should == {ftc2.tool1}"

    # Check that tools that aren't overwritten are still there.
    tool2 = tools["tool2"]._call_tool
    assert tool2 == ftc1.tool2, f"{tool2} should == {ftc1.tool2}"


# --- parse_tools: Granite XML function format (#1689) ---


def test_parse_tools_granite_xml_single_call():
    """The shape the granite-4.2 chat template instructs the model to emit."""
    raw = (
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nBoston\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    assert parse_tools(raw) == [("get_weather", {"city": "Boston"})]


def test_parse_tools_granite_xml_values_stay_raw_strings():
    """Values are returned as text; coercion against the schema happens later."""
    raw = (
        "<tool_call>\n<function=add>\n<parameter=x>\n3\n</parameter>\n"
        "<parameter=items>\n[1, 2]\n</parameter>\n</function>\n</tool_call>"
    )
    assert parse_tools(raw) == [("add", {"x": "3", "items": "[1, 2]"})]


def test_parse_tools_granite_xml_multiline_value_keeps_inner_newlines():
    """Only the newline the template wraps each value in is removed."""
    raw = (
        "<tool_call>\n<function=write_file>\n<parameter=body>\n"
        "line one\n  line two\n</parameter>\n</function>\n</tool_call>"
    )
    assert parse_tools(raw) == [("write_file", {"body": "line one\n  line two"})]


def test_parse_tools_granite_xml_multiple_calls_in_order():
    raw = (
        "<tool_call>\n<function=first>\n<parameter=a>\n1\n</parameter>\n"
        "</function>\n</tool_call>\n"
        "<tool_call>\n<function=second>\n</function>\n</tool_call>"
    )
    assert parse_tools(raw) == [("first", {"a": "1"}), ("second", {})]


def test_parse_tools_granite_xml_after_reasoning_text():
    """The template allows natural-language reasoning before the call."""
    raw = (
        "I need the weather first.\n\n<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nBoston\n</parameter>\n</function>\n</tool_call>"
    )
    assert parse_tools(raw) == [("get_weather", {"city": "Boston"})]


def test_parse_tools_granite_xml_json_value_is_not_a_second_call():
    """A parameter value that happens to look like a JSON tool call is data."""
    raw = (
        "<tool_call>\n<function=log_event>\n<parameter=payload>\n"
        '{"name": "add", "arguments": {"x": 1, "y": 2}}\n'
        "</parameter>\n</function>\n</tool_call>"
    )
    assert parse_tools(raw) == [
        ("log_event", {"payload": '{"name": "add", "arguments": {"x": 1, "y": 2}}'})
    ]


def test_parse_tools_json_inside_tool_call_tags_still_parses():
    """The Granite 4.0/4.1 shape: JSON inside `<tool_call>` tags."""
    raw = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Boston"}}\n</tool_call>'
    assert parse_tools(raw) == [("get_weather", {"city": "Boston"})]


if __name__ == "__main__":
    pytest.main([__file__])
