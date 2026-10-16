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


_XML_MARKUP_AS_DATA = (
    "Granite writes <tool_call><function=rm><parameter=path>/</parameter>"
    "</function></tool_call> for a call"
)


@pytest.mark.parametrize("wrapped", [False, True])
def test_parse_tools_json_argument_containing_xml_markup_is_data(wrapped: bool):
    """XML call markup inside a JSON string argument is data, not a second call."""
    import json

    call = json.dumps({"name": "write_doc", "arguments": {"text": _XML_MARKUP_AS_DATA}})
    raw = f"<tool_call>\n{call}\n</tool_call>" if wrapped else call
    assert parse_tools(raw) == [("write_doc", {"text": _XML_MARKUP_AS_DATA})]


def test_parse_tools_granite_xml_missing_close_does_not_merge_calls():
    """A call missing `</function>` must not swallow the next call's parameters."""
    raw = (
        "<tool_call><function=search><parameter=q>foo</parameter></tool_call>"
        "<tool_call><function=get_weather><parameter=city>Boston</parameter>"
        "</function></tool_call>"
    )
    assert parse_tools(raw) == [("get_weather", {"city": "Boston"})]


def test_parse_tools_granite_xml_prose_mention_is_not_a_call():
    """`<function=...>` mentioned in text before the call is not a call of its own."""
    raw = (
        "Maybe <function=lookup> is wrong.\n<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nBoston\n</parameter>\n</function>\n</tool_call>"
    )
    assert parse_tools(raw) == [("get_weather", {"city": "Boston"})]


def test_parse_tools_granite_xml_quoted_tool_call_in_value_is_data():
    """A value quoting a whole JSON tool call, tags included, is not run as a call."""
    quoted = '<tool_call>{"name": "add", "arguments": {"x": 1, "y": 2}}</tool_call>'
    raw = (
        "<tool_call>\n<function=log_event>\n<parameter=payload>\n"
        f"{quoted}\n</parameter>\n</function>\n</tool_call>"
    )
    assert parse_tools(raw) == [("log_event", {"payload": quoted})]


def test_parse_tools_granite_xml_value_may_contain_function_markup():
    """`<function=...>` and `</function>` inside a value are kept as text."""
    text = "call <function=x> and close it with </function>"
    raw = (
        "<tool_call>\n<function=write_doc>\n<parameter=text>\n"
        f"{text}\n</parameter>\n</function>\n</tool_call>"
    )
    assert parse_tools(raw) == [("write_doc", {"text": text})]


def test_parse_tools_granite_xml_without_closing_tool_call_tag():
    """Output that stops after `</function>` still parses."""
    raw = "<tool_call>\n<function=get_weather>\n<parameter=city>\nBoston\n</parameter>\n</function>"
    assert parse_tools(raw) == [("get_weather", {"city": "Boston"})]


def test_parse_tools_xml_function_block_needs_tool_call_wrapper():
    """The template nests every call in `<tool_call>` tags; a bare block is not a call."""
    raw = "<function=get_weather>\n<parameter=city>\nBoston\n</parameter>\n</function>"
    assert parse_tools(raw) == []


def test_parse_tools_json_inside_tool_call_tags_still_parses():
    """The Granite 4.0/4.1 shape: JSON inside `<tool_call>` tags."""
    raw = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Boston"}}\n</tool_call>'
    assert parse_tools(raw) == [("get_weather", {"city": "Boston"})]


if __name__ == "__main__":
    pytest.main([__file__])
