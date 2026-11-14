# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for assistant tool-call turns replayed through chat history.

`Instruction._parse` and `ReactThought._parse` reduce a model turn to its text
value, so `ChatFormatter.to_chat_messages` rebuilt those turns as plain
assistant messages and dropped `tool_calls`. A following `tool` message then
referenced a call the provider never saw, which strict providers reject with a
400 on the second turn of a tool loop.
"""

from mellea.backends.tools import MelleaTool
from mellea.core.base import ModelOutputThunk, ModelToolCall, RawProviderResponse
from mellea.formatters.template_formatter import TemplateFormatter as ChatFormatter
from mellea.stdlib.components.chat import ToolMessage
from mellea.stdlib.components.instruction import Instruction

TOOL_CALLS_WIRE = [
    {
        "id": "call_1",
        "type": "function",
        "function": {"name": "weather", "arguments": '{"city": "Paris"}'},
    }
]


def _tool_call_thunk() -> ModelOutputThunk:
    """Build an `instruct()`-style thunk whose parsed repr keeps only the text.

    Returns:
        A computed `ModelOutputThunk` carrying one recorded tool call and an
        OpenAI-shaped raw response, mirroring what a backend stores on the
        thunk after a tool-calling completion.
    """

    def weather(city: str) -> str:
        """Return the current weather for a city.

        Args:
            city (str): City name to look up.

        Returns:
            A short weather summary.
        """
        return f"Sunny in {city}"

    thunk = ModelOutputThunk(
        value="",
        parsed_repr=Instruction("What is the weather in Paris?"),
        tool_calls=[
            ModelToolCall(
                name="weather",
                func=MelleaTool.from_callable(weather),
                args={"city": "Paris"},
                tool_call_id="call_1",
            )
        ],
    )
    thunk.raw = RawProviderResponse(
        provider="openai",
        response={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": TOOL_CALLS_WIRE,
                    }
                }
            ]
        },
    )
    return thunk


def test_assistant_tool_call_turn_survives_replay_before_tool_message():
    """A tool-issuing assistant turn keeps its `tool_calls` when a tool message follows."""
    thunk = _tool_call_thunk()
    tool_message = ToolMessage(
        role="tool",
        content="18C",
        tool_output="18C",
        name="weather",
        args={"city": "Paris"},
        tool=thunk.tool_calls[0],
    )

    messages = ChatFormatter(model_id="test").to_chat_messages([thunk, tool_message])

    assert [message.role for message in messages] == ["assistant", "tool"]
    assert messages[0].tool_calls == TOOL_CALLS_WIRE


def test_no_dangling_tool_calls_without_tool_message():
    """A tool-issuing assistant turn without a following tool message stays text-only.

    Tool calls the caller never answered must not be replayed as dangling
    `tool_calls`; strict providers reject those as well.
    """
    thunk = _tool_call_thunk()

    messages = ChatFormatter(model_id="test").to_chat_messages([thunk])

    assert len(messages) == 1
    assert messages[0].role == "assistant"
    assert messages[0].tool_calls is None
