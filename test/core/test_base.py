from typing import Any

import pytest

from mellea.core import CBlock, Component, ModelOutputThunk
from mellea.stdlib.components import Message


def test_cblock():
    cb = CBlock(value="This is some text")
    str(cb)
    repr(cb)
    assert str(cb) == "This is some text"


def test_cblpock_meta():
    cb = CBlock("asdf", meta={"x": "y"})
    assert str(cb) == "asdf"
    assert cb._meta["x"] == "y"


def test_component():
    class _ClosuredComponent(Component[str]):
        def parts(self):
            return []

        def format_for_llm(self) -> str:
            return ""

        def _parse(self, computed: ModelOutputThunk) -> str:
            return ""

    c = _ClosuredComponent()
    assert len(c.parts()) == 0


def test_parse():
    class _ChatResponse:
        def __init__(self, msg: Message) -> None:
            self.message = msg

    source = Message(role="user", content="source message")
    result = ModelOutputThunk(
        value="result value",
        meta={
            "chat_response": _ChatResponse(
                Message(role="assistant", content="assistant reply")
            )
        },
    )

    result.parsed_repr = source.parse(result)
    assert isinstance(result.parsed_repr, Message), (
        "result's parsed repr should be a message when meta includes a chat_response"
    )
    assert result.parsed_repr.role == "assistant", (
        "result's parsed repr role should be assistant"
    )
    assert result.parsed_repr.content == "assistant reply"

    result = ModelOutputThunk(value="result value")
    result.parsed_repr = source.parse(result)
    assert isinstance(result.parsed_repr, Message), (
        "result's parsed repr should be a message when source component is a message"
    )
    assert result.parsed_repr.content == "result value"


if __name__ == "__main__":
    pytest.main([__file__])
