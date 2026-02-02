import pytest
from langchain_core.tools import Tool, tool  # type: ignore[import-not-found]
from pydantic_core import ValidationError

import mellea
from mellea.backends.model_options import ModelOption
from mellea.backends.tools import MelleaTool
from mellea.stdlib.session import MelleaSession


def callable(input: int) -> str:
    """Common callable to test tool functionality."""
    return str(input)


@tool
def langchain_tool(input: int) -> str:
    """Common langchain tool to test functionality."""
    return str(input)


@pytest.fixture(scope="module")
def session():
    return mellea.start_session()


def test_from_callable():
    t = MelleaTool.from_callable(callable)
    assert isinstance(t, MelleaTool)
    assert t.name == callable.__name__

    name_override = "new_name"
    t = MelleaTool.from_callable(callable, name_override)
    assert t.name == name_override

    assert t.as_json_tool is not None
    expected_t_json = {
        "type": "function",
        "function": {
            "name": "new_name",
            "description": "Common callable to test tool functionality.",
            "parameters": {
                "type": "object",
                "required": ["input"],
                "properties": {"input": {"type": "integer", "description": ""}},
            },
        },
    }
    assert t.as_json_tool == expected_t_json

    assert t.run(1) == "1"
    assert t.run(input=2) == "2"


@pytest.mark.qualitative
@pytest.mark.ollama
@pytest.mark.llm
def test_from_callable_generation(session: MelleaSession):
    t = MelleaTool.from_callable(callable, "mellea_tool")

    out = session.instruct(
        "Call a mellea tool.",
        model_options={ModelOption.TOOLS: [t], ModelOption.SEED: 1},
        strategy=None,
        tool_calls=True,
    )

    assert out.tool_calls is not None, "did not call tool when expected"
    assert len(out.tool_calls.keys()) > 0

    tool = out.tool_calls[t.name]
    assert isinstance(tool.call_func(), str), "tool call did not yield expected type"


def test_from_langchain():
    t = MelleaTool.from_langchain(langchain_tool)
    assert isinstance(t, MelleaTool)
    assert t.name == "langchain_tool"

    expected_t_json = {
        "type": "function",
        "function": {
            "name": "langchain_tool",
            "description": "Common langchain tool to test functionality.",
            "parameters": {
                "properties": {"input": {"type": "integer"}},
                "required": ["input"],
                "type": "object",
            },
        },
    }
    assert t.as_json_tool == expected_t_json

    # This works for regular callables, but the indirection necessitated by langchain
    # means it doesn't work. That's okay; generated requests don't fit this format.
    with pytest.raises(ValidationError):
        assert t.run("1") == "1"

    assert t.run(input=2) == "2"


@pytest.mark.qualitative
@pytest.mark.ollama
@pytest.mark.llm
def test_from_langchain_generation(session: MelleaSession):
    t = MelleaTool.from_langchain(langchain_tool)

    out = session.instruct(
        "Call a langchain tool.",
        model_options={ModelOption.TOOLS: [t], ModelOption.SEED: 1},
        strategy=None,
        tool_calls=True,
    )

    assert out.tool_calls is not None, "did not call tool when expected"
    assert len(out.tool_calls.keys()) > 0

    tool = out.tool_calls[t.name]
    assert isinstance(tool.call_func(), str), "tool call did not yield expected type"


if __name__ == "__main__":
    pytest.main([__file__])
