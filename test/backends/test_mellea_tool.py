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


def test_from_huggingface_basic():
    """Test basic smolagents tool loading and schema conversion.

    This test verifies that:
    1. A smolagents tool can be wrapped as a MelleaTool
    2. The tool name is correctly extracted
    3. The schema is converted to OpenAI-compatible format
    4. The tool can be executed with arguments
    """
    try:
        from smolagents import Tool
    except ImportError:
        pytest.skip(
            "smolagents not installed - install with: uv pip install 'mellea[smolagents]'"
        )

    # Create a simple smolagents tool
    class SimpleTool(Tool):
        name = "simple_tool"
        description = "A simple test tool"
        inputs = {"text": {"type": "string", "description": "Input text"}}
        output_type = "string"

        def forward(self, text: str) -> str:
            return f"Processed: {text}"

    hf_tool = SimpleTool()
    mellea_tool = MelleaTool.from_huggingface(hf_tool)

    # Verify tool properties
    assert isinstance(mellea_tool, MelleaTool)
    assert mellea_tool.name == "simple_tool"

    # Verify schema conversion
    json_schema = mellea_tool.as_json_tool
    assert json_schema is not None
    assert "function" in json_schema
    assert json_schema["function"]["name"] == "simple_tool"
    assert json_schema["function"]["description"] == "A simple test tool"

    # Verify parameters are present
    assert "parameters" in json_schema["function"]
    params = json_schema["function"]["parameters"]
    assert "properties" in params
    assert "text" in params["properties"]

    # Verify tool execution
    result = mellea_tool.run(text="hello")
    assert result == "Processed: hello"


def test_from_huggingface_multiple_inputs():
    """Test smolagents tool with multiple input parameters."""
    try:
        from smolagents import Tool
    except ImportError:
        pytest.skip(
            "smolagents not installed - install with: uv pip install 'mellea[smolagents]'"
        )

    class MultiInputTool(Tool):
        name = "multi_input_tool"
        description = "Tool with multiple inputs"
        inputs = {
            "x": {"type": "integer", "description": "First number"},
            "y": {"type": "integer", "description": "Second number"},
            "operation": {"type": "string", "description": "Operation to perform"},
        }
        output_type = "integer"

        def forward(self, x: int, y: int, operation: str) -> int:
            if operation == "add":
                return x + y
            elif operation == "multiply":
                return x * y
            return 0

    hf_tool = MultiInputTool()
    mellea_tool = MelleaTool.from_huggingface(hf_tool)

    # Verify all parameters are in schema
    json_schema = mellea_tool.as_json_tool
    params = json_schema["function"]["parameters"]
    assert "x" in params["properties"]
    assert "y" in params["properties"]
    assert "operation" in params["properties"]

    # Verify tool execution with multiple args
    result = mellea_tool.run(x=5, y=3, operation="add")
    assert result == 8

    result = mellea_tool.run(x=5, y=3, operation="multiply")
    assert result == 15


def test_from_huggingface_invalid_tool():
    """Test error handling for non-smolagents tool objects."""
    try:
        from smolagents import Tool
    except ImportError:
        pytest.skip(
            "smolagents not installed - install with: uv pip install 'mellea[smolagents]'"
        )

    # Try to create tool from non-Tool object
    class NotATool:
        name = "fake"

    with pytest.raises(ValueError) as exc_info:
        MelleaTool.from_huggingface(NotATool())

    error_msg = str(exc_info.value)
    assert "smolagents Tool type" in error_msg


if __name__ == "__main__":
    pytest.main([__file__])
