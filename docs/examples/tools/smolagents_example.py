# pytest: ollama, llm
"""Example showing how to use pre-built HuggingFace smolagents tools with Mellea.

This demonstrates loading existing tools from the smolagents ecosystem,
similar to how you can use langchain tools with MelleaTool.from_langchain().

The smolagents library provides various pre-built tools like:
- PythonInterpreterTool for code execution
- DuckDuckGoSearchTool for web search (requires ddgs package)
- WikipediaSearchTool for Wikipedia queries
- And many others from the HuggingFace ecosystem
"""

from mellea import start_session
from mellea.backends import ModelOption
from mellea.backends.tools import MelleaTool

try:
    # Import a pre-built tool from smolagents
    from smolagents import PythonInterpreterTool

    # Create the smolagents tool instance
    python_tool_hf = PythonInterpreterTool()

    # Convert to Mellea tool - now you can use it with Mellea!
    python_tool = MelleaTool.from_smolagents(python_tool_hf)

    # Use with Mellea session
    m = start_session()
    result = m.instruct(
        "Calculate the sum of numbers from 1 to 10 using Python",
        model_options={ModelOption.TOOLS: [python_tool]},
        tool_calls=True,
    )

    print(f"Response: {result}")

    if result.tool_calls:
        calc_result = result.tool_calls[python_tool.name].call_func()
        print(f"\nCalculation result: {calc_result}")

except ImportError as e:
    print("Please install smolagents: uv pip install 'mellea[smolagents]'")
    print(f"Error: {e}")
