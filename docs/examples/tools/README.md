---
title: Tool Calling Examples
description: Learn how to define and use tools with Mellea
---

This directory contains examples of using tool calling (function calling) with Mellea.

## Files

### interpreter_example.py

Comprehensive examples of using the code interpreter tool with LLMs.

**Key Features:**
- Direct code execution with `python_tool().run()`
- Tool integration with `m.instruct()`
- Forcing tool use with requirements
- Validating tool arguments
- Explicit tier selection for local execution

### smolagents_example.py
Shows how to use pre-built tools from Hugging Face's smolagents library.

**Key Features:**

- Loading existing smolagents tools (PythonInterpreterTool,
  WikipediaSearchTool, etc.)
- Converting to Mellea tools with `MelleaTool.from_smolagents()`
- Using tools from the Hugging Face ecosystem

### python_tool_example.py

Demonstrates creating and using custom Python tools with Mellea.

**Key Features:**

- Defining custom tool functions
- Tool registration and discovery
- Integrating custom tools with LLM calls

### tool_decorator_example.py

Shows how to use the `@tool` decorator for tool definition.

**Key Features:**

- Decorator-based tool creation
- Automatic argument parsing from type hints
- Tool documentation generation

## Concepts Demonstrated

- **Tool Definition**: Creating tools for LLM use
- **Tool Calling**: Having LLMs call functions
- **Tool Requirements**: Forcing or validating tool use
- **Argument Validation**: Ensuring correct tool arguments
- **Code Execution**: Running generated code safely

## Basic Usage

### Direct Tool Use

```python
from mellea.stdlib.tools import python_tool

# Create the tool
tool = python_tool(tier="local_unsafe", name="python")

# Execute code directly
result = tool.run(code="print(1+1)")
print(result.stdout)  # Output: 2
```

### Tool with LLM

```python
from mellea import start_session
from mellea.backends import ModelOption
from mellea.stdlib.tools import python_tool

m = start_session()
tool = python_tool(tier="local_unsafe", name="python")
result = m.instruct(
    "Make a plot of y=x^2",
    model_options={ModelOption.TOOLS: [tool]},
    tool_calls=True
)

# Print the tool calls
print("Tool calls:", result.tool_calls)

# Access the generated code
if result.tool_calls:
    code = result.tool_calls["python"].args["code"]
    print(f"Generated code:\n{code}")
```

### Forcing Tool Use
```python
from mellea import start_session
from mellea.backends import ModelOption
from mellea.stdlib.requirements import uses_tool
from mellea.stdlib.tools import python_tool

m = start_session()
tool = python_tool(tier="local_unsafe", name="python")
result = m.instruct(
    "Use the code interpreter to make a plot of y=x^2",
    requirements=[uses_tool("python")],
    model_options={ModelOption.TOOLS: [tool]},
    tool_calls=True
)

# Access the tool call
code = result.tool_calls["python"].args["code"]
print(f"Generated code:\n{code}")

# Execute the tool
exec_result = result.tool_calls["python"].call_func()
print(f"Execution success: {exec_result.success}")
print(f"Exit code: {exec_result.exit_code}")

# Check for generated artifacts (plots, images, etc.)
if exec_result.artifacts:
    print(f"Generated artifacts: {len(exec_result.artifacts)}")
    for artifact in exec_result.artifacts:
        print(f"  - {artifact.path}")
else:
    print("No artifacts generated (plot saved internally)")
```

### Validating Tool Arguments

```python
from mellea import start_session
from mellea.backends import ModelOption
from mellea.stdlib.requirements import tool_arg_validator, uses_tool
from mellea.stdlib.tools import python_tool

m = start_session()
tool = python_tool(tier="local_unsafe", name="python")
result = m.instruct(
    "Use the code interpreter to make a plot of y=x^2",
    requirements=[
        uses_tool("python"),
        tool_arg_validator(
            "The plot should be written to /tmp/output.png",
            tool_name="python",
            arg_name="code",
            validation_fn=lambda code: "/tmp/output.png" in code
        )
    ],
    model_options={ModelOption.TOOLS: [tool]},
    tool_calls=True
)

# Access the tool call
code = result.tool_calls["python"].args["code"]
print(f"Generated code:\n{code}")

# Verify the constraint was satisfied
if "/tmp/output.png" in code:
    print("\n✓ Code constraint satisfied: plot will be saved to /tmp/output.png")
else:
    print("\n✗ Code constraint NOT satisfied")

# Execute the tool
exec_result = result.tool_calls["python"].call_func()
print(f"Execution success: {exec_result.success}")
```

## Available Tools

### Code Interpreter
- `python_tool(tier="local_unsafe")`: Execute Python code locally (unrestricted subprocess)
- `python_tool(tier="docker_unsafe")`: Execute Python code in Docker (no resource limits)
- `python_tool(tier="docker")`: Execute Python code in Docker with capability policy

### Custom Tools

Create custom tools by defining functions:

```python
from mellea import start_session
from mellea.backends import ModelOption
from mellea.backends.tools import MelleaTool

def my_tool(arg1: str, arg2: int) -> str:
    """Tool description for the LLM."""
    return f"Processed {arg1} with {arg2}"

m = start_session()
result = m.instruct(
    "Use my_tool to process 'hello' with 5",
    model_options={ModelOption.TOOLS: [MelleaTool.from_callable(my_tool)]},
    tool_calls=True
)

# Access the tool call
print("Tool calls:", result.tool_calls)

if result.tool_calls:
    # Get the tool call details
    tool_call = result.tool_calls["my_tool"]
    print(f"Tool name: {tool_call.name}")
    print(f"Arguments: {tool_call.args}")

    # Execute the tool
    tool_result = tool_call.call_func()
    print(f"Tool result: {tool_result}")
```

## Tool Requirements

### uses_tool

Ensures the LLM uses a specific tool:

```python
from mellea.stdlib.requirements import uses_tool
requirements=[uses_tool("python")]
```

### tool_arg_validator

Validates tool arguments:

```python
from mellea.stdlib.requirements import tool_arg_validator

tool_arg_validator(
    description="Validation description",
    tool_name="my_tool",
    arg_name="arg1",
    validation_fn=lambda x: len(x) > 5
)
```

## Safety Considerations

- **Tier selection**: Pass `tier=` explicitly — `"local_unsafe"` runs code as an unrestricted subprocess; use `"docker"` for real isolation
- **Validation**: Always validate tool arguments
- **Permissions**: Be careful with file system access
- **Resource Limits**: Set timeouts and memory limits

## Related Documentation

- See `mellea/stdlib/tools/interpreter.py` for python_tool implementation
- See `mellea/stdlib/requirements/tool_reqs.py` for tool requirements
- See `test/backends/test_tool_calls.py` for more examples
