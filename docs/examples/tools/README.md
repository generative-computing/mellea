# Tool Calling Examples

This directory contains examples of using tool calling (function calling) with Mellea.

## Files

### interpreter_example.py
Comprehensive examples of using the code interpreter tool with LLMs.

**Key Features:**
- Direct code execution with `code_interpreter()`
- Tool integration with `m.instruct()`
- Forcing tool use with requirements
- Validating tool arguments
- Local vs. sandboxed execution

## Concepts Demonstrated

- **Tool Definition**: Creating tools for LLM use
- **Tool Calling**: Having LLMs call functions
- **Tool Requirements**: Forcing or validating tool use
- **Argument Validation**: Ensuring correct tool arguments
- **Code Execution**: Running generated code safely

## Basic Usage

### Direct Tool Use
```python
from mellea.stdlib.tools import code_interpreter

# Execute code directly
result = code_interpreter("print(1+1)")
print(result)  # Output: 2
```

### Tool with LLM
```python
from mellea import start_session
from mellea.backends import ModelOption
from mellea.stdlib.tools import local_code_interpreter

m = start_session()
result = m.instruct(
    "Make a plot of y=x^2",
    model_options={ModelOption.TOOLS: [local_code_interpreter]}
)
```

### Forcing Tool Use
```python
from mellea.stdlib.requirements import uses_tool

result = m.instruct(
    "Use the code interpreter to make a plot of y=x^2",
    requirements=[uses_tool(local_code_interpreter)],
    model_options={ModelOption.TOOLS: [local_code_interpreter]},
    tool_calls=True
)

# Access the tool call
code = result.tool_calls["local_code_interpreter"].args["code"]
print(f"Generated code:\n{code}")

# Execute the tool
exec_result = result.tool_calls["local_code_interpreter"].call_func()
```

### Validating Tool Arguments
```python
from mellea.stdlib.requirements import tool_arg_validator

result = m.instruct(
    "Use the code interpreter to make a plot of y=x^2",
    requirements=[
        uses_tool(local_code_interpreter),
        tool_arg_validator(
            "The plot should be written to /tmp/output.png",
            tool_name=local_code_interpreter,
            arg_name="code",
            validation_fn=lambda code: "/tmp/output.png" in code
        )
    ],
    model_options={ModelOption.TOOLS: [local_code_interpreter]},
    tool_calls=True
)
```

## Available Tools

### Code Interpreter
- `code_interpreter`: Execute Python code (sandboxed)
- `local_code_interpreter`: Execute Python code locally

### Custom Tools

Create custom tools by defining functions:
```python
def my_tool(arg1: str, arg2: int) -> str:
    """Tool description for the LLM."""
    return f"Processed {arg1} with {arg2}"

# Use in model_options
model_options={ModelOption.TOOLS: [my_tool]}
```

## Tool Requirements

### uses_tool
Ensures the LLM uses a specific tool:
```python
from mellea.stdlib.requirements import uses_tool
requirements=[uses_tool(my_tool)]
```

### tool_arg_validator
Validates tool arguments:
```python
from mellea.stdlib.requirements import tool_arg_validator

tool_arg_validator(
    description="Validation description",
    tool_name=my_tool,
    arg_name="arg1",
    validation_fn=lambda x: len(x) > 5
)
```

## Safety Considerations

- **Sandboxing**: Use `code_interpreter` for untrusted code
- **Validation**: Always validate tool arguments
- **Permissions**: Be careful with file system access
- **Resource Limits**: Set timeouts and memory limits

## Related Documentation

- See `mellea/stdlib/tools/interpreter.py` for interpreter implementation
- See `mellea/stdlib/requirements/tool_reqs.py` for tool requirements
- See `docs/dev/tool_calling.md` for architecture details
- See `test/backends/test_tool_calls.py` for more examples