# Tool Calling Examples

Examples demonstrating how to define tools, hand them to a model, and execute the
resulting tool calls in Mellea, including the code interpreter and shell tools. For
the API overview, see the [Tools and Agents how-to](../../docs/how-to/tools-and-agents.md)
and [Execute tool calls](../../docs/how-to/execute-tool-calls.md). For choosing an
execution tier for the code interpreter, see
[Execution tiers](../../docs/how-to/tools-and-agents.md#execution-tiers). For constraining
or auditing tool use with hooks, see [`plugins/`](../plugins/README.md) and
[`primitives/`](../primitives/README.md).

## Examples

| File | Description |
| :--- | :--- |
| [`tool_decorator_example.py`](./tool_decorator_example.py) | Defining tools with the `@tool` decorator: schema from type hints, `.run()`, custom names, and the older `MelleaTool.from_callable()` style. |
| [`python_tool_example.py`](./python_tool_example.py) | `python_tool().run()` without an LLM: execution tiers, on-demand `packages=`, and collecting files with `artifact_dir`. |
| [`interpreter_example.py`](./interpreter_example.py) | The code interpreter with a model: forcing the call with `uses_tool()` and constraining generated code with `tool_arg_validator()`. |
| [`shell_example.py`](./shell_example.py) | `bash_executor` for local commands: direct use, wrapping as a `MelleaTool`, forced tool use, `working_dir` restrictions, and safety checks. |
| [`smolagents_example.py`](./smolagents_example.py) | Wrapping a Hugging Face smolagents tool with `MelleaTool.from_smolagents()`. Needs `uv sync --extra tools`. |
