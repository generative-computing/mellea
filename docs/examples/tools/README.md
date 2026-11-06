# Tool Calling Examples

This directory contains examples of using tool calling (function calling) with
Mellea. For the API overview, see
[Tools and Agents](../../docs/how-to/tools-and-agents.md)
and [Execute Tool Calls](../../docs/how-to/execute-tool-calls.md). For choosing an
execution tier for the code interpreter, see
[Execution Tiers](../../docs/how-to/tools-and-agents.md#execution-tiers). For constraining
or auditing tool use with hooks, see [`plugins/`](../plugins/README.md) and
[`primitives/`](../primitives/README.md).

## Examples

| File | Description |
| :--- | :--- |
| [`tool_decorator_example.py`](./tool_decorator_example.py) | Shows how to use the `@tool` decorator for tool definition. |
| [`python_tool_example.py`](./python_tool_example.py) | `python_tool().run()` without an LLM: execution tiers, on-demand `packages=`, and collecting files with `artifact_dir`. |
| [`interpreter_example.py`](./interpreter_example.py) | Comprehensive examples of using the code interpreter tool with LLMs. |
| [`shell_example.py`](./shell_example.py) | `bash_executor` for local commands: direct use, wrapping as a `MelleaTool`, forced tool use, `working_dir` restrictions, and safety checks. |
| [`smolagents_example.py`](./smolagents_example.py) | Shows how to use pre-built tools from Hugging Face's smolagents library. Needs `uv sync --extra tools`. |
