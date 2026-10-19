# Primitives Examples

Low-level functional primitives for fine-grained control over tool execution, context management, and plugin hooks without high-level session abstractions.

## Examples

| File | Description |
| :--- | :--- |
| [`call_tools_basic.py`](./call_tools_basic.py) | Manual tool execution using `call_tools()` and `acall_tools()` with `SimpleContext`. |
| [`call_tools_with_hooks.py`](./call_tools_with_hooks.py) | Intercepting, auditing, and modifying tool execution using plugin hooks (`TOOL_PRE_INVOKE`, `TOOL_POST_INVOKE`). |
