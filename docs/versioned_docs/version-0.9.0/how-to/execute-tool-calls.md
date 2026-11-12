---
title: Execute Tool Calls
description: Use call_tools and acall_tools to implement custom agentic loops with low-level tool execution control.
---

When building custom agentic patterns or advanced session management, you may need direct control over tool execution. Mellea provides `call_tools()` and `acall_tools()` primitives for this purpose.

## Overview

Most applications use high-level APIs like `act()`, `instruct()`, or the `MelleaSession` which handle context management and tool call generation. However, these APIs return the model's output—you must call `call_tools()` to execute the generated tool calls. When implementing a custom ReACT loop, multi-turn tool orchestration, or specialized context management, you can call `call_tools()` directly for fine-grained control.

The `call_tools()` function:

- Takes a model's tool call output and executes each tool
- Fires `TOOL_PRE_INVOKE` and `TOOL_POST_INVOKE` hooks for observability
- Returns a list of `ToolMessage` objects with results
- Does **not** manage context — you handle that yourself

## When to Use

Use `call_tools()` when you need:

- **Custom agentic loops**: Implementing ReACT or similar patterns with specialized control flow
- **Advanced context management**: Managing multiple contexts or non-linear conversation flows
- **Fine-grained tool execution control**: Filtering, transforming, or inspecting tool calls before execution
- **Tool execution hooks**: Plugins that must observe or modify every tool call with full lifecycle visibility

Use **higher-level APIs** (`act()`, `instruct()`, `chat()`) when you:

- Want automatic context management
- Don't need to inspect or transform tool calls before execution
- Prefer simpler, more declarative code (single function calls vs. explicit tool execution + context management)

## Basic Usage

```python
from mellea.stdlib.functional import call_tools
from mellea.stdlib.context import SimpleContext

# Step 1: Generate with tool calls enabled
result, ctx = instruct(
    "Use the calculator to compute 2 + 2",
    context,
    backend,
    tool_calls=True,  # Enable tool calling
)

# Step 2: Execute the tools
tool_messages = call_tools(result, backend)

# Step 3: Add results to context manually
for tool_message in tool_messages:
    ctx = ctx.add(tool_message)

# Step 4: Continue the conversation
next_result, ctx = instruct(
    "What was the result?",
    ctx,
    backend,
)
```

## Async Usage

For async code, use `acall_tools()`:

```python
from mellea.stdlib.functional import acall_tools

# In async context
tool_messages = await acall_tools(result, backend)
```

## Understanding Return Values

`call_tools()` returns `list[ToolMessage]`. Each `ToolMessage` contains:

- `name`: Tool name (str)
- `content`: Formatted output (str)
- `_tool_output`: Raw Python object returned by the tool
- `arguments`: Arguments passed to the tool (Mapping)
- `_tool`: The `ModelToolCall` that was executed

```python
tool_messages = call_tools(result, backend)

for msg in tool_messages:
    print(f"Tool: {msg.name}")
    print(f"Arguments: {msg.arguments}")
    print(f"Output: {msg._tool_output}")
    print(f"Content: {msg.content}")
```

## Custom Context Management

When using `call_tools()`, you manage context transitions yourself:

```python
# Start with an empty context
ctx = SimpleContext()

# Add initial message
ctx = ctx.add(Message("user", "What is the weather in Boston?"))

# Generate with tools enabled
result, ctx = aact(
    instruction,
    ctx,
    backend,
    tool_calls=True,
    await_result=True,
)

# Execute tools
tool_messages = await acall_tools(result, backend)

# Add all tool results to context
for tool_msg in tool_messages:
    ctx = ctx.add(tool_msg)

# Generate final response
final_result, ctx = aact(
    Message("assistant", ""),  # Placeholder for response synthesis
    ctx,
    backend,
)
```

## Hook Integration

Tool execution fires two hooks you can use with the plugin system:

### TOOL_PRE_INVOKE

Fires **before** tool execution. Use for:

- Validating or modifying arguments
- Implementing allowlists/denylists
- Logging or auditing

```python
from mellea.plugins import hook, HookType

@hook(HookType.TOOL_PRE_INVOKE)
async def validate_tool(payload, backend):
    if payload.model_tool_call.name not in ALLOWED_TOOLS:
        return block(f"Tool not allowed: {payload.model_tool_call.name}")
```

### TOOL_POST_INVOKE

Fires **after** tool execution. Use for:

- Processing or transforming results
- Logging execution metrics
- Error recovery

```python
@hook(HookType.TOOL_POST_INVOKE)
async def log_execution(payload, backend):
    print(f"Tool {payload.model_tool_call.name} took {payload.execution_time_ms}ms")
    if payload.error:
        print(f"Error: {payload.error}")
```

## Real-World Example: Simple ReACT

Here's a minimal ReACT implementation using `call_tools()`:

```python
from mellea.stdlib.functional import aact, acall_tools
from mellea.stdlib.components import Message
from mellea.stdlib.context import ChatContext

async def simple_react(goal: str, backend, tools: list, max_steps: int = 5):
    """Simple ReACT: Think → Act → Observe → Repeat"""
    ctx = ChatContext().add(Message("user", f"Goal: {goal}"))
    
    for step in range(max_steps):
        print(f"\n--- Step {step + 1} ---")
        
        # Think & Act: Generate with tool calls enabled
        result, ctx = await aact(
            Message("system", "Reason about the goal, then call a tool if needed."),
            ctx,
            backend,
            tool_calls=True,
            await_result=True,
        )
        print(f"Thought: {result.value[:200]}...")
        
        # Check for final answer
        if "FINAL ANSWER" in result.value:
            return result.value
        
        # Observe: Execute tools
        tool_messages = await acall_tools(result, backend)
        if not tool_messages:
            print("No tools called. Stopping.")
            break
        
        # Add observations to context
        for msg in tool_messages:
            ctx = ctx.add(msg)
            print(f"Observation: {msg.name} → {msg.content[:100]}...")
    
    return "Max steps reached"
```

## Comparison with Higher-Level APIs

| Feature | `call_tools()` | `act()` | `instruct()` |
| --- | --- | --- | --- |
| Context management | Manual | Automatic | Automatic |
| Tool call generation | N/A | Automatic | Automatic |
| Tool execution | Manual | Manual (you call `call_tools()`) | Manual (you call `call_tools()`) |
| Hook support | Yes | Yes | Yes |
| Telemetry | Yes | Yes | Yes |
| Validation/repair loop | No | Optional | Optional |
| Use case | Control execution flow | General purpose generation | Tasks & instructions |
| Complexity | Higher | Medium | Low |

## Migration Path

If you're currently using private `_call_tools` or `_acall_tools`, migrate to the public API:

```python
# Old (deprecated)
from mellea.stdlib.functional import _call_tools
result = _call_tools(mot, backend)

# New (public)
from mellea.stdlib.functional import call_tools
result = call_tools(mot, backend)
```

The old names still work (aliased to the new ones) but are deprecated. Plan to migrate within the next major release.

## See Also

- [How-To: Act and Aact](act-and-aact.md) — Higher-level generation primitives
- [How-To: Use Context and Sessions](use-context-and-sessions.md) — Context management strategies
- [How-To: Debug with Plugins](debug-with-plugins.md) — Using tool hooks for observability
- [GitHub Discussion #1460](https://github.com/generative-computing/mellea/discussions/1460) — Design discussion on this API promotion
