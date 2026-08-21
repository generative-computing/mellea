---
title: Choosing Between Primitives and High-Level APIs
description: Understand when to use call_tools/acall_tools versus act/instruct and MelleaSession.
---

Mellea provides multiple levels of abstraction for building generative applications. Understanding which to use depends on your needs.

## The Abstraction Hierarchy

```text
High-level (simplest)
│
├─ MelleaSession (session management + context + generation)
├─ instruct() / chat() (generation + context management)
├─ act() / aact() (generation + basic context management)
│
Low-level (most control)
├─ call_tools() / acall_tools() (tool execution only)
└─ generate_from_context() (raw model call)
```

Each level adds convenience by automating lower-level concerns.

## Comparison Table

| Aspect | `MelleaSession` | `act()`/`instruct()` | `call_tools()` |
| --- | --- | --- | --- |
| **Context management** | Automatic, stateful | Automatic, immutable | Manual |
| **Generation** | Built-in | Yes | No |
| **Tool call generation** | Built-in (via generation) | Built-in (via generation) | N/A |
| **Tool execution** | Manual (via loop or hooks) | Manual (you call `call_tools()`) | Manual |
| **Hook support** | Full | Full | Full |
| **Sampling/repair** | Via sampling strategy | Via sampling strategy | N/A |
| **Validation** | Via requirements | Via requirements | N/A |
| **Lines of code** | 5-10 | 10-20 | 20-50 |
| **Learning curve** | Low | Medium | High |
| **Flexibility** | Medium | High | Very high |

## Decision Tree

Start here to pick the right level:

```text
Does your code need to:

1. "Manage a multi-turn conversation"?
   YES → Use MelleaSession
   NO  → Continue

2. "Make a single LLM call with tools"?
   YES → Use act() or instruct()
   NO  → Continue

3. "Execute already-generated tool calls"?
   YES → Use call_tools()
   NO  → Continue

4. "Make a raw LLM call"?
   YES → Use backend.generate_from_context()
```

## Detailed Scenarios

### Scenario 1: Simple Single-Turn Chat

**Task**: User asks a question, model answers.

**Best choice**: `MelleaSession` or `chat()`

```python
# With MelleaSession (simplest)
with start_session() as m:
    result = m.chat("What is 2+2?")
    print(result.value)

# With chat() (explicit, no session state)
from mellea.stdlib.functional import chat
result, ctx = chat("What is 2+2?", ctx, backend)
```

✅ Why: Handles context automatically, minimal code.

---

### Scenario 2: Multi-Turn Conversation

**Task**: Build a chatbot that remembers previous messages.

**Best choice**: `MelleaSession`

```python
with start_session() as m:
    m.chat("My name is Alice")
    m.chat("What is my name?")  # Remembers "Alice"
    m.chat("What is 10 + 5?")
```

✅ Why: Context accumulates automatically, stateful by design.

---

### Scenario 3: Controlled Tool Execution

**Task**: Generate tool calls, then inspect and filter before execution.

**Best choice**: `act()` or `instruct()` + `call_tools()`

```python
# Generate with tools
result, ctx = instruct(
    "Calculate 2 + 3",
    ctx,
    backend,
    tool_calls=True,
    model_options={ModelOption.TOOLS: [add, multiply]}
)

# Inspect and filter before executing
safe_calls = [tc for tc in result.tool_calls if is_safe(tc)]
if safe_calls:
    # Execute only safe tool calls
    tool_messages = call_tools(result, backend)
    for msg in tool_messages:
        ctx = ctx.add(msg)
```

✅ Why: Generate with higher-level APIs, use `call_tools()` for execution control.

---

### Scenario 4: Custom ReACT Loop

**Task**: Implement "Reason → Act → Observe → Repeat" with custom logic.

**Best choice**: `call_tools()` + `act()` in a loop

```python
async def custom_react(goal, backend, tools):
    ctx = ChatContext().add(Message("user", goal))

    for step in range(max_steps):
        # Think
        result, ctx = await aact(
            system_prompt, ctx, backend, tool_calls=True
        )

        # Act (manual tool execution gives you control)
        tool_messages = await acall_tools(result, backend)

        # Observe
        for msg in tool_messages:
            ctx = ctx.add(msg)
            # Can inspect/transform results before adding to context
```

✅ Why: You control the loop flow, context management, and tool filtering.

---

### Scenario 5: Sampling/Validation Loop

**Task**: Generate, validate, and repair until requirements are met.

**Best choice**: `act()` with a `SamplingStrategy`

```python
from mellea.stdlib.sampling import RejectionSamplingStrategy

result = act(
    instruction,
    ctx,
    backend,
    requirements=[must_be_brief, must_be_json],
    strategy=RejectionSamplingStrategy(loop_budget=3),
)
```

✅ Why: `act()` handles the validate-repair loop, you just provide requirements.

---

### Scenario 6: Tool Execution with Plugin Hooks

**Task**: Log, audit, or intercept every tool call.

**Best choice**: Any level — hooks work everywhere

```python
@hook(HookType.TOOL_PRE_INVOKE)
async def audit(payload, _):
    log.info(f"Tool: {payload.model_tool_call.name}")

# Hooks fire regardless of whether you use call_tools(),
# act(), instruct(), or MelleaSession
```

✅ Why: Hooks are orthogonal to the API level — use them everywhere.

---

## Migration Patterns

### From Raw API Calls → `act()`

**Before** (manual context management):

```python
result, ctx = backend.generate_from_context(component, ctx=ctx)
ctx = ctx.add(result)
```

**After** (automatic context management):

```python
result, ctx = act(component, ctx, backend)
```

✅ Benefits: Cleaner, handles edge cases, telemetry included.

---

### From `act()` → `MelleaSession`

**Before** (passing context manually):

```python
result1, ctx = act(msg1, ctx, backend)
result2, act(msg2, ctx, backend)
result3, ctx = act(msg3, ctx, backend)
```

**After** (context managed automatically):

```python
with start_session() as m:
    result1 = m.act(msg1, backend)
    result2 = m.act(msg2, backend)
    result3 = m.act(msg3, backend)
```

✅ Benefits: No context threading, cleaner, easier to reason about.

---

### From `act()` → `call_tools()` for Specialized Loops

**Before** (basic tool generation):

```python
result, ctx = act(prompt, ctx, backend, tool_calls=True)
# Tool calls generated but not executed
# You'd need to manually call call_tools() to execute them
```

**After** (explicit execution control with inspection):

```python
result, ctx = act(prompt, ctx, backend, tool_calls=True)

# Inspect tool calls before execution
for tc in result.tool_calls:
    print(f"Will execute: {tc.name} with {tc.args}")

# Execute and add to context
tool_msgs = call_tools(result, backend)
for msg in tool_msgs:
    if is_approved(msg):  # Custom filtering
        ctx = ctx.add(msg)
```

✅ Benefits: Full control over tool execution timing, inspection, and filtering.

---

## Summary: Quick Pick

| You want to... | Use... |
| --- | --- |
| Build a chatbot | `MelleaSession` |
| Make a single generation | `act()` or `instruct()` |
| Validate & repair | `act()` + `SamplingStrategy` |
| Custom agentic loop | `call_tools()` + your loop |
| Just execute tools | `call_tools()` |
| Raw LLM call | `backend.generate_from_context()` |
| Observe all tool calls | Add `@hook(HookType.TOOL_PRE/POST_INVOKE)` |

## See Also

- [How-To: Act and Aact](act-and-aact.md) — `act()` and `aact()` in detail
- [How-To: Execute Tool Calls](execute-tool-calls.md) — `call_tools()` and `acall_tools()`
- [How-To: Use Context and Sessions](use-context-and-sessions.md) — `MelleaSession` and context management
