# Component Examples

This directory contains examples demonstrating Mellea's component system, particularly focusing on component ID-based tool prefixing for collision-free tool composition.

## Files

### `duplicate_tool_names.py` (Private API)
**Reference example** - Demonstrates component ID-based tool prefixing using private APIs.

**⚠️ Note:** This example uses the private `_call_tools()` function from `mellea.stdlib.functional`. For new code, prefer the public API examples below.

**What it shows:**
- Two components (`DatabaseComponent` and `SearchComponent`) both define a `query` tool
- Automatic tool prefixing using component IDs (`component_{ID}__query`)
- Explicit tool extraction via `add_tools_from_context_actions()`
- Tool passing via `ModelOption.TOOLS` to the LLM
- Manual tool execution via private `_call_tools()` to enable telemetry
- Component ID extraction from prefixed tool names

**Run it:**
```bash
uv run python docs/examples/components/duplicate_tool_names.py
uv run pytest docs/examples/components/duplicate_tool_names.py -v
```

**View telemetry metrics:**
```bash
export MELLEA_METRICS_ENABLED=true
export MELLEA_METRICS_CONSOLE=true
uv run python docs/examples/components/duplicate_tool_names.py
```

**Key outputs:**
- Tools extracted with ID-based prefixes: `component_1adeba40__query`, `component_1c611a00__query`
- LLM successfully calls both tools in a single prompt
- Tool calls executed via private `_call_tools()` to enable telemetry recording

---

### `duplicate_tool_names_public_api.py` (Public API - Recommended)
**Modern example** - Demonstrates component ID-based tool prefixing using public APIs only.

**What it shows:**
- Two components with identical tool names handled via automatic prefixing
- Public `MelleaSession.act()` API with `tool_calls=True`
- Backend automatically extracts and prefixes tools (no manual extraction needed)
- Tool execution via public `call_tools()` API with telemetry support
- All public APIs, no private imports

**Important:** Uses `ChatContext` instead of `SimpleContext` (required for tool extraction from session context).

**Run it:**
```bash
uv run python docs/examples/components/duplicate_tool_names_public_api.py
uv run pytest docs/examples/components/duplicate_tool_names_public_api.py -v
```

**View telemetry metrics:**
```bash
export MELLEA_METRICS_ENABLED=true
export MELLEA_METRICS_CONSOLE=true
uv run python docs/examples/components/duplicate_tool_names_public_api.py
```

**Key points:**
- Uses public `act()` with `tool_calls=True` instead of `instruct()` + explicit tool passing
- **Requires `ChatContext`** — `SimpleContext` returns empty history and tools won't be extracted
- Backend auto-extracts tools from context components (no `ModelOption.TOOLS` needed)
- **Important:** `act()` does NOT automatically execute tools; call `call_tools()` explicitly to execute them and record telemetry
- Cleaner, more declarative API for handling multiple components

---

### `pattern2_context_and_tools.py` (Private API)
**Reference example** - Shows how to combine components in context with automatic tool extraction using private APIs.

**⚠️ Note:** This example uses the private `_call_tools()` function. For new code, prefer the public API example below.

**What it shows:**
- Pattern 2 approach: Components in session context with automatic tool extraction
- How to add components to the session context
- Backend automatically extracts tools via `add_tools_from_context_actions()` when `tool_calls=True`
- Proper tool execution via private `_call_tools()` (enables telemetry)
- Multi-turn stability with component ID-based prefixing
- Components with templates render in the conversation

**Run it:**
```bash
uv run python docs/examples/components/pattern2_context_and_tools.py
uv run pytest docs/examples/components/pattern2_context_and_tools.py -v
```

**View telemetry metrics:**
```bash
export MELLEA_METRICS_ENABLED=true
export MELLEA_METRICS_CONSOLE=true
uv run python docs/examples/components/pattern2_context_and_tools.py
```

---

### `pattern2_public_api.py` (Public API - Recommended)
**Modern example** - Shows Pattern 2 (components in context) using public APIs only.

**What it shows:**
- Components with templates live in session context
- Public `MelleaSession.act()` API with `tool_calls=True`
- Backend automatically extracts tools from all context components
- Tool execution via public `call_tools()` API with telemetry support
- Multi-component composition with stable component IDs
- All public APIs, no private imports

**Important:** Uses `ChatContext` instead of `SimpleContext` (required for tool extraction from session context).

**Run it:**
```bash
uv run python docs/examples/components/pattern2_public_api.py
uv run pytest docs/examples/components/pattern2_public_api.py -v
```

**View telemetry metrics:**
```bash
export MELLEA_METRICS_ENABLED=true
export MELLEA_METRICS_CONSOLE=true
uv run python docs/examples/components/pattern2_public_api.py
```

**Key concepts:**
- Pattern 1: Extract tools only (simple tool calling) → use `duplicate_tool_names_public_api.py`
- Pattern 2: Components in context with auto-extraction → use `pattern2_public_api.py`
- Both patterns use component ID-based prefixing
- **Requires `ChatContext`** — `SimpleContext` is stateless and won't extract tools from context
- Backend auto-extracts tools—NO explicit `ModelOption.TOOLS` needed
- **Important:** `act()` only extracts and passes tools to LLM; doesn't execute them. Call `_call_tools(response, backend)` to execute tool calls and trigger telemetry
- Components must have valid templates for rendering

---

## Concepts

### Component ID-Based Prefixing

When multiple components define tools with the same name, Mellea prevents collisions by prefixing each tool name with its component ID:

```
Original:     query, query
Prefixed:     component_1adeba40__query, component_1c611a00__query
```

**How it works:**
1. Each component instance gets a unique ID: `hex(id(object))[-8:]`
2. Tools from each component are extracted and prefixed
3. Prefixed names are collision-free
4. Same component instances always produce same IDs (stable for multi-turn)

---

## Key Takeaways

1. **Composability**: Multiple components can safely define tools with identical names
2. **Determinism**: Component IDs are stable within a session for the same instance
3. **Flexibility**: You can control which tools reach the LLM via filtering
4. **Scalability**: Works smoothly with 2, 3, or more components
5. **Observability**: Prefixed names and component IDs enable tracing and debugging

---

## Related Source Files

- `mellea/backends/tools.py` - `add_tools_from_context_actions()` implementation
- `mellea/stdlib/functional.py` - `_call_tools()` implementation (executes tools via pipeline)
- `mellea/telemetry/metrics_plugins.py` - `ToolMetricsPlugin` (records tool metrics)
- `mellea/telemetry/metrics.py` - `record_tool_call()` function (telemetry recording)
- Tests: `test/backends/test_tool_helpers.py` - Unit tests for tool prefixing
