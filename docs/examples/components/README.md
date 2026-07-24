# Component Examples

This directory contains examples demonstrating Mellea's component system, particularly focusing on component ID-based tool prefixing for collision-free tool composition.

## Files

### `duplicate_tool_names.py`
**Main example** - Demonstrates how Mellea handles multiple components with identical tool names.

**What it shows:**
- Two components (`DatabaseComponent` and `SearchComponent`) both define a `query` tool
- Automatic tool prefixing using component IDs (`component_{ID}.query`)
- LLM receiving and using both prefixed tools
- Proper tool execution via Mellea's pipeline (enables telemetry)
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
- Tools extracted with ID-based prefixes: `component_1adeba40.query`, `component_1c611a00.query`
- LLM successfully calls both tools in a single prompt
- Tool calls executed via `_call_tools()` to enable telemetry recording
- Both tools' component IDs visible in `mellea.tool.calls` metrics

---

### `duplicate_tool_names_experiments.py`
**Advanced experiments** - Explores various scenarios and edge cases with component tool prefixing.

**Experiments:**
1. **Three Components with Same Tool Name** - Verify scaling beyond 2 components
2. **Tool Name Mapping Inspection** - Examine the extracted tool objects and structure
3. **Prefixing Stability** - Show that same instances get same IDs, new instances get different IDs
4. **Selective Tool Access** - Filter tools before passing to LLM
5. **Tool Deduplication** - Behavior when same component added multiple times
6. **LLM with Filtered Tools** - Call LLM with a subset of available tools

**Run it:**
```bash
uv run python docs/examples/components/duplicate_tool_names_experiments.py
uv run pytest docs/examples/components/duplicate_tool_names_experiments.py -v
```

**Key findings:**
- Component IDs are stable for the same instance (multi-turn ready)
- New component instances get new IDs (expected behavior)
- Duplicate component instances are gracefully handled with warnings
- Tool filtering works by subsetting the tools dict
- LLM respects prompt guidance even when given multiple tools

---

### `pattern2_context_and_tools.py`
**Pattern 2 demonstration** - Shows how to combine components in context with explicit tool passing for tool calling.

**What it shows:**
- Pattern 2 approach: Components in session context + explicit tools via `ModelOption.TOOLS`
- How to add context blocks and components to the session
- Separating concerns: context rendering vs. tool availability
- Proper tool execution via Mellea's pipeline (enables telemetry)
- Multi-turn stability with component ID-based prefixing

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

**Key concepts:**
- Pattern 1: Extract tools only (simple tool calling)
- Pattern 2: Components in context + explicit tools (full control)
- Both patterns use component ID-based prefixing
- Tools must be explicitly passed even when components are in context
- Each tool call recorded in `mellea.tool.calls` metric with component_id

---

### `telemetry_tool_calling_demo.py`
**Telemetry demonstration** - Shows how to enable and view tool calling metrics.

**What it shows:**
- How to enable telemetry with environment variables
- Setting up console exporter for metric viewing
- Tool calls executed via `_call_tools()` to trigger telemetry hooks
- JSON format of OpenTelemetry metrics
- Component ID extraction in `mellea.tool.calls` metric
- Multi-exporter setup (Console, OTLP, Prometheus)

**Run it (with telemetry):**
```bash
export MELLEA_METRICS_ENABLED=true
export MELLEA_METRICS_CONSOLE=true
uv run python docs/examples/components/telemetry_tool_calling_demo.py
```

**Run it (without telemetry):**
```bash
uv run python docs/examples/components/telemetry_tool_calling_demo.py
```

**Key outputs:**
- Tool calls listed with their component IDs
- OpenTelemetry JSON metrics showing `mellea.tool.calls` counter
- Each tool invocation tracked with:
  - `tool`: Full tool name (e.g., `component_203e1b50.query`)
  - `status`: `"success"` or `"failure"`
  - `component_id`: Extracted from tool name (e.g., `203e1b50`)

---

## Concepts

### Component ID-Based Prefixing

When multiple components define tools with the same name, Mellea prevents collisions by prefixing each tool name with its component ID:

```
Original:     query, query
Prefixed:     component_1adeba40.query, component_1c611a00.query
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
- `mellea/core/base.py` - `TemplateRepresentation` with component metadata
- `mellea/stdlib/functional.py` - `_call_tools()` implementation (executes tools via pipeline)
- `mellea/telemetry/metrics_plugins.py` - `ToolMetricsPlugin` (records tool metrics)
- `mellea/telemetry/metrics.py` - `record_tool_call()` function (telemetry recording)
- Tests: `test/backends/test_tool_helpers.py` - Unit tests for tool prefixing
