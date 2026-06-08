# Mellea Plugins

This directory contains examples of extending Mellea with custom plugins—hooks for observability, modification, and control over the generation pipeline.

## Prerequisites

These examples require a running Ollama instance:

```bash
ollama serve
```

## Examples

### Quick Start

```bash
uv run quickstart.py
```

Demonstrates:
- Registering a single hook function
- Using `HookType.GENERATION_PRE_CALL` to log before LLM calls
- Minimal plugin setup (30 lines)

### Standalone Hooks

```bash
uv run standalone_hooks.py
```

Demonstrates:
- Function-based hooks without a class
- Token budget enforcement
- Generation latency monitoring
- Multiple hook types in one program

### Class-Based Plugins

```bash
uv run class_plugin.py
```

Demonstrates:
- Organizing hooks in a Plugin subclass
- PII protection patterns
- Input blocking before execution
- Output scanning (observe-only)

### Execution Modes

```bash
uv run execution_modes.py
```

Demonstrates:
- All five PluginMode execution strategies:
  - `SEQUENTIAL` — serial, can block and modify
  - `TRANSFORM` — serial, modify-only
  - `AUDIT` — serial, observe-only
  - `CONCURRENT` — parallel, can block
  - `FIRE_AND_FORGET` — background, observe-only

### Plugin Set Composition

```bash
uv run plugin_set_composition.py
```

Demonstrates:
- Grouping related hooks into PluginSets
- Global vs. per-session registration
- Organizing by concern (security, observability)

### Session-Scoped Plugins

```bash
uv run session_scoped.py
```

Demonstrates:
- Global hooks (fire for all sessions)
- Per-session plugins (fire only in specific session)
- Content policy enforcement by session

### Payload Modification

```bash
uv run payload_modification.py
```

Demonstrates:
- Using `modify()` to change payload fields
- Model copying with `model_copy(update={...})`
- Handling read-only vs. writable fields

### Tool Hooks

```bash
uv run tool_hooks.py
```

Demonstrates:
- `TOOL_PRE_INVOKE` and `TOOL_POST_INVOKE` hooks
- Tool allow-listing
- Argument validation and sanitization
- Tool call auditing

### Testing Plugins

```bash
uv run testing_plugins.py
```

Demonstrates:
- Unit-testing hooks without a live session
- Constructing payloads manually
- Direct hook invocation with `await`
- Testing blocking and pass-through behavior

## Hook Types

Mellea supports hooks at various stages of the generation pipeline:

- `GENERATION_PRE_CALL` — Before LLM call
- `GENERATION_POST_CALL` — After LLM call
- `SESSION_CREATE` — On session creation
- `SESSION_DESTROY` — On session cleanup
- (See `mellea.plugins.HookType` for complete list)

## Key Concepts

**Hooks**: Functions that execute at specific pipeline stages, allowing observation and modification of Mellea's behavior.

**Payload Modification**: Hooks can transform request/response payloads before/after LLM calls.

**Session Scope**: Plugins can be registered globally or per-session.

**Async Support**: Hooks can be synchronous or asynchronous.

## See Also

- [../telemetry/](../telemetry/) — Monitoring and metrics collection
- [../requirements/](../requirements/) — Custom validation logic
- Documentation: `mellea.plugins` module
