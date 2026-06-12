# Asynchronous Mellea Workflows

This directory contains examples of using Mellea with asynchronous Python patterns, including streaming responses and concurrent operations.

## Prerequisites

These examples use async/await and require Python 3.11+. Most examples also need a running Ollama instance:

```bash
ollama serve
```

## Examples

### Basic Async Usage

```bash
uv run async-simple.py
```

Demonstrates:
- Async session initialization
- Using `ainstruct()` for asynchronous instruction execution
- Streaming responses with `ModelOption.STREAM`
- Lazy compute evaluation

### Async with Lazy Compute

```bash
uv run async-with-lazy-compute.py
```

Demonstrates:
- Combining async operations with lazy evaluation
- Creating ModelOutputThunk objects
- Deferred computation patterns

## Key Concepts

**Async Backend Operations**: Mellea backends support async methods (`ainstruct`, `aact`, `achat`) that allow concurrent execution without blocking.

**Streaming**: Set `ModelOption.STREAM: True` in model options to receive responses as they're generated, useful for real-time feedback or long-running operations.

**Lazy Compute**: Defer execution to later in your program using lazy evaluation patterns.

## See Also

- [../streaming/](../streaming/) — Real-time token streaming
- [../sessions/](../sessions/) — Session configuration
- [../telemetry/](../telemetry/) — Monitoring async operations
