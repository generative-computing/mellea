# Asynchronous Mellea Workflows

This directory contains examples of using Mellea with asynchronous Python patterns, including streaming responses and concurrent operations.
For the concepts, see [Use Async and Streaming](../../docs/how-to/use-async-and-streaming.md)
and [Tutorial 02: Streaming and Async](../../docs/tutorials/02-streaming-and-async.md).

## Examples

| File | Description |
| :--- | :--- |
| [`async-simple.py`](./async-simple.py) | Async session initialization, `ainstruct()` for asynchronous instruction execution, streaming responses with `ModelOption.STREAM`, and lazy compute evaluation. |
| [`async-with-lazy-compute.py`](./async-with-lazy-compute.py) | Combining async operations with lazy evaluation, creating `ModelOutputThunk` objects, and deferred computation patterns. |
| [`async-iterator.py`](./async-iterator.py) | Consuming a streamed response with `async for delta in response`, using `async with` so an early exit cancels the in-flight generation. |

## See Also

- [../streaming/](../streaming/) — Real-time token streaming
- [../sessions/](../sessions/) — Session configuration
- [../telemetry/](../telemetry/) — Monitoring async operations
