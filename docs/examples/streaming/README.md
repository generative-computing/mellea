# Real-Time Streaming of LLM Responses

This directory contains examples of streaming LLM responses in real-time, with support for chunked validation and progressive output.
For the concepts, see [Use Async and Streaming](../../docs/how-to/use-async-and-streaming.md)
and [Tutorial 06: Streaming Validation](../../docs/tutorials/06-streaming-validation.md).

## Examples

| File | Description |
| :--- | :--- |
| [`validated_streaming.py`](./validated_streaming.py) | Streaming token-by-token generation, sentence-level chunking via `stream()`, per-chunk validation with custom `_stream_validate()` methods, iterating typed stream events via `stream(as_events=True)`, and early exit on validation failure. |
| [`word_chunking.py`](./word_chunking.py) | Word-based chunking strategy, progressive output with word boundaries, and custom chunking implementations. |
| [`paragraph_chunking.py`](./paragraph_chunking.py) | Paragraph-based chunking strategy, processing longer text segments, and multi-paragraph validation. |
| [`custom_chunking.py`](./custom_chunking.py) | Implementing custom chunking logic, defining custom stream validators, and advanced streaming patterns. |
| [`per_requirement_chunking.py`](./per_requirement_chunking.py) | Each `Requirement` carrying its own `chunking=` strategy, independent of the stream's; sentence-level and word-level requirements validating one stream at once. |
| [`multi_stream_events.py`](./multi_stream_events.py) | Consuming `STREAMING_EVENT` hook events with a small plugin, and correlating events across multiple concurrent streams by `streaming_id`. |
| [`sampling_streaming.py`](./sampling_streaming.py) | Passing `strategy=` to `stream()` so an attempt that fails mid-stream is aborted and repaired, observing `RetryEvent`s, and reading per-attempt history from `streamer.attempts`. |

## See Also

- [../async/](../async/) — Asynchronous patterns
- [../requirements/](../requirements/) — Custom validation requirements
- [../telemetry/](../telemetry/) — Monitoring streaming operations
