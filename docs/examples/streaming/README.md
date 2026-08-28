# Real-Time Streaming of LLM Responses

This directory contains examples of streaming LLM responses in real-time, with support for chunked validation and progressive output.

## Prerequisites

These examples require a running Ollama instance:

```bash
ollama serve
```

## Examples

### Basic Streaming with Chunking

```bash
uv run validated_streaming.py
```

Demonstrates:
- Streaming token-by-token generation
- Sentence-level chunking via `stream()`
- Per-chunk validation with custom `stream_validate()` methods
- Accessing stream events via the STREAMING_EVENT hook
- Early exit on validation failure

### Word-Level Chunking

```bash
uv run word_chunking.py
```

Demonstrates:
- Word-based chunking strategy
- Progressive output with word boundaries
- Custom chunking implementations

### Paragraph-Level Chunking

```bash
uv run paragraph_chunking.py
```

Demonstrates:
- Paragraph-based chunking strategy
- Processing longer text segments
- Multi-paragraph validation

### Custom Chunking Strategies

```bash
uv run custom_chunking.py
```

Demonstrates:
- Implementing custom chunking logic
- Defining custom stream validators
- Advanced streaming patterns

### Events Across Multiple Concurrent Streams

```bash
uv run multi_stream_events.py
```

Demonstrates:
- Consuming `STREAMING_EVENT` hook events with a small plugin
- Correlating events across multiple concurrent streams by `streaming_id`

## Key Concepts

**Streaming**: Receive LLM output token-by-token in real-time instead of waiting for complete generation.

**Chunking**: Group tokens into meaningful units (words, sentences, paragraphs) for validation and processing.

**Stream Validation**: Apply requirements at chunk level for early exit—stop generation when a constraint is violated.

**Stream Events**: Process stream events through the `STREAMING_EVENT` hook to monitor generation progress:
- `ChunkEvent` — A new chunk of text
- `QuickCheckEvent` — Initial validation result
- `FullValidationEvent` — Complete validation after full generation
- `StreamingDoneEvent` — Generation complete
- `CompletedEvent` — Stream exited (success or not)
- `ErrorEvent` — An exception occurred mid-stream

## See Also

- [../async/](../async/) — Asynchronous patterns
- [../requirements/](../requirements/) — Custom validation requirements
- [../telemetry/](../telemetry/) — Monitoring streaming operations
