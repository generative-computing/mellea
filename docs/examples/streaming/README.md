# Streaming Examples

This directory contains examples demonstrating Mellea's streaming capabilities.

## Examples

### basic_streaming.py
Basic example showing how to stream model outputs token by token. This is the simplest way to get started with streaming in Mellea.

**Run:**
```bash
uv run --with mellea docs/examples/streaming/basic_streaming.py
```

### interactive_chat.py
Interactive chat application with streaming responses. Shows how to build a conversational interface where the AI's responses appear incrementally.

**Run:**
```bash
uv run --with mellea docs/examples/streaming/interactive_chat.py
```

### advanced_streaming.py
Advanced example showing error handling, buffering, and other best practices for production streaming applications.

**Run:**
```bash
uv run --with mellea docs/examples/streaming/advanced_streaming.py
```

## Key Concepts

### Streaming Requires Async
Streaming is only available with async functions (`ainstruct`, `aact`) using `await_result=False`:

```python
# This works - async with await_result=False
thunk = await m.ainstruct("Hello", await_result=False)
last_length = 0
while not thunk.is_computed():
    current_value = await thunk.astream()
    new_content = current_value[last_length:]
    print(new_content, end="")
    last_length = len(current_value)

# This doesn't work - sync functions always await
result = m.instruct("Hello")  # Already computed, cannot stream
```

**Note**: `astream()` returns the accumulated output so far, not individual chunks. You need to track what you've already displayed to show only new content.

### ModelOutputThunk Types
- **`ModelOutputThunk`**: Uncomputed, can be streamed
- **`ComputedModelOutputThunk`**: Already computed, cannot be streamed

### Limitations
- Cannot stream when using `SamplingStrategy` (validation requires complete output)
- Cannot stream from synchronous functions (would cause deadlock)
- Streaming requires an async context

## See Also
- [Tutorial Chapter 13: Streaming Model Outputs](../../tutorial.md#chapter-13-streaming-model-outputs)
- [Tutorial Chapter 12: Asynchronicity](../../tutorial.md#chapter-12-asynchronicity)