# M Serve Examples

This directory contains examples for deploying Mellea programs as API services
using the `m serve` CLI command. For a full reference, see
[m serve](../../docs/integrations/m-serve.md).

## Structure

Each subdirectory contains a server implementation and its matching client(s).

Subdirectories marked with a `client_responses*.py` file include a Responses API
client alongside the standard Chat Completions client. Both clients use the same
server program — no server changes are needed to use `/v1/responses`.

## Examples

| Subdirectory | Description |
| :--- | :--- |
| [`simple/`](./simple/) | Basic request/response with rejection sampling |
| [`streaming/`](./streaming/) | Real-time token streaming via Server-Sent Events (SSE) |
| [`response-format/`](./response-format/) | Structured output with `response_format` / JSON schema |
| [`tool-calling/`](./tool-calling/) | Function/tool calling through the API |
| [`multimodal-image/`](./multimodal-image/) | Vision model serving with image inputs (Chat Completions + Responses API) |
| [`multimodal-audio/`](./multimodal-audio/) | Audio-text-to-text serving (llama-server and Ollama/Granite variants) |
| [`pii/`](./pii/) | PII detection service |
| [`model-routing/`](./model-routing/) | Using or ignoring the client-supplied model ID |
| [`sessions/`](./sessions/) | Multi-turn sessions via `previous_response_id` (Responses API) |

## Running the Server

### Sampling

```bash
# Start the sampling example server
m serve docs/examples/m_serve/simple/m_serve_example_simple.py

# In another terminal, test with the non-streaming client
python docs/examples/m_serve/simple/client.py
```

### Streaming

```bash
# Start the dedicated streaming example server
m serve docs/examples/m_serve/streaming/m_serve_example_streaming.py

# In another terminal, test with the streaming client
python docs/examples/m_serve/streaming/client_streaming.py
```

### Response Format

```bash
# Start the response_format example server
m serve docs/examples/m_serve/response-format/m_serve_example_response_format.py

# In another terminal, test with the response_format client
python docs/examples/m_serve/response-format/client_response_format.py
```

### Multimodal Images

```bash
# Start the multimodal image example server
m serve docs/examples/m_serve/multimodal-image/m_serve_example_multimodal_image.py

# In another terminal, test with the multimodal client
uv run python docs/examples/m_serve/multimodal-image/client_multimodal_image.py
```

**Responses API Support:**

The `/v1/responses` endpoint also supports multimodal inputs with image content blocks.
Both endpoints (`/v1/chat/completions` and `/v1/responses`) handle multimodal content
identically — the same server code works for both APIs.

```bash
# Test via Responses API
uv run python docs/examples/m_serve/multimodal-image/client_responses_multimodal.py
```

**Content Block Format:**

The Responses API uses a slightly different content block format than Chat Completions:

| API | Text Block | Image Block |
|-----|-----------|-------------|
| Chat Completions | `{"type": "text", "text": "..."}` | `{"type": "image_url", "image_url": {"url": "..."}}` |
| Responses | `{"type": "input_text", "text": "..."}` | `{"type": "input_image", "image_url": "..."}` |

The server automatically converts Responses API format to the internal ChatMessage format,
so your Mellea program receives the same multimodal `ChatMessage` objects regardless of
which API endpoint the client uses.

### Multimodal Audio (llama-server + Gemma)

Requires a local llama-server with an audio-capable Gemma checkpoint and
`--mmproj` loaded (see the file header for the full `llama-server` command).
Configure via `LLAMA_SERVER_URL`, `LLAMA_SERVER_API_KEY`, and
`LLAMA_SERVER_MODEL` environment variables.

```bash
# Start the llama-server audio example
m serve docs/examples/m_serve/multimodal-audio/m_serve_example_multimodal_audio_llama_server.py

# In another terminal, test with the audio client
uv run python docs/examples/m_serve/multimodal-audio/client_multimodal_audio.py
```

### Multimodal Audio (Ollama + Granite — two-step transcribe + chat)

Requires both models pulled locally:

```bash
ollama pull hf.co/ibm-granite/granite-speech-4.1-2b-GGUF:Q4_K_M
ollama pull granite4.1:3b
```

```bash
# Start the Granite two-step audio example
m serve docs/examples/m_serve/multimodal-audio/m_serve_example_multimodal_audio_granite.py

# In another terminal, test with the audio client
uv run python docs/examples/m_serve/multimodal-audio/client_multimodal_audio.py
```

### Tool Calling

**Client-side tool execution** (client handles `function_call` items):

```bash
# Start the server
uv run m serve docs/examples/m_serve/tool-calling/m_serve_example_tool_calling.py

# Test via Chat Completions API
uv run python docs/examples/m_serve/tool-calling/client_tool_calling.py

# Test via Responses API (handles function_call items + follow-up)
uv run python docs/examples/m_serve/tool-calling/client_responses_tool_calling.py

# Test streaming + tool calling
uv run python docs/examples/m_serve/tool-calling/client_streaming_tool_calling.py
```

**Server-side tool execution** (server runs full agentic loop, returns complete answer):

```bash
# Start the server-side tool calling server
uv run m serve docs/examples/m_serve/tool-calling/m_serve_example_tool_calling_server_side.py

# Test via Responses API — single request, complete answer
uv run python docs/examples/m_serve/tool-calling/client_responses_tool_calling_server_side.py
```

### Model Routing

```bash
# Start the model-routing example server
uv run m serve docs/examples/m_serve/model-routing/m_serve_example_model_routing.py

# In another terminal, run the client
uv run python docs/examples/m_serve/model-routing/client_model_routing.py
```

### Multi-Turn Sessions

```bash
# Start the sessions example server
m serve docs/examples/m_serve/sessions/m_serve_example_sessions.py

# In another terminal, run the session client
python docs/examples/m_serve/sessions/client_responses_sessions.py
```

## Response Format Support

The server supports structured output via the `text` parameter (Responses API, `text={"format": {...}}` — the shape the OpenAI SDK sends) or the `response_format` parameter (Chat Completions API), which allows you to control the format of the model's response. This is compatible with OpenAI's response format API. A top-level `format` parameter on `/v1/responses` is also accepted as an alias.

**Three Format Types:**

1. **`text`** (default): Plain text output
2. **`json_object`**: Unstructured JSON output (model decides the schema)
3. **`json_schema`**: Structured output validated against a JSON schema

**Key Features:**
- Automatic JSON schema to Pydantic model conversion
- Schema validation for structured outputs
- OpenAI-compatible API
- Works with the `format` parameter in serve functions
- Supported on both `/v1/chat/completions` and `/v1/responses` endpoints

**Example - JSON Schema (Chat Completions API):**
```python
import openai

client = openai.OpenAI(api_key="na", base_url="http://localhost:8080/v1")

# Define a schema for structured output
person_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string"},
    },
    "required": ["name", "age", "email"],
}

response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Generate a person named Alice"}],
    model="granite4:micro-h",
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "Person",
            "schema": person_schema,
            "strict": True,
        },
    },
)

# Response will be valid JSON matching the schema
print(response.choices[0].message.content)
```

**Example - JSON Schema (Responses API):**
```python
import openai

client = openai.OpenAI(api_key="na", base_url="http://localhost:8080/v1")

response = client.responses.create(
    input="Generate a person named Alice",
    model="granite4:micro-h",
    text={
        "format": {
            "type": "json_schema",
            "name": "Person",
            "schema": person_schema,
            "strict": True,
        }
    },
)

# Response will be valid JSON matching the schema
print(response.output_text)
```

**Server Implementation:**
Your serve function must accept a `format` parameter to support `json_schema`:

```python
def serve(
    input: list[ChatMessage],
    requirements: list[str] | None = None,
    model_options: dict | None = None,
    format: type | None = None,  # Add this parameter
) -> ModelOutputThunk:
    result = session.instruct(
        description=input[-1].content,
        requirements=requirements,
        model_options=model_options,
        format=format,  # Pass to instruct()
    )
    return result
```

## Streaming Support

The server supports streaming responses via Server-Sent Events (SSE) when the
`stream=True` parameter is set in the request. This allows clients to receive
tokens as they are generated, providing a better user experience for long-running
generations.

For a real streaming demo, serve `streaming/m_serve_example_streaming.py`. That example
supports both normal and streaming responses consistently. The sampling example
(`simple/m_serve_example_simple.py`) demonstrates rejection sampling and validation,
not token-by-token streaming.

**Key Features:**
- Real-time token streaming using SSE
- OpenAI-compatible streaming format (`ChatCompletionChunk`)
- Final chunk includes usage statistics when the backend provides usage data
- The dedicated streaming example supports both `stream=False` and `stream=True`
- Works with any backend that supports `ModelOutputThunk.astream()`

**Example:**
```python
import openai

client = openai.OpenAI(api_key="na", base_url="http://localhost:8080/v1")

# Enable streaming with stream=True
stream = client.chat.completions.create(
    messages=[{"role": "user", "content": "Tell me a story"}],
    model="granite4.1:3b",
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Multi-Turn Sessions (Responses API)

The Responses API stores completed responses **in memory** and lets clients
continue a conversation without resending history.  Each response carries an
`id` (e.g. `resp_abc123`); pass it as `previous_response_id` in the next
request and the server reconstructs the full history automatically.

> **Note:** the store is in-process only.  Sessions are lost when the server
> restarts.  For durable persistence across restarts, store the response `id`
> externally and resend the full history when reconnecting.

```python
import openai

client = openai.OpenAI(api_key="na", base_url="http://localhost:8080/v1")

# Turn 1
r1 = client.responses.create(model="granite4.1:3b", input="My name is Alice.")
print(r1.output_text)    # "Nice to meet you, Alice!"
print(r1.expires_at)     # Unix timestamp when the session will be evicted

# Turn 2 — only send the new message, not the history
r2 = client.responses.create(
    model="granite4.1:3b",
    input="What is my name?",
    previous_response_id=r1.id,
)
print(r2.output_text)   # "Your name is Alice."
```

Pass `store=False` if you don't need history chaining and want to skip storing
the response altogether (e.g. for sensitive inputs):

```python
r = client.responses.create(model="granite4.1:3b", input="...", store=False)
# r.expires_at is None — response is never persisted
```

Stored responses can also be retrieved by ID:

```bash
curl http://localhost:8080/v1/responses/resp_abc123
```

Sessions expire after 30 minutes by default.  Pass `--response-ttl <seconds>`
to `m serve` to change the TTL:

```bash
m serve my_app.py --response-ttl 3600   # 1-hour sessions
```

At most 1000 responses are kept in memory; when the cap is reached the
oldest stored responses are dropped first.  Pass
`--max-stored-responses <n>` to change it:

```bash
m serve my_app.py --max-stored-responses 5000
```

See `sessions/` for a full runnable example.

## API Endpoints

The `m serve` command registers:

- `POST /v1/chat/completions` — OpenAI Chat Completions API
- `POST /v1/responses` — OpenAI Responses API (supports `previous_response_id` sessions)
- `GET /v1/responses/{id}` — Retrieve a stored response by ID
- `GET /health` — health check

Both POST endpoints call the same `serve()` function in your program.
FastAPI's interactive docs are available at `GET /docs` while the server is running.
