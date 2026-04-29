# M Serve Examples

This directory contains examples for deploying Mellea programs as API services using the `m serve` CLI command.

## Files

### m_serve_example_simple.py
A simple example showing how to structure a Mellea program for serving as an API.

**Key Features:**
- Defining a `serve()` function that takes input and returns output
- Using requirements and sampling strategies in served programs
- Custom validation functions for API constraints
- Handling chat message inputs

### m_serve_example_streaming.py
A dedicated streaming example for `m serve` that supports both modes:
- `stream=False` returns a normal computed response
- `stream=True` returns an uncomputed thunk so the server can emit
  incremental Server-Sent Events (SSE) chunks

### m_serve_example_response_format.py
Example demonstrating structured output with the `response_format` parameter.

**Key Features:**
- Supporting the `format` parameter in serve functions
- Structured output validation with JSON schemas
- Three format types: `text`, `json_object`, `json_schema`

### pii_serve.py
Example of serving a PII (Personally Identifiable Information) detection service.

### client.py
Client code for testing the served API endpoints with non-streaming requests.

### client_streaming.py
Client code demonstrating streaming responses using Server-Sent Events (SSE)
against `m_serve_example_streaming.py`.

### client_response_format.py
Client code demonstrating all three `response_format` types with examples.

## Concepts Demonstrated

- **API Deployment**: Exposing Mellea programs as REST APIs
- **Input Handling**: Processing structured inputs (chat messages, requirements)
- **Output Formatting**: Returning appropriate response types
- **Validation in Production**: Using requirements in deployed services
- **Model Options**: Passing model configuration through API
- **Streaming Responses**: Real-time token streaming via Server-Sent Events (SSE)
- **Structured Output**: Using `response_format` for JSON schema validation

## Basic Pattern

```python
from mellea import start_session
from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.core import Requirement

session = start_session()

def serve(input: list[ChatMessage], 
          requirements: list[str] | None = None,
          model_options: dict | None = None):
    """Main serving function - called by m serve."""
    message = input[-1].content
    
    result = session.instruct(
        description=message,
        requirements=requirements or [],
        strategy=RejectionSamplingStrategy(loop_budget=3),
        model_options=model_options
    )
    return result
```

## Running the Server

### Sampling

```bash
# Start the sampling example server
m serve docs/examples/m_serve/m_serve_example_simple.py

# In another terminal, test with the non-streaming client
python docs/examples/m_serve/client.py
```

### Streaming

```bash
# Start the dedicated streaming example server
m serve docs/examples/m_serve/m_serve_example_streaming.py

# In another terminal, test with the streaming client
python docs/examples/m_serve/client_streaming.py
```

### Response Format

```bash
# Start the response_format example server
m serve docs/examples/m_serve/m_serve_example_response_format.py

# In another terminal, test with the response_format client
python docs/examples/m_serve/client_response_format.py
```

## Response Format Support

The server supports structured output via the `response_format` parameter, which allows you to control the format of the model's response. This is compatible with OpenAI's response format API.

**Three Format Types:**

1. **`text`** (default): Plain text output
2. **`json_object`**: Unstructured JSON output (model decides the schema)
3. **`json_schema`**: Structured output validated against a JSON schema

**Key Features:**
- Automatic JSON schema to Pydantic model conversion
- Schema validation for structured outputs
- OpenAI-compatible API
- Works with the `format` parameter in serve functions

**Example - JSON Schema:**
```python
import openai

client = openai.OpenAI(api_key="na", base_url="http://0.0.0.0:8080/v1")

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

For a real streaming demo, serve `m_serve_example_streaming.py`. That example
supports both normal and streaming responses consistently. The sampling example
(`m_serve_example_simple.py`) demonstrates rejection sampling and validation,
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

client = openai.OpenAI(api_key="na", base_url="http://0.0.0.0:8080/v1")

# Enable streaming with stream=True
stream = client.chat.completions.create(
    messages=[{"role": "user", "content": "Tell me a story"}],
    model="granite4:micro-h",
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## API Endpoints

The `m serve` command automatically creates:
- `POST /generate`: Main generation endpoint
- `GET /health`: Health check endpoint
- `GET /docs`: API documentation (Swagger UI)

## Use Cases

- **Production Deployment**: Deploy Mellea programs as microservices
- **API Integration**: Integrate with existing systems via REST API
- **Scalability**: Run multiple instances behind a load balancer
- **Monitoring**: Add logging and metrics to served programs

## Related Documentation

- See `cli/serve/` for server implementation
- See `mellea/stdlib/session.py` for session management