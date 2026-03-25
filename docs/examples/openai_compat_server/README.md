# OpenAI-Compatible Server Examples

This directory contains examples for using mellea's OpenAI-compatible HTTP server.

## Overview

The OpenAI-compatible server allows you to wrap any mellea backend and expose it through OpenAI's API format. This means you can use the official OpenAI Python SDK or any other OpenAI-compatible client to interact with local models, Ollama, or other backends supported by mellea.

## Files

- **`basic_server.py`**: Shows how to start the server with basic configuration
- **`client_example.py`**: Demonstrates using the OpenAI Python SDK to interact with the server
- **`curl_examples.sh`**: Command-line examples using curl

## Quick Start

### 1. Start the Server

```python
from mellea.integrations.openai_compat.server import ServerConfig, run_server

config = ServerConfig(
    backend_name="ollama",
    model_id="granite4:micro",
)

run_server(host="0.0.0.0", port=8000, config=config)
```

Or run the example:

```bash
uv run python docs/examples/openai_compat_server/basic_server.py
```

### 2. Use the OpenAI SDK

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="granite4:micro",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
)

print(response.choices[0].message.content)
```

### 3. Or Use curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "granite4:micro",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Supported Endpoints

- **`POST /v1/chat/completions`**: Chat completions (with streaming support)
- **`GET /v1/models`**: List available models
- **`GET /health`**: Health check

## Interactive API Documentation

FastAPI automatically provides interactive API documentation when the server is running:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

Visit the Swagger UI to explore all endpoints, view request/response schemas, and test the API directly from your browser.

## Configuration Options

The `ServerConfig` class accepts the following parameters:

- **`backend_name`**: Name of the mellea backend (e.g., "ollama", "openai", "hf")
- **`model_id`**: Model identifier to use
- **`base_url`**: Base URL for the backend API (if applicable)
- **`api_key`**: API key for the backend (if applicable)
- **`default_model_options`**: Default model options for all requests
- **`**backend_kwargs`**: Additional arguments passed to the backend

## Supported Features

✅ Chat completions
✅ Streaming responses
✅ System messages
✅ Temperature, top_p, max_tokens
✅ Seed for reproducibility
✅ Tool calling (when supported by backend)
✅ Token usage tracking

## Backend Support

The server works with any mellea backend:

- **Ollama**: Local models via Ollama
- **OpenAI**: OpenAI API models
- **HuggingFace**: Local HuggingFace models
- **Watsonx**: IBM Watsonx models
- **LiteLLM**: Any LiteLLM-supported provider

## Deployment

The server can be deployed as a standalone service:

```bash
# Using uvicorn directly
uvicorn mellea.integrations.openai_compat.server:app --host 0.0.0.0 --port 8000

# Or using the run_server function
python -c "from mellea.integrations.openai_compat.server import run_server; run_server()"
```

## Testing

Run the tests:

```bash
uv run pytest test/integrations/test_openai_compat_server.py
```

Run the client example (requires server to be running):

```bash
uv run python docs/examples/openai_compat_server/client_example.py