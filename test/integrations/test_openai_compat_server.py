"""Tests for the OpenAI-compatible server."""

import json

import pytest
from fastapi.testclient import TestClient

from mellea.integrations.openai_compat.server import ServerConfig, create_app


@pytest.mark.ollama
@pytest.mark.llm
def test_create_app():
    """Test that the app can be created."""
    config = ServerConfig(backend_name="ollama", model_id="granite4:micro")
    app = create_app(config)
    assert app is not None


@pytest.mark.ollama
@pytest.mark.llm
def test_health_endpoint():
    """Test the health check endpoint."""
    config = ServerConfig(backend_name="ollama", model_id="granite4:micro")
    app = create_app(config)
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@pytest.mark.ollama
@pytest.mark.llm
def test_list_models():
    """Test the models listing endpoint."""
    config = ServerConfig(backend_name="ollama", model_id="granite4:micro")
    app = create_app(config)
    client = TestClient(app)

    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) > 0
    assert data["data"][0]["id"] == "granite4:micro"


@pytest.mark.ollama
@pytest.mark.llm
@pytest.mark.qualitative
def test_chat_completions_basic():
    """Test basic chat completion without streaming."""
    config = ServerConfig(backend_name="ollama", model_id="granite4:micro")
    app = create_app(config)
    client = TestClient(app)

    request_data = {
        "model": "granite4:micro",
        "messages": [{"role": "user", "content": "Say hello in one word"}],
        "temperature": 0.7,
        "max_tokens": 10,
    }

    response = client.post("/v1/chat/completions", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "granite4:micro"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] is not None
    assert len(data["choices"][0]["message"]["content"]) > 0


@pytest.mark.ollama
@pytest.mark.llm
@pytest.mark.qualitative
def test_chat_completions_streaming():
    """Test streaming chat completion."""
    config = ServerConfig(backend_name="ollama", model_id="granite4:micro")
    app = create_app(config)
    client = TestClient(app)

    request_data = {
        "model": "granite4:micro",
        "messages": [{"role": "user", "content": "Count to 3"}],
        "stream": True,
        "max_tokens": 20,
    }

    with client.stream("POST", "/v1/chat/completions", json=request_data) as response:
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        chunks = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                data_str = line[6:]  # Remove "data: " prefix
                if data_str == "[DONE]":
                    break
                chunk_data = json.loads(data_str)
                chunks.append(chunk_data)

        # Verify we got multiple chunks
        assert len(chunks) > 0
        # Verify chunk structure
        assert chunks[0]["object"] == "chat.completion.chunk"
        assert chunks[0]["model"] == "granite4:micro"


@pytest.mark.ollama
@pytest.mark.llm
@pytest.mark.qualitative
def test_chat_completions_with_system_message():
    """Test chat completion with system message."""
    config = ServerConfig(backend_name="ollama", model_id="granite4:micro")
    app = create_app(config)
    client = TestClient(app)

    request_data = {
        "model": "granite4:micro",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ],
        "max_tokens": 20,
    }

    response = client.post("/v1/chat/completions", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert data["choices"][0]["message"]["content"] is not None


@pytest.mark.ollama
@pytest.mark.llm
def test_chat_completions_with_seed():
    """Test that seed parameter is accepted."""
    config = ServerConfig(backend_name="ollama", model_id="granite4:micro")
    app = create_app(config)
    client = TestClient(app)

    request_data = {
        "model": "granite4:micro",
        "messages": [{"role": "user", "content": "Hello"}],
        "seed": 42,
        "max_tokens": 10,
    }

    response = client.post("/v1/chat/completions", json=request_data)
    assert response.status_code == 200


def test_server_config_defaults():
    """Test ServerConfig default values."""
    config = ServerConfig()
    assert config.backend_name == "ollama"
    assert config.model_id is None
    assert config.base_url is None
    assert config.api_key is None
    assert config.default_model_options == {}


def test_server_config_custom():
    """Test ServerConfig with custom values."""
    config = ServerConfig(
        backend_name="openai",
        model_id="gpt-4",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        default_model_options={"temperature": 0.5},
        timeout=30,
    )
    assert config.backend_name == "openai"
    assert config.model_id == "gpt-4"
    assert config.base_url == "https://api.openai.com/v1"
    assert config.api_key == "test-key"
    assert config.default_model_options == {"temperature": 0.5}
    assert config.backend_kwargs == {"timeout": 30}


@pytest.mark.ollama
@pytest.mark.llm
@pytest.mark.qualitative
def test_chat_completions_with_tools():
    """Test chat completion with tool calling."""
    config = ServerConfig(backend_name="ollama", model_id="granite4:micro")
    app = create_app(config)
    client = TestClient(app)

    # Define a simple tool
    tool_definition = {
        "type": "function",
        "function": {
            "name": "get_temperature",
            "description": "Returns today's temperature of the given city in Celsius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "A city name"}
                },
                "required": ["location"],
            },
        },
    }

    request_data = {
        "model": "granite4:micro",
        "messages": [
            {
                "role": "user",
                "content": "What is today's temperature in Boston? Use the get_temperature tool.",
            }
        ],
        "tools": [tool_definition],
        "tool_choice": "auto",
        "max_tokens": 100,
    }

    response = client.post("/v1/chat/completions", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1

    message = data["choices"][0]["message"]
    assert message["role"] == "assistant"

    # Check if tool was called
    if message.get("tool_calls"):
        tool_call = message["tool_calls"][0]
        assert tool_call["function"]["name"] == "get_temperature"
        # Parse arguments
        args = json.loads(tool_call["function"]["arguments"])
        assert "location" in args
        assert "boston" in args["location"].lower()


@pytest.mark.ollama
@pytest.mark.llm
def test_chat_completions_empty_message():
    """Test that empty message content is rejected."""
    config = ServerConfig(backend_name="ollama", model_id="granite4:micro")
    app = create_app(config)
    client = TestClient(app)

    request_data = {
        "model": "granite4:micro",
        "messages": [{"role": "user", "content": ""}],
    }

    response = client.post("/v1/chat/completions", json=request_data)
    # Server returns 500 with HTTPException detail about empty content
    assert response.status_code == 500
    assert "Last message must have content" in response.json()["detail"]


@pytest.mark.ollama
@pytest.mark.llm
def test_chat_completions_with_stop_sequences():
    """Test chat completion with stop sequences."""
    config = ServerConfig(backend_name="ollama", model_id="granite4:micro")
    app = create_app(config)
    client = TestClient(app)

    request_data = {
        "model": "granite4:micro",
        "messages": [{"role": "user", "content": "Count: 1, 2, 3"}],
        "stop": [",", "."],
        "max_tokens": 20,
    }

    response = client.post("/v1/chat/completions", json=request_data)
    assert response.status_code == 200


@pytest.mark.ollama
@pytest.mark.llm
def test_chat_completions_with_penalties():
    """Test chat completion with presence and frequency penalties."""
    config = ServerConfig(backend_name="ollama", model_id="granite4:micro")
    app = create_app(config)
    client = TestClient(app)

    request_data = {
        "model": "granite4:micro",
        "messages": [{"role": "user", "content": "Hello"}],
        "presence_penalty": 0.5,
        "frequency_penalty": 0.3,
        "max_tokens": 10,
    }

    response = client.post("/v1/chat/completions", json=request_data)
    assert response.status_code == 200


@pytest.mark.ollama
@pytest.mark.llm
@pytest.mark.qualitative
def test_chat_completions_multi_turn():
    """Test multi-turn conversation."""
    config = ServerConfig(backend_name="ollama", model_id="granite4:micro")
    app = create_app(config)
    client = TestClient(app)

    # First turn
    request_data = {
        "model": "granite4:micro",
        "messages": [{"role": "user", "content": "My name is Alice."}],
        "max_tokens": 20,
    }

    response = client.post("/v1/chat/completions", json=request_data)
    assert response.status_code == 200
    first_response = response.json()["choices"][0]["message"]["content"]

    # Second turn with history
    request_data = {
        "model": "granite4:micro",
        "messages": [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": first_response},
            {"role": "user", "content": "What is my name?"},
        ],
        "max_tokens": 20,
    }

    response = client.post("/v1/chat/completions", json=request_data)
    assert response.status_code == 200


@pytest.mark.ollama
@pytest.mark.llm
def test_chat_completions_with_top_p():
    """Test chat completion with top_p parameter."""
    config = ServerConfig(backend_name="ollama", model_id="granite4:micro")
    app = create_app(config)
    client = TestClient(app)

    request_data = {
        "model": "granite4:micro",
        "messages": [{"role": "user", "content": "Hello"}],
        "top_p": 0.9,
        "max_tokens": 10,
    }

    response = client.post("/v1/chat/completions", json=request_data)
    assert response.status_code == 200


def test_create_app_with_no_config():
    """Test that app can be created without explicit config."""
    app = create_app()
    assert app is not None
    assert app.state.config.backend_name == "ollama"


def test_server_config_with_backend_kwargs():
    """Test ServerConfig with additional backend kwargs."""
    config = ServerConfig(
        backend_name="ollama", model_id="granite4:micro", timeout=60, max_retries=3
    )
    assert config.backend_kwargs["timeout"] == 60
    assert config.backend_kwargs["max_retries"] == 3


@pytest.mark.ollama
@pytest.mark.llm
def test_session_reuse():
    """Test that sessions are reused for the same model."""
    config = ServerConfig(backend_name="ollama", model_id="granite4:micro")
    app = create_app(config)
    client = TestClient(app)

    request_data = {
        "model": "granite4:micro",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10,
    }

    # Make two requests
    response1 = client.post("/v1/chat/completions", json=request_data)
    assert response1.status_code == 200

    response2 = client.post("/v1/chat/completions", json=request_data)
    assert response2.status_code == 200

    # Verify session was reused (same session key)
    session_key = f"{config.backend_name}:{request_data['model']}"
    assert session_key in app.state.sessions


@pytest.mark.ollama
@pytest.mark.llm
def test_usage_information():
    """Test that usage information is returned."""
    config = ServerConfig(backend_name="ollama", model_id="granite4:micro")
    app = create_app(config)
    client = TestClient(app)

    request_data = {
        "model": "granite4:micro",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10,
    }

    response = client.post("/v1/chat/completions", json=request_data)
    assert response.status_code == 200

    data = response.json()
    # Usage may or may not be present depending on backend
    if data.get("usage"):
        assert "prompt_tokens" in data["usage"]
        assert "completion_tokens" in data["usage"]
        assert "total_tokens" in data["usage"]


@pytest.mark.ollama
@pytest.mark.llm
def test_model_list_with_custom_model():
    """Test models endpoint returns configured model."""
    config = ServerConfig(backend_name="ollama", model_id="custom-model")
    app = create_app(config)
    client = TestClient(app)

    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["data"][0]["id"] == "custom-model"
    assert data["data"][0]["owned_by"] == "ollama"


def test_model_list_without_model_id():
    """Test models endpoint with no model_id configured."""
    config = ServerConfig(backend_name="ollama")
    app = create_app(config)
    client = TestClient(app)

    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["data"][0]["id"] == "default"


@pytest.mark.ollama
@pytest.mark.llm
def test_backend_config_with_base_url():
    """Test that base_url config is accepted and session is created."""
    config = ServerConfig(
        backend_name="ollama",
        model_id="granite4:micro",
        base_url="http://localhost:11434",
    )
    app = create_app(config)
    client = TestClient(app)

    request_data = {
        "model": "granite4:micro",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10,
    }

    response = client.post("/v1/chat/completions", json=request_data)
    assert response.status_code == 200

    # Verify session was created successfully with the base_url config
    session_key = f"{config.backend_name}:{request_data['model']}"
    assert session_key in app.state.sessions
    assert app.state.sessions[session_key] is not None


@pytest.mark.ollama
@pytest.mark.llm
def test_session_with_different_models():
    """Test that different models create separate sessions."""
    config = ServerConfig(backend_name="ollama", model_id="granite4:micro")
    app = create_app(config)
    client = TestClient(app)

    # Request with first model
    request_data_1 = {
        "model": "granite4:micro",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10,
    }
    response1 = client.post("/v1/chat/completions", json=request_data_1)
    assert response1.status_code == 200

    # Request with different model (simulated)
    request_data_2 = {
        "model": "llama3.2:1b",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10,
    }
    response2 = client.post("/v1/chat/completions", json=request_data_2)
    assert response2.status_code == 200

    # Verify separate sessions were created
    session_key_1 = f"{config.backend_name}:granite4:micro"
    session_key_2 = f"{config.backend_name}:llama3.2:1b"
    assert session_key_1 in app.state.sessions
    assert session_key_2 in app.state.sessions
    assert app.state.sessions[session_key_1] != app.state.sessions[session_key_2]


@pytest.mark.ollama
@pytest.mark.llm
@pytest.mark.qualitative
def test_chat_completions_streaming_with_tools():
    """Test streaming chat completion with tool calling."""
    config = ServerConfig(backend_name="ollama", model_id="granite4:micro")
    app = create_app(config)
    client = TestClient(app)

    tool_definition = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"],
            },
        },
    }

    request_data = {
        "model": "granite4:micro",
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in Paris? Use the get_weather tool.",
            }
        ],
        "tools": [tool_definition],
        "stream": True,
        "max_tokens": 100,
    }

    with client.stream("POST", "/v1/chat/completions", json=request_data) as response:
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        chunks = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                chunk_data = json.loads(data_str)
                chunks.append(chunk_data)

        # Verify we got chunks
        assert len(chunks) > 0
        assert chunks[0]["object"] == "chat.completion.chunk"


@pytest.mark.ollama
@pytest.mark.llm
@pytest.mark.qualitative
def test_chat_completions_with_multiple_tools():
    """Test chat completion with multiple tool definitions."""
    config = ServerConfig(backend_name="ollama", model_id="granite4:micro")
    app = create_app(config)
    client = TestClient(app)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Get current time for a timezone",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {"type": "string", "description": "Timezone name"}
                    },
                    "required": ["timezone"],
                },
            },
        },
    ]

    request_data = {
        "model": "granite4:micro",
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in Tokyo? Use the appropriate tool.",
            }
        ],
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 100,
    }

    response = client.post("/v1/chat/completions", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert data["object"] == "chat.completion"

    # If tools were called, verify the structure
    message = data["choices"][0]["message"]
    if message.get("tool_calls"):
        for tool_call in message["tool_calls"]:
            assert "id" in tool_call
            assert "type" in tool_call
            assert tool_call["type"] == "function"
            assert "function" in tool_call
            assert "name" in tool_call["function"]
            assert tool_call["function"]["name"] in ["get_weather", "get_time"]
