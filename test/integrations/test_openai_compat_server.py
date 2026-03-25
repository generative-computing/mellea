"""Tests for the OpenAI-compatible server."""

import json

import pytest
from fastapi.testclient import TestClient

from mellea.integrations.openai_compat.server import ServerConfig, create_app


@pytest.mark.ollama
@pytest.mark.llm
def test_create_app():
    """Test that the app can be created."""
    config = ServerConfig(backend_name="ollama", model_id="llama3.2:1b")
    app = create_app(config)
    assert app is not None


@pytest.mark.ollama
@pytest.mark.llm
def test_health_endpoint():
    """Test the health check endpoint."""
    config = ServerConfig(backend_name="ollama", model_id="llama3.2:1b")
    app = create_app(config)
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@pytest.mark.ollama
@pytest.mark.llm
def test_list_models():
    """Test the models listing endpoint."""
    config = ServerConfig(backend_name="ollama", model_id="llama3.2:1b")
    app = create_app(config)
    client = TestClient(app)

    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) > 0
    assert data["data"][0]["id"] == "llama3.2:1b"


@pytest.mark.ollama
@pytest.mark.llm
@pytest.mark.qualitative
def test_chat_completions_basic():
    """Test basic chat completion without streaming."""
    config = ServerConfig(backend_name="ollama", model_id="llama3.2:1b")
    app = create_app(config)
    client = TestClient(app)

    request_data = {
        "model": "llama3.2:1b",
        "messages": [{"role": "user", "content": "Say hello in one word"}],
        "temperature": 0.7,
        "max_tokens": 10,
    }

    response = client.post("/v1/chat/completions", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "llama3.2:1b"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] is not None
    assert len(data["choices"][0]["message"]["content"]) > 0


@pytest.mark.ollama
@pytest.mark.llm
@pytest.mark.qualitative
def test_chat_completions_streaming():
    """Test streaming chat completion."""
    config = ServerConfig(backend_name="ollama", model_id="llama3.2:1b")
    app = create_app(config)
    client = TestClient(app)

    request_data = {
        "model": "llama3.2:1b",
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
        assert chunks[0]["model"] == "llama3.2:1b"


@pytest.mark.ollama
@pytest.mark.llm
@pytest.mark.qualitative
def test_chat_completions_with_system_message():
    """Test chat completion with system message."""
    config = ServerConfig(backend_name="ollama", model_id="llama3.2:1b")
    app = create_app(config)
    client = TestClient(app)

    request_data = {
        "model": "llama3.2:1b",
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
    config = ServerConfig(backend_name="ollama", model_id="llama3.2:1b")
    app = create_app(config)
    client = TestClient(app)

    request_data = {
        "model": "llama3.2:1b",
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
