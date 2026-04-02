"""Tests for the OpenAI-compatible serve endpoint."""

from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cli.serve.app import make_chat_endpoint
from cli.serve.models import ChatCompletionRequest, ChatMessage


@pytest.fixture
def mock_module_success():
    """Create a mock module that returns a successful response."""
    module = Mock()
    module.__name__ = "test_module"
    output = Mock()
    output.value = "Test response"
    output.usage = None  # No usage info in this test
    module.serve = Mock(return_value=output)
    return module


@pytest.fixture
def mock_module_attribute_error():
    """Create a mock module that raises AttributeError."""
    module = Mock()
    module.__name__ = "test_module"
    output = Mock(spec=[])  # No 'value' attribute
    module.serve = Mock(return_value=output)
    return module


@pytest.fixture
def mock_module_value_error():
    """Create a mock module that raises ValueError."""
    module = Mock()
    module.__name__ = "test_module"
    module.serve = Mock(side_effect=ValueError("Invalid input"))
    return module


@pytest.fixture
def mock_module_generic_error():
    """Create a mock module that raises a generic exception."""
    module = Mock()
    module.__name__ = "test_module"
    module.serve = Mock(side_effect=RuntimeError("Unexpected error"))
    return module


@pytest.fixture
def sample_request():
    """Create a sample chat completion request."""
    return ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="Hello")],
        requirements=None,
    )


@pytest.mark.unit
def test_successful_completion(mock_module_success, sample_request):
    """Test successful chat completion."""
    app = FastAPI()
    endpoint = make_chat_endpoint(mock_module_success)
    app.add_api_route("/test/completions", endpoint, methods=["POST"])
    client = TestClient(app)

    response = client.post("/test/completions", json=sample_request.model_dump())

    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "Test response"
    assert data["model"] == "test-model"
    assert "id" in data
    assert data["object"] == "chat.completion"


@pytest.mark.unit
def test_attribute_error_handling(mock_module_attribute_error, sample_request):
    """Test handling of AttributeError (e.g., missing 'value' attribute)."""
    app = FastAPI()
    endpoint = make_chat_endpoint(mock_module_attribute_error)
    app.add_api_route("/test/attribute-error", endpoint, methods=["POST"])
    client = TestClient(app)

    response = client.post("/test/attribute-error", json=sample_request.model_dump())

    assert response.status_code == 500
    data = response.json()
    assert "error" in data
    assert data["error"]["type"] == "server_error"
    assert "Internal server error" in data["error"]["message"]


@pytest.mark.unit
def test_value_error_handling(mock_module_value_error, sample_request):
    """Test handling of ValueError (validation errors)."""
    app = FastAPI()
    endpoint = make_chat_endpoint(mock_module_value_error)
    app.add_api_route("/test/value-error", endpoint, methods=["POST"])
    client = TestClient(app)

    response = client.post("/test/value-error", json=sample_request.model_dump())

    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["type"] == "invalid_request_error"
    assert "Invalid request" in data["error"]["message"]
    assert "Invalid input" in data["error"]["message"]


@pytest.mark.unit
def test_generic_error_handling(mock_module_generic_error, sample_request):
    """Test handling of generic exceptions."""
    app = FastAPI()
    endpoint = make_chat_endpoint(mock_module_generic_error)
    app.add_api_route("/test/generic-error", endpoint, methods=["POST"])
    client = TestClient(app)

    response = client.post("/test/generic-error", json=sample_request.model_dump())

    assert response.status_code == 500
    data = response.json()
    assert "error" in data
    assert data["error"]["type"] == "server_error"
    assert "Internal server error" in data["error"]["message"]
    assert "Unexpected error" in data["error"]["message"]


@pytest.mark.unit
def test_endpoint_name_generation(mock_module_success):
    """Test that endpoint names are generated correctly."""
    endpoint = make_chat_endpoint(mock_module_success)
    assert endpoint.__name__ == "chat_test_module_endpoint"
