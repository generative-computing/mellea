"""Tests for the m serve OpenAI-compatible API server."""

from unittest.mock import Mock

import pytest

from cli.serve.app import make_chat_endpoint
from cli.serve.models import (
    ChatCompletion,
    ChatCompletionRequest,
    ChatMessage,
    CompletionUsage,
)
from mellea.core.base import ModelOutputThunk


@pytest.fixture
def mock_module():
    """Create a mock module with a serve function."""
    module = Mock()
    module.__name__ = "test_module"
    return module


@pytest.fixture
def sample_request():
    """Create a sample ChatCompletionRequest."""
    return ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="Hello, world!")],
        temperature=0.7,
        max_tokens=100,
    )


class TestChatEndpoint:
    """Tests for the chat completion endpoint."""

    @pytest.mark.asyncio
    async def test_basic_completion(self, mock_module, sample_request):
        """Test basic chat completion returns correct structure."""
        # Setup mock output
        mock_output = ModelOutputThunk("Hello! How can I help you?")
        mock_module.serve.return_value = mock_output

        # Create endpoint and call it
        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(sample_request)

        # Verify response structure
        assert isinstance(response, ChatCompletion)
        assert response.model == "test-model"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Hello! How can I help you?"
        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].index == 0

    @pytest.mark.asyncio
    async def test_finish_reason_included(self, mock_module, sample_request):
        """Test that finish_reason is included in the response."""
        mock_output = ModelOutputThunk("Test response")
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(sample_request)

        assert response.choices[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_usage_field_populated(self, mock_module, sample_request):
        """Test that usage field is populated when available."""
        mock_output = ModelOutputThunk("Test response")
        mock_output.usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(sample_request)

        assert response.usage is not None
        assert isinstance(response.usage, CompletionUsage)
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_usage_field_none_when_unavailable(self, mock_module, sample_request):
        """Test that usage field is None when not available."""
        mock_output = ModelOutputThunk("Test response")
        # Don't set usage field
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(sample_request)

        assert response.usage is None

    @pytest.mark.asyncio
    async def test_system_fingerprint_always_none(self, mock_module, sample_request):
        """Test that system_fingerprint is always None.

        Per OpenAI spec, system_fingerprint represents a hash of backend config,
        not the model name. The model name is in response.model.
        We don't currently track backend config fingerprints.
        """
        mock_output = ModelOutputThunk("Test response")
        mock_output.model = "gpt-4-turbo"
        mock_output.provider = "openai"
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(sample_request)

        # system_fingerprint should be None, not the model name
        assert response.system_fingerprint is None
        # Model name should be in the model field
        assert response.model == sample_request.model

    @pytest.mark.asyncio
    async def test_model_options_passed_correctly(self, mock_module, sample_request):
        """Test that model options are passed to serve function correctly."""
        from mellea.backends.model_options import ModelOption

        mock_output = ModelOutputThunk("Test response")
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        await endpoint(sample_request)

        # Verify serve was called with correct arguments
        call_args = mock_module.serve.call_args
        assert call_args is not None
        assert "model_options" in call_args.kwargs
        model_options = call_args.kwargs["model_options"]

        # Should include ModelOption sentinels for temperature and max_tokens
        assert ModelOption.TEMPERATURE in model_options
        assert model_options[ModelOption.TEMPERATURE] == 0.7
        assert ModelOption.MAX_NEW_TOKENS in model_options
        assert model_options[ModelOption.MAX_NEW_TOKENS] == 100
        assert "messages" not in model_options
        assert "requirements" not in model_options

    @pytest.mark.asyncio
    async def test_completion_id_format(self, mock_module, sample_request):
        """Test that completion ID follows OpenAI format."""
        mock_output = ModelOutputThunk("Test response")
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(sample_request)

        # Should start with "chatcmpl-" and have a non-empty suffix
        assert response.id.startswith("chatcmpl-")
        assert len(response.id) > len("chatcmpl-"), "ID should have a suffix"

    @pytest.mark.asyncio
    async def test_created_timestamp_present(self, mock_module, sample_request):
        """Test that created timestamp is present and reasonable."""
        mock_output = ModelOutputThunk("Test response")
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(sample_request)

        # Should be a Unix timestamp (positive integer)
        assert isinstance(response.created, int)
        assert response.created > 0

    @pytest.mark.asyncio
    async def test_object_type_correct(self, mock_module, sample_request):
        """Test that object type is set correctly."""
        mock_output = ModelOutputThunk("Test response")
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(sample_request)

        assert response.object == "chat.completion"

    @pytest.mark.asyncio
    async def test_usage_with_partial_data(self, mock_module, sample_request):
        """Test that usage handles missing fields gracefully."""
        mock_output = ModelOutputThunk("Test response")
        # Only provide some fields
        mock_output.usage = {
            "prompt_tokens": 10
            # Missing completion_tokens and total_tokens
        }
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(sample_request)

        assert response.usage is not None
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 0  # Should default to 0
        assert (
            response.usage.total_tokens == 10
        )  # Should be prompt_tokens + completion_tokens

    @pytest.mark.asyncio
    async def test_all_fields_together(self, mock_module, sample_request):
        """Test that all new fields work together correctly."""
        mock_output = ModelOutputThunk("Complete response")
        mock_output.usage = {
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "total_tokens": 30,
        }
        mock_output.model = "gpt-4"
        mock_output.provider = "openai"
        mock_module.serve.return_value = mock_output

        endpoint = make_chat_endpoint(mock_module)
        response = await endpoint(sample_request)

        # Verify all fields are present
        assert response.choices[0].finish_reason == "stop"
        assert response.usage is not None
        assert response.usage.total_tokens == 30
        assert response.system_fingerprint is None  # Not tracking backend config
        assert response.object == "chat.completion"
        assert response.id.startswith("chatcmpl-")
