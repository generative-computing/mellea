"""Tests for sync/async serve function handling in m serve."""

import asyncio
from unittest.mock import Mock

import pytest

from cli.serve.app import make_chat_endpoint
from cli.serve.models import ChatCompletionRequest, ChatMessage
from mellea.backends.model_options import ModelOption
from mellea.core import ModelOutputThunk


@pytest.fixture
def mock_sync_module():
    """Create a mock module with a synchronous serve function."""
    module = Mock()
    module.__name__ = "test_sync_module"

    def sync_serve(input, requirements=None, model_options=None):
        """Synchronous serve function."""
        # Simulate some work
        result = Mock(spec=ModelOutputThunk)
        result.value = f"Sync response to: {input[-1].content}"
        return result

    # Use Mock to wrap the function so we can track calls
    module.serve = Mock(side_effect=sync_serve)
    return module


@pytest.fixture
def mock_async_module():
    """Create a mock module with an asynchronous serve function."""
    module = Mock()
    module.__name__ = "test_async_module"

    async def async_serve(input, requirements=None, model_options=None):
        """Asynchronous serve function."""
        # Simulate async work
        await asyncio.sleep(0.01)
        result = Mock(spec=ModelOutputThunk)
        result.value = f"Async response to: {input[-1].content}"
        return result

    module.serve = async_serve
    return module


@pytest.fixture
def mock_slow_sync_module():
    """Create a mock module with a slow synchronous serve function."""
    module = Mock()
    module.__name__ = "test_slow_sync_module"

    def slow_sync_serve(input, requirements=None, model_options=None):
        """Slow synchronous serve function that would block event loop."""
        import time

        time.sleep(0.1)  # Simulate blocking work
        result = Mock(spec=ModelOutputThunk)
        result.value = f"Slow sync response to: {input[-1].content}"
        return result

    module.serve = slow_sync_serve
    return module


class TestSyncAsyncServeHandling:
    """Test that serve handles both sync and async serve functions correctly."""

    @pytest.mark.asyncio
    async def test_sync_serve_function(self, mock_sync_module):
        """Test that synchronous serve functions work correctly."""
        endpoint = make_chat_endpoint(mock_sync_module)

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello sync!")],
        )

        response = await endpoint(request)

        assert response.choices[0].message.content == "Sync response to: Hello sync!"
        assert response.model == "test-model"
        assert response.object == "chat.completion"

    @pytest.mark.asyncio
    async def test_async_serve_function(self, mock_async_module):
        """Test that asynchronous serve functions work correctly."""
        endpoint = make_chat_endpoint(mock_async_module)

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello async!")],
        )

        response = await endpoint(request)

        assert response.choices[0].message.content == "Async response to: Hello async!"
        assert response.model == "test-model"
        assert response.object == "chat.completion"

    @pytest.mark.asyncio
    async def test_slow_sync_does_not_block(self, mock_slow_sync_module):
        """Test that slow sync functions run in thread pool and don't block event loop.

        This test verifies non-blocking behavior by measuring timing. If the sync
        function blocked the event loop, two sequential calls would take 2x the time.
        With proper threading, they should overlap and take only slightly more than 1x.
        """
        import time

        endpoint = make_chat_endpoint(mock_slow_sync_module)

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello slow!")],
        )

        # Time two concurrent requests
        start = time.time()
        results = await asyncio.gather(endpoint(request), endpoint(request))
        elapsed = time.time() - start

        # If blocking: would take ~0.2s (0.1s + 0.1s sequentially)
        # If non-blocking: should take ~0.1s (both run concurrently in threads)
        # Allow some overhead, but should be much less than 0.2s
        assert elapsed < 0.15, (
            f"Took {elapsed:.3f}s - appears to be blocking (expected ~0.1s)"
        )
        assert all(
            r.choices[0].message.content == "Slow sync response to: Hello slow!"
            for r in results
        )

    @pytest.mark.asyncio
    async def test_concurrent_requests_with_sync_serve(self, mock_slow_sync_module):
        """Test that multiple sync requests can be handled concurrently."""
        endpoint = make_chat_endpoint(mock_slow_sync_module)

        requests = [
            ChatCompletionRequest(
                model="test-model",
                messages=[ChatMessage(role="user", content=f"Request {i}")],
            )
            for i in range(3)
        ]

        # Run requests concurrently
        responses = await asyncio.gather(*[endpoint(req) for req in requests])

        # All should complete successfully
        assert len(responses) == 3
        for i, response in enumerate(responses):
            assert (
                response.choices[0].message.content
                == f"Slow sync response to: Request {i}"
            )

    @pytest.mark.asyncio
    async def test_requirements_passed_to_serve(self, mock_sync_module):
        """Test that requirements are correctly passed to serve function."""
        endpoint = make_chat_endpoint(mock_sync_module)

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Test")],
            requirements=["req1", "req2"],
        )

        await endpoint(request)

        # Verify serve was called with requirements
        mock_sync_module.serve.assert_called_once()
        call_kwargs = mock_sync_module.serve.call_args.kwargs
        assert call_kwargs["requirements"] == ["req1", "req2"]

    @pytest.mark.asyncio
    async def test_model_options_passed_to_serve(self, mock_sync_module):
        """Test that model options are correctly passed to serve function."""
        endpoint = make_chat_endpoint(mock_sync_module)

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Test")],
            temperature=0.7,
            max_tokens=100,
        )

        await endpoint(request)

        # Verify serve was called with model_options
        mock_sync_module.serve.assert_called_once()
        call_kwargs = mock_sync_module.serve.call_args.kwargs
        model_options = call_kwargs["model_options"]
        assert ModelOption.TEMPERATURE in model_options
        assert ModelOption.MAX_NEW_TOKENS in model_options

    @pytest.mark.asyncio
    async def test_openai_params_mapped_to_model_options(self, mock_sync_module):
        """Test that OpenAI parameters are mapped to ModelOption sentinels."""
        endpoint = make_chat_endpoint(mock_sync_module)

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Test")],
            temperature=0.8,
            max_tokens=150,
            seed=42,
        )

        await endpoint(request)

        # Verify parameters are mapped correctly
        mock_sync_module.serve.assert_called_once()
        call_kwargs = mock_sync_module.serve.call_args.kwargs
        model_options = call_kwargs["model_options"]

        assert model_options[ModelOption.TEMPERATURE] == 0.8
        assert model_options[ModelOption.MAX_NEW_TOKENS] == 150
        assert model_options[ModelOption.SEED] == 42


class TestEndpointIntegration:
    """Integration tests for the full endpoint."""

    def test_endpoint_name_set_correctly(self, mock_sync_module):
        """Test that endpoint function name is set correctly."""
        endpoint = make_chat_endpoint(mock_sync_module)
        assert endpoint.__name__ == "chat_test_sync_module_endpoint"

    @pytest.mark.asyncio
    async def test_completion_id_generated(self, mock_sync_module):
        """Test that each response gets a unique completion ID."""
        endpoint = make_chat_endpoint(mock_sync_module)

        request = ChatCompletionRequest(
            model="test-model", messages=[ChatMessage(role="user", content="Test")]
        )

        response1 = await endpoint(request)
        response2 = await endpoint(request)

        assert response1.id.startswith("chatcmpl-")
        assert response2.id.startswith("chatcmpl-")
        assert response1.id != response2.id

    @pytest.mark.asyncio
    async def test_timestamp_generated(self, mock_sync_module):
        """Test that response includes a timestamp."""
        endpoint = make_chat_endpoint(mock_sync_module)

        request = ChatCompletionRequest(
            model="test-model", messages=[ChatMessage(role="user", content="Test")]
        )

        response = await endpoint(request)

        assert isinstance(response.created, int)
        assert response.created > 0
