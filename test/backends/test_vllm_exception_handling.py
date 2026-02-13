"""Non-generative tests for vLLM backend exception handling.

These tests verify that the vLLM backend properly handles exception chunks
passed to the processing() function during astream(). This covers the code
path where an exception occurs during generation and is yielded as a chunk.

See: https://github.com/generative-computing/mellea/issues/432
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from mellea.core import CBlock, ModelOutputThunk
from mellea.core.base import GenerateType


class TestVLLMExceptionHandling:
    """Test exception handling in vLLM backend's processing function."""

    @pytest.fixture
    def mock_vllm_chunk(self):
        """Factory to create mock vLLM RequestOutput chunks."""

        def _create(text: str):
            mock_output = Mock()
            mock_output.outputs = [Mock()]
            mock_output.outputs[0].text = text
            return mock_output

        return _create

    @pytest.fixture
    def model_output_thunk(self):
        """Create a fresh ModelOutputThunk for testing."""
        mot = ModelOutputThunk(value=None)
        mot._underlying_value = None
        return mot

    @pytest.mark.asyncio
    async def test_partial_output_preserved_on_exception(
        self, mock_vllm_chunk, model_output_thunk
    ):
        """Test that partial output is preserved when an exception occurs mid-stream.

        Simulates a realistic scenario where:
        1. Several valid chunks are processed successfully (partial output accumulated)
        2. An exception occurs (e.g., timeout, OOM, CUDA error)
        3. The processing() function should:
           - Not crash on the exception chunk
           - Preserve the partial output from valid chunks
           - Allow the framework to re-raise the original exception

        The fix should add to processing():
            if isinstance(chunk, Exception):
                return
        """
        try:
            from mellea.backends.vllm import LocalVLLMBackend
        except ImportError:
            pytest.skip("vLLM backend not available")

        backend = Mock(spec=LocalVLLMBackend)
        backend.processing = LocalVLLMBackend.processing

        # Simulate successful partial generation
        valid_chunks = [
            mock_vllm_chunk("Hello"),
            mock_vllm_chunk(", "),
            mock_vllm_chunk("world"),
        ]

        for chunk in valid_chunks:
            await backend.processing(backend, model_output_thunk, chunk)

        # Verify partial output accumulated correctly
        assert model_output_thunk._underlying_value == "Hello, world"

        # Now simulate a mid-stream failure (e.g., timeout, OOM)
        timeout_error = TimeoutError("Generation timed out after 30s")

        # Processing should handle the exception gracefully
        await backend.processing(backend, model_output_thunk, timeout_error)

        # Partial output should be preserved (not corrupted or lost)
        assert model_output_thunk._underlying_value == "Hello, world", (
            "Partial output should be preserved after exception"
        )

    @pytest.mark.asyncio
    async def test_various_exception_types_handled(self, model_output_thunk):
        """Test that various exception types are handled gracefully.

        Different failures can occur during generation:
        - TimeoutError: Generation took too long
        - MemoryError: OOM on GPU or CPU
        - RuntimeError: CUDA errors, model failures
        - Exception: Generic failures

        All should be skipped by processing() without crashing.
        """
        try:
            from mellea.backends.vllm import LocalVLLMBackend
        except ImportError:
            pytest.skip("vLLM backend not available")

        backend = Mock(spec=LocalVLLMBackend)
        backend.processing = LocalVLLMBackend.processing

        exception_types = [
            TimeoutError("Generation timed out"),
            MemoryError("CUDA out of memory"),
            RuntimeError("CUDA error: device-side assert triggered"),
            Exception("Unknown model failure"),
        ]

        for exc in exception_types:
            # Reset the MOT for each test
            model_output_thunk._underlying_value = None

            # Should not raise - should return early
            await backend.processing(backend, model_output_thunk, exc)

            # Value should remain unchanged
            assert model_output_thunk._underlying_value is None, (
                f"Exception {type(exc).__name__} should be skipped, not processed"
            )


class TestExceptionReraising:
    """Test that exceptions are properly re-raised through astream().

    These tests verify the complete flow:
    1. Valid chunks are processed and accumulated
    2. An exception chunk arrives
    3. processing() skips the exception (doesn't crash)
    4. The framework re-raises the original exception after cleanup
    """

    @pytest.fixture
    def mock_vllm_chunk(self):
        """Factory to create mock vLLM RequestOutput chunks."""

        def _create(text: str):
            mock_output = Mock()
            mock_output.outputs = [Mock()]
            mock_output.outputs[0].text = text
            return mock_output

        return _create

    @pytest.mark.asyncio
    async def test_exception_reraises_with_original_message(self, mock_vllm_chunk):
        """Test that the original exception is re-raised after astream() cleanup.

        This tests the full flow through astream() with the real processing()
        function, verifying that:
        1. Valid chunks are processed correctly
        2. Exception chunks don't crash processing()
        3. The original exception is re-raised with its message intact
        """
        try:
            from mellea.backends.vllm import LocalVLLMBackend
        except ImportError:
            pytest.skip("vLLM backend not available")

        # Create a mock backend with the real processing method
        backend = Mock(spec=LocalVLLMBackend)
        backend.processing = LocalVLLMBackend.processing

        # Create a MOT configured for async streaming
        mot = ModelOutputThunk(value=None)
        mot._computed = False
        mot._generate = None
        mot._generate_extra = None
        mot._generate_type = GenerateType.ASYNC
        mot._chunk_size = 1
        mot._action = CBlock("test")
        mot._model_options = {}

        # Wire up the real processing function
        async def process_wrapper(mot_arg, chunk):
            await backend.processing(backend, mot_arg, chunk)

        mot._process = process_wrapper
        mot._post_process = AsyncMock()

        # Populate the queue: valid chunks then an exception
        valid_chunks = [
            mock_vllm_chunk("Hello"),
            mock_vllm_chunk(", "),
            mock_vllm_chunk("world"),
        ]
        original_error = RuntimeError("CUDA error: device-side assert triggered")

        for chunk in valid_chunks:
            await mot._async_queue.put(chunk)
        await mot._async_queue.put(original_error)

        # astream() should re-raise the original exception
        with pytest.raises(
            RuntimeError, match="CUDA error: device-side assert triggered"
        ):
            await mot.astream()

        # Verify partial output was accumulated before the error
        assert mot._underlying_value == "Hello, world", (
            "Partial output should be accumulated before exception"
        )

    @pytest.mark.asyncio
    async def test_exception_reraises_after_cleanup(self, mock_vllm_chunk):
        """Test that post_process (cleanup) runs before exception is re-raised.

        This verifies that telemetry spans and other cleanup happen even
        when an exception occurs, and the exception is re-raised afterward.
        """
        try:
            from mellea.backends.vllm import LocalVLLMBackend
        except ImportError:
            pytest.skip("vLLM backend not available")

        backend = Mock(spec=LocalVLLMBackend)
        backend.processing = LocalVLLMBackend.processing

        mot = ModelOutputThunk(value=None)
        mot._computed = False
        mot._generate = None
        mot._generate_extra = None
        mot._generate_type = GenerateType.ASYNC
        mot._chunk_size = 1
        mot._action = CBlock("test")
        mot._model_options = {}

        async def process_wrapper(mot_arg, chunk):
            await backend.processing(backend, mot_arg, chunk)

        mot._process = process_wrapper

        # Track whether post_process was called
        post_process_called = False

        async def tracking_post_process(mot_arg):
            nonlocal post_process_called
            post_process_called = True

        mot._post_process = tracking_post_process

        # Queue: one valid chunk then an exception
        await mot._async_queue.put(mock_vllm_chunk("partial"))
        await mot._async_queue.put(MemoryError("CUDA out of memory"))

        # Should raise the original exception
        with pytest.raises(MemoryError, match="CUDA out of memory"):
            await mot.astream()

        # post_process should have been called (cleanup happened)
        assert post_process_called, (
            "post_process should be called for cleanup before re-raising exception"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
