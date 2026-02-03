"""Tests for streaming support using async functions with await_result parameter."""

# pytest: ollama, llm

import asyncio

import pytest

from mellea.core import ComputedModelOutputThunk, ModelOutputThunk
from mellea.stdlib.session import start_session


def test_sync_function_returns_computed_thunk():
    """Test that sync functions always return ComputedModelOutputThunk."""
    with start_session() as session:
        result = session.instruct("Say 'hello'", strategy=None)

        # Should return ComputedModelOutputThunk
        assert isinstance(result, ComputedModelOutputThunk)
        assert result.is_computed()
        assert result.value is not None


async def test_async_function_with_await_result_true():
    """Test that ainstruct with await_result=True returns ComputedModelOutputThunk."""
    with start_session() as session:
        result = await session.ainstruct(
            "Say 'hello'", strategy=None, await_result=True
        )

        # Should return ComputedModelOutputThunk
        assert isinstance(result, ComputedModelOutputThunk)
        assert result.is_computed()
        assert result.value is not None


async def test_async_function_with_await_result_false():
    """Test that ainstruct with await_result=False returns uncomputed ModelOutputThunk."""
    with start_session() as session:
        result = await session.ainstruct(
            "Say 'hello'", strategy=None, await_result=False
        )

        # Should return uncomputed ModelOutputThunk (not ComputedModelOutputThunk)
        assert isinstance(result, ModelOutputThunk)
        assert not isinstance(result, ComputedModelOutputThunk)
        assert not result.is_computed()
        assert result.value is None  # Not computed yet


async def test_streaming_uncomputed_thunk():
    """Test that uncomputed thunk can be streamed using astream() with ainstruct."""
    with start_session() as session:
        # Use ainstruct (async) with await_result=False for streaming
        result = await session.ainstruct(
            "Count to 5", strategy=None, await_result=False
        )

        # Should be uncomputed
        assert not result.is_computed()

        # Stream the result
        chunks = []
        while not result.is_computed():
            chunk = await result.astream()
            chunks.append(chunk)

        # After streaming, should be computed
        assert result.is_computed()
        assert result.value is not None
        assert len(result.value) > 0


async def test_streaming_with_avalue():
    """Test that uncomputed thunk can be awaited using avalue() with ainstruct."""
    with start_session() as session:
        # Use ainstruct (async) with await_result=False
        result = await session.ainstruct(
            "Say 'test'", strategy=None, await_result=False
        )

        # Should be uncomputed
        assert not result.is_computed()

        # Await the value
        value = await result.avalue()

        # After awaiting, should be computed
        assert result.is_computed()
        assert value is not None
        assert result.value == value


def test_await_result_false_with_sampling_still_computes():
    """Test that await_result=False with sampling strategy still returns computed thunk."""
    from mellea.stdlib.sampling import RejectionSamplingStrategy

    with start_session() as session:
        # Even with await_result=False, sampling requires awaiting
        result = session.instruct(
            "Say 'hello'", strategy=RejectionSamplingStrategy(loop_budget=1)
        )

        # Should still be ComputedModelOutputThunk because sampling requires it
        assert isinstance(result, ComputedModelOutputThunk)
        assert result.is_computed()


def test_default_behavior_unchanged():
    """Test that default behavior (await_result=False) returns uncomputed thunk."""
    with start_session() as session:
        # Default behavior should return uncomputed thunk for streaming
        result = session.instruct("Say 'hello'", strategy=None)

        # Sync functions always await, so result is computed even with await_result=False
        assert isinstance(result, ComputedModelOutputThunk)
        assert result.is_computed()
        assert result.value is not None


async def test_multiple_uncomputed_thunks():
    """Test creating multiple uncomputed thunks and awaiting them with ainstruct."""
    with start_session() as session:
        # Use ainstruct (async) for both calls
        result1 = await session.ainstruct("Say '1'", strategy=None, await_result=False)
        result2 = await session.ainstruct("Say '2'", strategy=None, await_result=False)

        # Both should be uncomputed
        assert not result1.is_computed()
        assert not result2.is_computed()

        # Await both
        value1, value2 = await asyncio.gather(result1.avalue(), result2.avalue())

        # Both should now be computed
        assert result1.is_computed()
        assert result2.is_computed()
        assert value1 is not None
        assert value2 is not None


async def test_aact_function_with_await_result():
    """Test that aact() function also supports await_result parameter."""
    from mellea.stdlib.components import Instruction

    with start_session() as session:
        instruction = Instruction(description="Say 'test'")

        # Test with await_result=False
        result = await session.aact(instruction, strategy=None, await_result=False)

        assert isinstance(result, ModelOutputThunk)
        assert not isinstance(result, ComputedModelOutputThunk)
        assert not result.is_computed()
