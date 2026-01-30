"""Tests for ComputedModelOutputThunk."""

# pytest: ollama, llm

import pytest

from mellea.core import ComputedModelOutputThunk, ModelOutputThunk
from mellea.stdlib.session import start_session


def test_computed_thunk_initialization():
    """Test that ComputedModelOutputThunk can be initialized with a value."""
    thunk = ComputedModelOutputThunk(value="test output")

    assert thunk.value == "test output"
    assert thunk.is_computed()
    assert thunk._computed is True


def test_computed_thunk_requires_value():
    """Test that ComputedModelOutputThunk requires a non-None value."""
    with pytest.raises(ValueError, match="requires a non-None value"):
        ComputedModelOutputThunk(value=None)  # type: ignore


async def test_computed_thunk_avalue():
    """Test that avalue() returns immediately for ComputedModelOutputThunk."""
    thunk = ComputedModelOutputThunk(value="test output")

    result = await thunk.avalue()
    assert result == "test output"


async def test_computed_thunk_cannot_stream():
    """Test that astream() raises an error for ComputedModelOutputThunk."""
    thunk = ComputedModelOutputThunk(value="test output")

    with pytest.raises(
        RuntimeError, match="Cannot stream from a ComputedModelOutputThunk"
    ):
        await thunk.astream()


def test_computed_thunk_with_parsed_repr():
    """Test that ComputedModelOutputThunk preserves parsed_repr."""
    thunk = ComputedModelOutputThunk(value="test output", parsed_repr="parsed value")

    assert thunk.value == "test output"
    assert thunk.parsed_repr == "parsed value"


def test_sync_functions_return_computed_thunks():
    """Test that synchronous session functions return ComputedModelOutputThunk."""
    with start_session() as session:
        result = session.instruct("Say 'hello'", strategy=None)

        # The result should be a ComputedModelOutputThunk
        assert isinstance(result, ComputedModelOutputThunk)
        assert result.is_computed()
        assert result.value is not None


def test_sync_functions_with_sampling_return_computed_thunks():
    """Test that synchronous functions with sampling return ComputedModelOutputThunk."""
    from mellea.stdlib.sampling import RejectionSamplingStrategy

    with start_session() as session:
        result = session.instruct(
            "Say 'hello'", strategy=RejectionSamplingStrategy(loop_budget=1)
        )

        # The result should be a ComputedModelOutputThunk
        assert isinstance(result, ComputedModelOutputThunk)
        assert result.is_computed()
        assert result.value is not None


async def test_async_functions_return_computed_thunks():
    """Test that async session functions return ComputedModelOutputThunk after awaiting."""
    with start_session() as session:
        result = await session.ainstruct("Say 'hello'", strategy=None)

        # The result should be a ComputedModelOutputThunk
        assert isinstance(result, ComputedModelOutputThunk)
        assert result.is_computed()
        assert result.value is not None


def test_computed_thunk_type_distinction():
    """Test that ComputedModelOutputThunk is distinguishable from ModelOutputThunk."""
    computed = ComputedModelOutputThunk(value="test")
    uncomputed = ModelOutputThunk(value=None)

    assert isinstance(computed, ModelOutputThunk)
    assert isinstance(computed, ComputedModelOutputThunk)
    assert isinstance(uncomputed, ModelOutputThunk)
    assert not isinstance(uncomputed, ComputedModelOutputThunk)
