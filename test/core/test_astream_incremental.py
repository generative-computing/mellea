"""Tests for ModelOutputThunk.astream() incremental return behavior.

Tests the changes around lines 278-281 and 352-356 in mellea/core/base.py
that ensure astream() returns only new content added since the beginning of
each astream() call, not the entire accumulated value.
"""

import pytest

from mellea.backends import ModelOption
from mellea.core import CBlock, ModelOutputThunk
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.session import start_session


@pytest.mark.ollama
@pytest.mark.llm
async def test_astream_returns_incremental_chunks():
    """Test that astream() returns only new content, not accumulated content.

    This tests the fix where beginning_length is captured at the start of
    astream() and the return value is sliced to only include new content.
    """
    session = start_session()
    model_opts = {ModelOption.STREAM: True}

    mot, _ = await session.backend.generate_from_context(
        CBlock("Count from 1 to 5 slowly."), SimpleContext(), model_options=model_opts
    )

    # First astream call - should return content from beginning
    chunk1 = await mot.astream()
    assert chunk1 is not None, "First chunk should not be None"
    assert len(chunk1) > 0, "First chunk should have content"

    # Second astream call - should return only NEW content since first call
    chunk2 = await mot.astream()

    if not mot.is_computed():
        # If not computed, chunk2 should be new content only
        assert chunk2 is not None, "Second chunk should not be None if not computed"

        # The key test: chunk2 should NOT start with chunk1
        # (it should be incremental, not accumulated)
        if len(chunk2) > 0:
            # chunk2 should be different from chunk1 (new content)
            assert chunk2 != chunk1, (
                "Second chunk should be different from first (incremental)"
            )

            # Get final value
            final_val = await mot.avalue()

            # Final value should contain both chunks in order
            assert final_val.startswith(chunk1), (
                "Final value should start with first chunk"
            )
            # The concatenation of chunks should be a prefix of or equal to final value
            accumulated = chunk1 + chunk2
            assert final_val.startswith(accumulated) or accumulated.startswith(
                final_val
            ), "Accumulated chunks should match final value progression"
    else:
        # If computed after first astream, chunk2 should be empty or the remainder
        final_val = await mot.avalue()
        # chunk1 should be a prefix of final value
        assert final_val.startswith(chunk1), "Final value should start with first chunk"


@pytest.mark.ollama
@pytest.mark.llm
async def test_astream_multiple_calls_accumulate_correctly():
    """Test that multiple astream() calls accumulate to the final value.

    Note: The final astream() call that marks the thunk as computed returns
    the FULL value (line 350 in base.py), not just the incremental part.
    """
    session = start_session()
    model_opts = {ModelOption.STREAM: True}

    mot, _ = await session.backend.generate_from_context(
        CBlock("Write a short sentence."), SimpleContext(), model_options=model_opts
    )

    accumulated = ""
    chunks = []

    # Stream until computed
    while not mot.is_computed():
        chunk = await mot.astream()
        if chunk:
            chunks.append(chunk)
            # Only accumulate if this wasn't the final (completing) chunk
            if not mot.is_computed():
                accumulated += chunk

        # Safety: don't loop forever
        if len(chunks) > 100:
            break

    # Get final value
    final_val = await mot.avalue()

    # The last chunk should be the full value when computed
    if len(chunks) > 0:
        assert chunks[-1] == final_val, (
            f"Last chunk (when computed) should be full value.\n"
            f"Last chunk: {chunks[-1]!r}\n"
            f"Final: {final_val!r}"
        )

    # All chunks except the last should be incremental
    if len(chunks) > 1:
        incremental_accumulated = "".join(chunks[:-1])
        assert final_val.startswith(incremental_accumulated), (
            f"Incremental chunks should be prefix of final value.\n"
            f"Accumulated: {incremental_accumulated!r}\n"
            f"Final: {final_val!r}"
        )


@pytest.mark.ollama
@pytest.mark.llm
async def test_astream_beginning_length_tracking():
    """Test that beginning_length is correctly tracked across astream calls.

    This specifically tests the logic at lines 278-281 where beginning_length
    is captured at the start of each astream() call.
    """
    session = start_session()
    model_opts = {ModelOption.STREAM: True}

    mot, _ = await session.backend.generate_from_context(
        CBlock("Say hello."), SimpleContext(), model_options=model_opts
    )

    # First call: beginning_length should be 0 (or length of any pre-existing value)
    chunk1 = await mot.astream()

    # Second call: beginning_length should be captured at start of this call
    chunk2 = await mot.astream()

    if chunk2 and len(chunk2) > 0:
        # chunk2 should not include chunk1's content
        # This verifies the slicing logic at lines 352-356
        if chunk1:
            assert not chunk2.startswith(chunk1), (
                "Second chunk should not start with first chunk (should be incremental)"
            )


@pytest.mark.ollama
@pytest.mark.llm
async def test_astream_empty_beginning():
    """Test astream when _underlying_value starts as None."""
    session = start_session()
    model_opts = {ModelOption.STREAM: True}

    mot, _ = await session.backend.generate_from_context(
        CBlock("Hi"), SimpleContext(), model_options=model_opts
    )

    # At the start, _underlying_value might be None
    # beginning_length should be 0 in this case (line 280)
    chunk = await mot.astream()

    assert chunk is not None, "Should get a chunk even when starting from None"

    # When beginning_length is 0, should return full _underlying_value (line 354)
    if mot._underlying_value:
        assert chunk == mot._underlying_value or mot._underlying_value.startswith(
            chunk
        ), "When beginning_length is 0, should return the full underlying value"


@pytest.mark.ollama
@pytest.mark.llm
async def test_astream_computed_returns_full_value():
    """Test that astream returns full value when already computed."""
    # Create a pre-computed thunk
    mot = ModelOutputThunk(value="Hello, world!")
    mot._computed = True

    # astream should return the full value immediately (line 272)
    result = await mot.astream()

    assert result == "Hello, world!", "Computed thunk should return full value"


@pytest.mark.ollama
@pytest.mark.llm
async def test_astream_final_call_returns_full_value():
    """Test that the final astream call returns the full value when computed.

    This tests the behavior at line 350 in base.py where the final call
    (when _computed becomes True) returns the full _underlying_value.
    """
    session = start_session()
    model_opts = {ModelOption.STREAM: True}

    mot, _ = await session.backend.generate_from_context(
        CBlock("Count: 1, 2, 3"), SimpleContext(), model_options=model_opts
    )

    chunks = []

    # Collect all chunks
    while not mot.is_computed():
        chunk = await mot.astream()
        if chunk:
            chunks.append(chunk)

        if len(chunks) > 100:  # Safety
            break

    # Get final value
    final_val = await mot.avalue()

    # The last chunk should be the full value (not incremental)
    if len(chunks) > 0:
        assert chunks[-1] == final_val, (
            f"Final chunk should be the complete value.\n"
            f"Last chunk: {chunks[-1]!r}\n"
            f"Final value: {final_val!r}"
        )

    # All chunks before the last should be incremental (non-overlapping)
    for i in range(len(chunks) - 2):  # Exclude the last chunk
        for j in range(i + 1, len(chunks) - 1):  # Exclude the last chunk
            # Earlier incremental chunks shouldn't be prefixes of later ones
            if chunks[j] and chunks[i]:
                assert not chunks[j].startswith(chunks[i]), (
                    f"Incremental chunk {j} should not start with chunk {i}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
