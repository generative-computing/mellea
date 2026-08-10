# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ComputedModelOutputThunk."""

import copy
import pickle

import pytest

from mellea.core import ComputedModelOutputThunk, ModelOutputThunk
from mellea.stdlib.session import start_session


def test_computed_thunk_initialization():
    """Test that ComputedModelOutputThunk can be initialized from a computed thunk."""
    base_thunk = ModelOutputThunk(value="test output")
    computed_thunk = ComputedModelOutputThunk(base_thunk)

    assert computed_thunk.value == "test output"
    assert computed_thunk.is_computed()
    assert computed_thunk._computed is True


def test_computed_thunk_requires_computed_thunk():
    """Test that ComputedModelOutputThunk requires a computed ModelOutputThunk."""
    uncomputed_thunk = ModelOutputThunk(value=None)

    assert not uncomputed_thunk._computed, (
        "thunk should be uncomputed when passed a None value"
    )

    with pytest.raises(
        ValueError,
        match="ComputedModelOutputThunk requires a computed ModelOutputThunk;",
    ):
        ComputedModelOutputThunk(uncomputed_thunk)


def test_computed_thunk_requires_value():
    """Test that ComputedModelOutputThunk requires a non-None value."""
    # Create a thunk that's computed but has None value (edge case)
    base_thunk = ModelOutputThunk(value="test")
    base_thunk.value = None  # type: ignore

    with pytest.raises(ValueError, match="requires a non-None value"):
        ComputedModelOutputThunk(base_thunk)


async def test_computed_thunk_avalue():
    """Test that avalue() returns immediately for ComputedModelOutputThunk."""
    base_thunk = ModelOutputThunk(value="test output")
    computed_thunk = ComputedModelOutputThunk(base_thunk)

    result = await computed_thunk.avalue()
    assert result == "test output"


async def test_computed_thunk_cannot_stream():
    """Test that astream() raises an error for ComputedModelOutputThunk."""
    base_thunk = ModelOutputThunk(value="test output")
    computed_thunk = ComputedModelOutputThunk(base_thunk)

    with pytest.raises(
        RuntimeError, match="Cannot stream from a ComputedModelOutputThunk"
    ):
        await computed_thunk.astream()


def test_computed_thunk_with_parsed_repr():
    """Test that ComputedModelOutputThunk preserves parsed_repr."""
    base_thunk = ModelOutputThunk(value="test output", parsed_repr="parsed value")
    computed_thunk = ComputedModelOutputThunk(base_thunk)

    assert computed_thunk.value == "test output"
    assert computed_thunk.parsed_repr == "parsed value"


@pytest.mark.ollama
@pytest.mark.e2e
def test_sync_functions_return_computed_thunks():
    """Test that synchronous session functions return ComputedModelOutputThunk."""
    with start_session() as session:
        result = session.instruct("Say 'hello'", strategy=None)

        # The result should be a ComputedModelOutputThunk
        assert isinstance(result, ComputedModelOutputThunk)
        assert result.is_computed()
        assert result.value is not None


@pytest.mark.ollama
@pytest.mark.e2e
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


@pytest.mark.ollama
@pytest.mark.e2e
async def test_async_functions_return_computed_thunks():
    """Test that async session functions return ComputedModelOutputThunk when await_result=True."""
    with start_session() as session:
        result = await session.ainstruct(
            "Say 'hello'", strategy=None, await_result=True
        )

        # The result should be a ComputedModelOutputThunk
        assert isinstance(result, ComputedModelOutputThunk)
        assert result.is_computed()
        assert result.value is not None


def test_computed_thunk_type_distinction():
    """Test that ComputedModelOutputThunk is distinguishable from ModelOutputThunk."""
    base_thunk = ModelOutputThunk(value="test")
    computed = ComputedModelOutputThunk(base_thunk)
    uncomputed = ModelOutputThunk(value=None)

    assert isinstance(computed, ModelOutputThunk)
    assert isinstance(computed, ComputedModelOutputThunk)
    assert isinstance(uncomputed, ModelOutputThunk)
    assert not isinstance(uncomputed, ComputedModelOutputThunk)


def test_computed_thunk_zero_copy_identity():
    """Test that ComputedModelOutputThunk uses zero-copy (same object)."""
    base_thunk = ModelOutputThunk(value="test output")
    computed_thunk = ComputedModelOutputThunk(base_thunk)
    assert computed_thunk is base_thunk


def test_computed_thunk_no_arg_construction_rejected():
    """Constructing without a thunk must raise; the argument is mandatory."""
    with pytest.raises(TypeError):
        ComputedModelOutputThunk()  # type: ignore[call-arg]


def test_computed_thunk_non_thunk_arg_rejected():
    """Passing a non-ModelOutputThunk must raise TypeError, not silently proceed."""
    with pytest.raises(TypeError, match="requires a computed ModelOutputThunk"):
        ComputedModelOutputThunk("not a thunk")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("_computed", False),
        ("_computed", None),
        ("_underlying_value", None),
        ("_underlying_value", 123),
        ("value", None),
        ("value", 123),
    ],
)
def test_computed_thunk_invariant_violating_writes_rejected(field, bad_value):
    """Assignments that would uncompute the thunk or invalidate its value are rejected."""
    computed = ComputedModelOutputThunk(ModelOutputThunk(value="original"))
    with pytest.raises(AttributeError, match="computed invariant"):
        setattr(computed, field, bad_value)
    # The invariant still holds after the rejected write.
    assert computed.value == "original"
    assert computed.is_computed()


def test_computed_thunk_value_may_be_replaced_with_valid_string():
    """The value may be swapped for another valid computed string (react.py relies on this)."""
    computed = ComputedModelOutputThunk(ModelOutputThunk(value="original"))
    computed.value = "replaced"
    assert computed.value == "replaced"
    assert computed.is_computed()
    # The private field route is equally allowed for a valid string.
    computed._underlying_value = "again"
    assert computed.value == "again"


def test_computed_thunk_derived_and_status_fields_writable():
    """Fields outside the invariant guard remain freely mutable after wrapping.

    `parsed_repr` is finalized by sampling strategies (mellea/stdlib/sampling/base.py),
    `_cancelled` is a status flag, and `thinking` is derived output — none are guarded.
    """
    computed = ComputedModelOutputThunk(ModelOutputThunk(value="ok"))
    computed.thinking = "some reasoning"
    computed.parsed_repr = "finalized"
    computed._cancelled = True
    assert computed.thinking == "some reasoning"
    assert computed.parsed_repr == "finalized"
    assert computed._cancelled is True


def test_pre_wrap_value_edit_still_allowed():
    """The guard must not engage on the still-plain ModelOutputThunk before wrapping.

    Setting `.value` on the base thunk prior to wrapping must succeed (the existing
    `test_computed_thunk_requires_value` flow relies on this).
    """
    base = ModelOutputThunk(value="original")
    base.value = "edited"  # allowed: base thunk is not sealed
    computed = ComputedModelOutputThunk(base)
    assert computed.value == "edited"


def test_deepcopy_preserves_computed_subclass():
    """deepcopy of a computed thunk stays a sealed, computed ComputedModelOutputThunk."""
    computed = ComputedModelOutputThunk(ModelOutputThunk(value="hello"))
    dc = copy.deepcopy(computed)

    assert isinstance(dc, ComputedModelOutputThunk)
    assert dc.is_computed()
    assert dc.value == "hello"
    with pytest.raises(AttributeError, match="computed invariant"):
        dc.value = None  # type: ignore[assignment]


def test_copy_preserves_computed_subclass():
    """copy.copy of a computed thunk stays a sealed, computed ComputedModelOutputThunk."""
    computed = ComputedModelOutputThunk(ModelOutputThunk(value="hello"))
    cc = copy.copy(computed)

    assert isinstance(cc, ComputedModelOutputThunk)
    assert cc.is_computed()
    assert cc.value == "hello"
    with pytest.raises(AttributeError, match="computed invariant"):
        cc._computed = False


def test_pickle_roundtrip_preserves_computed_subclass():
    """A pickled/unpickled computed thunk stays a sealed, computed ComputedModelOutputThunk."""
    computed = ComputedModelOutputThunk(ModelOutputThunk(value="hello"))
    restored = pickle.loads(pickle.dumps(computed))

    assert isinstance(restored, ComputedModelOutputThunk)
    assert restored.is_computed()
    assert restored.value == "hello"
    with pytest.raises(AttributeError, match="computed invariant"):
        restored.value = None  # type: ignore[assignment]
