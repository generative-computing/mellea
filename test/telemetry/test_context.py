"""Unit tests for mellea.telemetry.context."""

import asyncio
import logging

import pytest

from mellea.telemetry.context import (
    MelleaContextFilter,
    _model_id,
    _request_id,
    _sampling_iteration,
    _session_id,
    async_with_context,
    generate_request_id,
    get_current_context,
    get_model_id,
    get_request_id,
    get_sampling_iteration,
    get_session_id,
    with_context,
)

# ---------------------------------------------------------------------------
# get_* helpers — default to None
# ---------------------------------------------------------------------------


def test_getters_return_none_by_default():
    assert get_session_id() is None
    assert get_request_id() is None
    assert get_model_id() is None
    assert get_sampling_iteration() is None


# ---------------------------------------------------------------------------
# get_current_context — omits None values
# ---------------------------------------------------------------------------


def test_get_current_context_empty():
    assert get_current_context() == {}


def test_get_current_context_partial():
    with with_context(session_id="s-1"):
        ctx = get_current_context()
    assert ctx == {"session_id": "s-1"}


# ---------------------------------------------------------------------------
# with_context — sets and restores
# ---------------------------------------------------------------------------


def test_with_context_sets_values():
    with with_context(session_id="s-1", model_id="granite"):
        assert get_session_id() == "s-1"
        assert get_model_id() == "granite"
    # Restored after exit
    assert get_session_id() is None
    assert get_model_id() is None


def test_with_context_nested():
    with with_context(session_id="outer"):
        with with_context(session_id="inner"):
            assert get_session_id() == "inner"
        assert get_session_id() == "outer"
    assert get_session_id() is None


def test_with_context_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown context fields"):
        with with_context(unknown_field="x"):
            pass


def test_with_context_restores_on_exception():
    try:
        with with_context(session_id="boom"):
            raise RuntimeError("oops")
    except RuntimeError:
        pass
    assert get_session_id() is None


# ---------------------------------------------------------------------------
# async_with_context — same semantics in async code
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_with_context_sets_values():
    async with async_with_context(request_id="r-1", model_id="llm"):
        assert get_request_id() == "r-1"
        assert get_model_id() == "llm"
    assert get_request_id() is None
    assert get_model_id() is None


@pytest.mark.asyncio
async def test_async_tasks_isolated():
    """Each asyncio Task should see its own ContextVar values."""
    results: list[str | None] = []

    async def task_a():
        async with async_with_context(session_id="task-a"):
            await asyncio.sleep(0)
            results.append(get_session_id())

    async def task_b():
        async with async_with_context(session_id="task-b"):
            await asyncio.sleep(0)
            results.append(get_session_id())

    await asyncio.gather(task_a(), task_b())
    assert sorted(results) == ["task-a", "task-b"]


# ---------------------------------------------------------------------------
# generate_request_id
# ---------------------------------------------------------------------------


def test_generate_request_id_unique():
    ids = {generate_request_id() for _ in range(100)}
    assert len(ids) == 100


def test_generate_request_id_is_hex():
    rid = generate_request_id()
    assert isinstance(rid, str)
    int(rid, 16)  # should not raise


# ---------------------------------------------------------------------------
# MelleaContextFilter
# ---------------------------------------------------------------------------


def test_mellea_context_filter_injects_fields():
    filt = MelleaContextFilter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="hello",
        args=(),
        exc_info=None,
    )
    with with_context(session_id="s-99", model_id="my-model"):
        filt.filter(record)

    assert record.session_id == "s-99"  # type: ignore[attr-defined]
    assert record.model_id == "my-model"  # type: ignore[attr-defined]


def test_mellea_context_filter_does_not_overwrite_existing_attribute():
    filt = MelleaContextFilter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.session_id = "pre-existing"  # type: ignore[attr-defined]
    with with_context(session_id="new-value"):
        filt.filter(record)

    assert record.session_id == "pre-existing"  # type: ignore[attr-defined]


def test_mellea_context_filter_allows_all_records():
    filt = MelleaContextFilter()
    record = logging.LogRecord(
        name="test",
        level=logging.DEBUG,
        pathname="",
        lineno=0,
        msg="hi",
        args=(),
        exc_info=None,
    )
    assert filt.filter(record) is True


def test_mellea_context_filter_empty_context():
    filt = MelleaContextFilter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="hi",
        args=(),
        exc_info=None,
    )
    filt.filter(record)
    assert not hasattr(record, "session_id")
    assert not hasattr(record, "request_id")


# ---------------------------------------------------------------------------
# sampling_iteration ContextVar directly
# ---------------------------------------------------------------------------


def test_sampling_iteration_var():
    tok = _sampling_iteration.set(3)
    assert get_sampling_iteration() == 3
    _sampling_iteration.reset(tok)
    assert get_sampling_iteration() is None


def test_get_current_context_multiple_fields():
    with with_context(
        session_id="s-1", request_id="r-1", model_id="m", sampling_iteration=2
    ):
        ctx = get_current_context()
    assert ctx == {
        "session_id": "s-1",
        "request_id": "r-1",
        "model_id": "m",
        "sampling_iteration": 2,
    }
    assert get_current_context() == {}


def test_with_context_sampling_iteration_cleaned_up():
    """sampling_iteration must be None after with_context exits (simulates success-path cleanup)."""
    with with_context(sampling_iteration=1):
        assert get_sampling_iteration() == 1
    assert get_sampling_iteration() is None
