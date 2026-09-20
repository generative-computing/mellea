# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextvars
import gc
import multiprocessing
import sys
import threading
import time
import warnings
from unittest import mock

import pytest

import mellea.helpers.event_loop_helper as elh
import mellea.helpers.event_loop_helper as elh2


def test_event_loop_handler_singleton():
    assert elh.__event_loop_handler is not None
    assert elh.__event_loop_handler == elh2.__event_loop_handler


def test_run_async_in_thread():
    async def testing() -> bool:
        return True

    assert elh._run_async_in_thread(testing()), "somehow the wrong value was returned"


def test_schedule_async_in_thread_returns_before_the_coroutine_finishes():
    """The scheduling variant hands work to the loop without waiting on it."""
    release = threading.Event()
    finished = threading.Event()

    async def work() -> str:
        await asyncio.to_thread(release.wait, 5.0)
        finished.set()
        return "done"

    future = elh._schedule_async_in_thread(work())
    assert not finished.is_set(), "scheduling should not have waited for the coroutine"

    release.set()
    assert future.result(timeout=5.0) == "done"
    assert finished.is_set()


def test_schedule_async_in_thread_closes_coroutine_on_scheduling_failure():
    """A failure before the coroutine is scheduled must close it, not leak it."""
    started = False

    async def never_scheduled() -> None:
        nonlocal started
        started = True

    co = never_scheduled()

    with pytest.raises(RuntimeError, match="boom"):
        with mock.patch.object(
            elh._EventLoopHandler, "_reinit_if_forked", side_effect=RuntimeError("boom")
        ):
            elh._schedule_async_in_thread(co)

    # Same as the blocking variant: a closed-but-never-started coroutine never runs
    # its body, and re-sending into it raises rather than warning about a leak.
    assert started is False
    with pytest.raises(RuntimeError):
        co.send(None)


def test_run_async_in_thread_propagates_calling_thread_contextvars():
    """The calling thread's contextvars are visible inside the new Task."""
    var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
        "test_var", default=None
    )
    var.set("from-calling-thread")

    async def read_var() -> str | None:
        return var.get()

    assert elh._run_async_in_thread(read_var()) == "from-calling-thread"


def test_run_async_in_thread_does_not_leak_task_mutations_back():
    """Mutations to contextvars inside the Task don't reflect on the caller."""
    var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
        "test_var2", default="caller-value"
    )

    async def mutate() -> None:
        var.set("task-mutation")

    elh._run_async_in_thread(mutate())
    assert var.get() == "caller-value"


def test_run_async_in_thread_same_loop_recursion_propagates_contextvars():
    """Same-loop recursion still propagates the caller's contextvars."""
    var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
        "test_var_recursive", default=None
    )
    var.set("outer-value")

    async def inner_read() -> str | None:
        return var.get()

    async def outer() -> str | None:
        # We're on the singleton's event loop now. A nested call from here
        # takes the same-loop branch in _EventLoopHandler.__call__.
        return elh._run_async_in_thread(inner_read())

    assert elh._run_async_in_thread(outer()) == "outer-value"


def test_event_loop_handler_init_and_del():
    # Do not ever instantiate this manually. Only doing here for testing.
    new_event_loop_handler = elh._EventLoopHandler()

    async def testing() -> int:
        return 1

    out = new_event_loop_handler(testing())
    assert out == 1, "somehow the wrong value was returned"

    del new_event_loop_handler

    # Make sure this didn't delete the actual singleton.
    assert elh.__event_loop_handler is not None


def test_nested_call_does_not_leak_threads():
    """Repeated same-loop nested calls must not leave threads running.

    Regression guard for issue #349: the nested branch in
    `_EventLoopHandler.__call__` used to build a fresh `_EventLoopHandler()`
    per call and drop it, leaking a thread (plus its loop and fds) each time.
    """

    async def inner() -> int:
        return 1

    async def outer() -> int:
        # Runs on the singleton's event loop; a nested call from here takes
        # the same-loop branch in _EventLoopHandler.__call__.
        return elh._run_async_in_thread(inner())

    baseline = threading.active_count()
    for _ in range(5):
        assert elh._run_async_in_thread(outer()) == 1
    assert threading.active_count() == baseline


def test_close_event_loop_closes_thread_and_loop_and_is_idempotent():
    # Do not ever instantiate this manually. Only doing here for testing.
    handler = elh._EventLoopHandler()
    loop = handler._event_loop
    thread = handler._thread

    handler._close_event_loop()

    assert loop.is_closed()
    assert not thread.is_alive()
    assert handler._event_loop is None

    # A second call (mirrors __del__ running after an explicit close) must
    # be a no-op, not raise, and not attempt to close the loop again.
    handler._close_event_loop()
    assert loop.is_closed()


def test_del_skips_its_waits_during_interpreter_finalization(monkeypatch):
    """A daemon thread that can't be rescheduled must not be waited on at exit."""
    # Do not ever instantiate this manually. Only doing here for testing.
    handler = elh._EventLoopHandler()
    loop = handler._event_loop
    thread = handler._thread

    blocked = threading.Event()
    release = threading.Event()

    def hold_the_loop():
        blocked.set()
        release.wait(10.0)

    # Stands in for a loop thread frozen by finalization: nothing this handler
    # schedules can run, so both waits would burn their full timeout.
    loop.call_soon_threadsafe(hold_the_loop)
    assert blocked.wait(timeout=5.0)

    monkeypatch.setattr(sys, "is_finalizing", lambda: True)
    try:
        start = time.monotonic()
        handler.__del__()
        assert time.monotonic() - start < 0.5, "__del__ waited on a frozen loop thread"
    finally:
        release.set()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5.0)
        loop.close()


def test_submit_after_shutdown_raises_runtime_error():
    """A cleared loop reference must not surface as an AttributeError from asyncio."""
    # Do not ever instantiate this manually. Only doing here for testing.
    handler = elh._EventLoopHandler()
    handler._close_event_loop()

    async def work() -> None:
        return None

    co = work()
    with pytest.raises(RuntimeError, match="shut down"):
        handler.submit(co)

    # Same contract as any other scheduling failure: `co` is closed, not leaked.
    with pytest.raises(RuntimeError):
        co.send(None)


def test_call_after_shutdown_raises_runtime_error():
    """Without an explicit check this would run on a throwaway nested loop instead."""
    # Do not ever instantiate this manually. Only doing here for testing.
    handler = elh._EventLoopHandler()
    handler._close_event_loop()

    ran = False

    async def work() -> None:
        nonlocal ran
        ran = True

    co = work()
    with pytest.raises(RuntimeError, match="shut down"):
        handler(co)

    assert ran is False, "the coroutine ran despite the handler being shut down"
    with pytest.raises(RuntimeError):
        co.send(None)


def test_run_async_in_thread_closes_coroutine_on_scheduling_failure():
    """A failure before the coroutine is scheduled must close it, not leak it."""
    closed = False

    async def never_scheduled() -> None:
        nonlocal closed
        closed = True

    co = never_scheduled()

    with pytest.raises(RuntimeError, match="boom"):
        with mock.patch.object(
            elh, "get_current_event_loop", side_effect=RuntimeError("boom")
        ):
            elh._run_async_in_thread(co)

    # A closed-but-never-started coroutine raises StopIteration internally,
    # which close() swallows; the body (setting `closed`) never runs.
    assert closed is False
    with pytest.raises(RuntimeError):
        co.send(None)


def test_run_async_in_thread_closes_wrapped_coroutine_on_scheduling_failure():
    """A scheduling failure must also close the internal `_wrapped()` coroutine.

    Regression guard: closing `co` alone isn't enough — `_wrapped()`'s own
    coroutine object is created before `run_coroutine_threadsafe` is called, so
    if scheduling itself raises, `_wrapped()` never starts and never gets to
    `await co`, leaking a second "coroutine was never awaited" warning distinct
    from `co`'s.
    """

    async def caller_coroutine() -> None:
        return None

    co = caller_coroutine()

    with mock.patch.object(
        elh.asyncio,
        "run_coroutine_threadsafe",
        side_effect=RuntimeError("scheduling failed"),
    ):
        with pytest.raises(RuntimeError, match="scheduling failed"):
            elh._run_async_in_thread(co)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        del co
        gc.collect()

    unawaited = [
        w
        for w in caught
        if issubclass(w.category, RuntimeWarning) and "never awaited" in str(w.message)
    ]
    assert unawaited == [], f"leaked unawaited coroutine(s): {unawaited}"


def test_run_async_in_thread_reraises_coroutine_exception():
    """Exceptions raised by the coroutine itself still propagate normally."""

    async def boom() -> None:
        raise ValueError("from inside the coroutine")

    with pytest.raises(ValueError, match="from inside the coroutine"):
        elh._run_async_in_thread(boom())


def test_event_loop_handler_with_forking():
    """Importing mellea before fork must not crash the child process."""

    ctx = multiprocessing.get_context("fork")

    def child():
        import mellea.helpers.event_loop_helper as elh

        async def hello():
            return 42

        result = elh._run_async_in_thread(hello())
        assert result == 42

    p = ctx.Process(target=child)

    try:
        p.start()
        p.join(timeout=15)
        assert p.exitcode == 0, (
            f"Child process failed after fork (exit code: {p.exitcode if p.exitcode is not None else 'timed out'})"
        )

    finally:
        # Make sure we always clean up the process.
        if p.is_alive():
            p.kill()
            p.join(timeout=15)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
