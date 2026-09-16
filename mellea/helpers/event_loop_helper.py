# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper for event loop management. Allows consistently running async generate requests in sync code."""

import asyncio
import concurrent.futures
import contextvars
import os
import sys
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

from .async_helpers import get_current_event_loop

R = TypeVar("R")


class _EventLoopHandler:
    """A class that handles the event loop for Mellea code. Do not directly instantiate this. Use `_run_async_in_thread`."""

    def _event_loop_setup(self):
        """Sets up the event loop and thread."""
        # This code lives in a helper function since both __init__ and _reinit_if_forked
        # will need to use it.
        self._pid = os.getpid()  # Store the pid in case users fork this process.
        self._event_loop = asyncio.new_event_loop()
        self._thread: threading.Thread = threading.Thread(  # type: ignore[annotation-unchecked]
            target=self._event_loop.run_forever,
            daemon=True,  # type: ignore
        )
        self._thread.start()

    def __init__(self):
        """Instantiates an EventLoopHandler. Used to ensure consistency when calling async code from sync code in Mellea.

        Do not instantiate this class. Rely on the exported `_run_async_in_thread` function.
        """
        self._event_loop_setup()

    def _reinit_if_forked(self) -> None:
        """Reinitialize the event loop and thread if we're in a forked child to prevent hanging on awaited tasks."""
        if os.getpid() != self._pid:
            # If the process has been forked, reset the event loop and thread.
            # Don't cleanup the parent's objects.
            self._event_loop_setup()

    def __del__(self):
        """Delete the event loop handler.

        Uses a short join timeout: `__del__` can run during interpreter
        finalization, where blocking on a thread that may never be rescheduled
        would stall process exit. Once `sys.is_finalizing()` is true the waits
        are skipped altogether — a daemon thread is not guaranteed to be
        scheduled again at that point, so waiting on one can only burn the
        timeout without changing the outcome.
        """
        self._close_event_loop(join_timeout=0.0 if sys.is_finalizing() else 1.0)

    def _close_event_loop(self, join_timeout: float = 5.0) -> None:
        """Shut down the event loop and its thread, releasing the loop's file descriptors.

        Idempotent: the loop reference is cleared up front, so a second call (or a
        `__del__` following an explicit call) is a no-op.

        The ordering matters. `loop.stop()` is not thread-safe and on its own often
        fails to wake the selector, leaving `run_forever` blocked and the thread
        alive for the life of the process; `call_soon_threadsafe` does wake it.
        `loop.close()` can only run once `run_forever` has returned, hence the join
        in between — without it the loop's selector and self-pipe descriptors leak.

        Args:
            join_timeout: Seconds to wait for the loop thread to exit before giving
                up and leaving the loop unclosed. `0` blocks nowhere: the task
                cancellation round-trip is skipped and the thread is not joined, for
                callers that cannot afford to wait at all.
        """
        loop = self._event_loop
        if loop is None:
            return
        # Clear first so this is idempotent even if a step below raises.
        self._event_loop = None  # type: ignore[assignment]
        thread = getattr(self, "_thread", None)

        async def finalize_tasks() -> None:
            # Runs on `loop`, so all_tasks()/current_task() are safe to call here.
            # TODO: We can log errors here if needed.
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await loop.shutdown_asyncgens()

        if loop.is_running():
            # With no time to wait, don't schedule `finalize_tasks()` at all: the
            # coroutine object would be created and then never awaited, trading the
            # blocking wait for a "coroutine was never awaited" warning.
            if join_timeout > 0:
                try:
                    asyncio.run_coroutine_threadsafe(finalize_tasks(), loop).result(
                        join_timeout
                    )
                except Exception:
                    pass
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception:
                pass

        if thread is not None:
            try:
                if thread.is_alive():
                    thread.join(timeout=join_timeout)
            except Exception:
                pass

        # A still-running loop would raise; leave it rather than mask the join failure.
        try:
            if not loop.is_closed() and not loop.is_running():
                loop.close()
        except Exception:
            pass

    def _require_event_loop(self) -> asyncio.AbstractEventLoop:
        """Return this handler's event loop, or raise if it has been shut down.

        `_close_event_loop` clears the loop reference, so without this check the
        loop's absence surfaces as an `AttributeError` from deep inside `asyncio`
        (`submit`) or as a silent fallback onto a throwaway nested loop (`__call__`).

        Returns:
            The handler's event loop.

        Raises:
            RuntimeError: If the loop has already been shut down.
        """
        loop = self._event_loop
        if loop is None:
            raise RuntimeError(
                "This event loop handler has been shut down; its event loop is closed "
                "and cannot run further work."
            )
        return loop

    def submit(self, co: Coroutine[Any, Any, R]) -> concurrent.futures.Future[R]:
        """Schedule the coroutine on the event loop without waiting for its result.

        The non-blocking counterpart to `__call__`. Safe to call from the loop's own
        thread precisely because it never blocks on the returned future — doing so
        from inside the loop would deadlock.

        Contextvars are deliberately not propagated: the caller does not wait for the
        coroutine, so its snapshot of the calling context may be stale by the time the
        coroutine runs. Use `__call__` when the coroutine needs the caller's context.

        Args:
            co: coroutine to schedule.

        Returns:
            A future for the scheduled coroutine. Callers that never inspect it get
            fire-and-forget semantics; the coroutine's exceptions are stored on the
            future rather than raised here.

        Raises:
            RuntimeError: If the coroutine could not be scheduled, e.g. this handler
                has already been shut down by `_close_event_loop`. `co` is closed
                first, so it does not additionally emit a "coroutine was never
                awaited" warning.
        """
        try:
            self._reinit_if_forked()
            return asyncio.run_coroutine_threadsafe(co, self._require_event_loop())
        except Exception:
            # `run_coroutine_threadsafe` only wraps `co` in a task from inside the
            # callback it hands to the loop, so a raise here means `co` never started
            # and closing it is a clean no-op. `Exception`, not `BaseException`, for
            # the same reason as `__call__`: see its docstring.
            co.close()
            raise

    def __call__(self, co: Coroutine[Any, Any, R]) -> R:
        """Run the coroutine in the event loop, propagating the calling thread's contextvars.

        The caller's `contextvars` snapshot is applied inside the new Task so
        contextvar-backed state is visible to the coroutine. One-way:
        mutations inside the Task don't leak back.

        `_reinit_if_forked`/`get_current_event_loop` run before the coroutine
        is scheduled, so an exception there (or a failure to schedule) would
        otherwise leave `co` unawaited and emit a "coroutine was never
        awaited" warning on top of whatever raised. `co.close()` covers that:
        if scheduling never succeeded, `co` hasn't started and closing it is
        a clean no-op; if `.result()` raised the task's own exception instead,
        the future is only marked done after the scheduled task has fully
        finished running `co`, so closing an already-completed coroutine is
        also a no-op.

        Deliberately `except Exception`, not `BaseException`: a `Future.result()`
        with no timeout blocks on a `threading.Condition`, and a `KeyboardInterrupt`
        delivered to this (calling) thread can unblock that wait before the task
        on the event-loop thread has actually finished — `co` may still be running
        there. Closing it from here at that point would run `co`'s `GeneratorExit`
        handling in this thread while the event loop thread is concurrently
        stepping the same frame, which is not safe. Excluding
        `KeyboardInterrupt`/`SystemExit` means that rare case leaks the
        unawaited-coroutine warning instead of risking that race.

        `_wrapped()`'s own coroutine object needs the same treatment as `co`:
        if scheduling fails before `run_coroutine_threadsafe` hands it to the
        loop, `_wrapped()` never starts and never gets to `await co` — leaking
        both `_wrapped()`'s and (via the outer `except`) `co`'s "coroutine was
        never awaited" warnings otherwise.

        Raises:
            RuntimeError: If this handler has been shut down by `_close_event_loop`.
                The check is explicit because a cleared loop reference would
                otherwise compare equal to a sync caller's absent loop (`None ==
                None`) and quietly run `co` on a throwaway nested loop.
        """
        wrapped_co: Coroutine[Any, Any, R] | None = None
        try:
            self._reinit_if_forked()
            event_loop = self._require_event_loop()
            if event_loop == get_current_event_loop():
                # If this gets called from the same event loop, launch in a separate thread to prevent blocking.
                # The nested handler owns an event loop and a thread, so tear it down
                # explicitly rather than leaving it to __del__: a dropped handler's
                # thread stays parked in run_forever, leaking a thread, a loop, and
                # the loop's descriptors on every nested call.
                nested = _EventLoopHandler()
                try:
                    return nested(co)
                finally:
                    nested._close_event_loop()

            parent_ctx = contextvars.copy_context()

            async def _wrapped() -> R:
                for var, value in parent_ctx.items():
                    var.set(value)
                return await co

            wrapped_co = _wrapped()
            return asyncio.run_coroutine_threadsafe(wrapped_co, event_loop).result()
        except Exception:
            co.close()
            if wrapped_co is not None:
                wrapped_co.close()
            raise


# Instantiate this class once. It will not be re-instantiated.
__event_loop_handler = _EventLoopHandler()


def _run_async_in_thread(co: Coroutine[Any, Any, R]) -> R:
    """Call to run async code from synchronous code in Mellea.

    In Mellea, we utilize async code underneath sync code to speed up
    inference requests. This puts us in a difficult situation since most
    api providers and sdks use async clients that get bound to a specific event
    loop to make requests. These clients are typically long-lasting and sometimes
    cannot be easily reinstantiated on demand to avoid these issues.
    By declaring a single event loop for these async requests,
    Mellea avoids these client issues.

    Note: This implementation requires that sessions/backends be run only through
    the top-level / session sync or async interfaces, not both. You will need to
    reinstantiate your backend if switching between the two.

    Args:
        co: coroutine to run

    Returns:
        output of the coroutine
    """
    return __event_loop_handler(co)


def _schedule_async_in_thread(
    co: Coroutine[Any, Any, R],
) -> concurrent.futures.Future[R]:
    """Schedule async code on Mellea's event loop without waiting for it to finish.

    The non-blocking counterpart to `_run_async_in_thread`, for work whose result the
    caller does not need — releasing a client that has just been evicted from a cache,
    for instance. Blocking there would stall whatever loop the caller is running on,
    and the wait can be unbounded when Mellea's loop is busy with a generate call.

    Args:
        co: coroutine to schedule

    Returns:
        A future for the scheduled coroutine, which callers may ignore. Exceptions
        raised by `co` are stored on the future instead of surfacing anywhere.

    Raises:
        RuntimeError: If the coroutine could not be scheduled at all, e.g. Mellea's
            event loop has already been shut down.
    """
    return __event_loop_handler.submit(co)


__all__ = ["_run_async_in_thread", "_schedule_async_in_thread"]
