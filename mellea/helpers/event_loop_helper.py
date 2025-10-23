"""Helper for event loop management. Allows consistently running async generate requests in sync code."""

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

R = TypeVar("R")


class _EventLoopHandler:
    """A class that handles the event loop for Mellea code. Do not directly instantiate this. Use `_run_async_in_thread`."""

    def __init__(self):
        """Instantiates an EventLoopHandler. Used to ensure consistency when calling async code from sync code in Mellea.

        Do not instantiate this class. Rely on the exported `_run_async_in_thread` function.
        """
        self._event_loop = asyncio.new_event_loop()
        self._thread: threading.Thread = threading.Thread(
            target=self._event_loop.run_forever, daemon=True
        )
        self._thread.start()

    def __del__(self):
        """Delete the event loop handler."""
        self._close_event_loop()

    def _close_event_loop(self) -> None:
        """Called when deleting the event loop handler. Cleans up the event loop and thread."""
        if self._event_loop:
            try:
                tasks = asyncio.all_tasks(self._event_loop)
                for task in tasks:
                    task.cancel()

                async def finalize_tasks():
                    # TODO: We can log errors here if needed.
                    await asyncio.gather(*tasks, return_exceptions=True)

                out = asyncio.run_coroutine_threadsafe(
                    finalize_tasks(), self._event_loop
                )

                # Timeout if needed.
                out.result(5)
            except Exception:
                pass

            # Finally stop the event loop for this session.
            self._event_loop.stop()

    def __call__(self, co: Coroutine[Any, Any, R]) -> R:
        """Runs the coroutine in the event loop."""
        return asyncio.run_coroutine_threadsafe(co, self._event_loop).result()


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


__all__ = ["_run_async_in_thread"]
