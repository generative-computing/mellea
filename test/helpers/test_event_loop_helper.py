import contextvars
import multiprocessing

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
