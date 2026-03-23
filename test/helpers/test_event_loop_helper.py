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
