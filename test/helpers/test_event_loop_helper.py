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


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
