# pytest: unit
"""WindowCompactor — keep the last N body components.

Demonstrates the default behaviour, the ``window_size=`` sugar on
``ChatContext``, and how the auto-pinned system prefix is preserved.
"""

from mellea.stdlib.components.chat import Message
from mellea.stdlib.context import (
    ChatContext,
    WindowCompactor,
    pin_nothing,
    pin_system_and_initial_user,
)


def basic_window():
    """``ChatContext()`` keeps the full history by default; opt in via
    ``compactor=`` to start truncating.
    """
    ctx = ChatContext(compactor=WindowCompactor(size=5))
    for i in range(8):
        ctx = ctx.add(Message("user", f"msg {i}"))
    return [m.content for m in ctx.as_list()]
    # → ['msg 3', 'msg 4', 'msg 5', 'msg 6', 'msg 7']


def window_size_sugar():
    """``window_size=`` is sugar for ``WindowCompactor(size=...)``."""
    ctx = ChatContext(window_size=3)
    for i in range(6):
        ctx = ctx.add(Message("user", f"msg {i}"))
    return [m.content for m in ctx.as_list()]
    # → ['msg 3', 'msg 4', 'msg 5']


def system_prefix_pinned():
    """Default predicate ``pin_system`` keeps a leading system message."""
    ctx = ChatContext(window_size=3)
    ctx = ctx.add(Message("system", "You are a helpful assistant."))
    for i in range(6):
        ctx = ctx.add(Message("user", f"msg {i}"))
    return [(m.role, m.content) for m in ctx.as_list()]
    # → [('system', '...'), ('user', 'msg 3'), ('user', 'msg 4'), ('user', 'msg 5')]


def pin_initial_user_too():
    """Use ``pin_system_and_initial_user`` to also keep the user's first turn."""
    ctx = ChatContext(
        compactor=WindowCompactor(size=3, pin_predicate=pin_system_and_initial_user)
    )
    ctx = ctx.add(Message("system", "You are helpful."))
    ctx = ctx.add(Message("user", "What is the capital of France?"))
    for i in range(6):
        ctx = ctx.add(Message("assistant", f"reply {i}"))
    return [(m.role, m.content) for m in ctx.as_list()]


def pure_last_n():
    """``pin_nothing`` disables prefix pinning — the system message is dropped."""
    ctx = ChatContext(compactor=WindowCompactor(size=3, pin_predicate=pin_nothing))
    ctx = ctx.add(Message("system", "ignored after a few turns"))
    for i in range(6):
        ctx = ctx.add(Message("user", f"msg {i}"))
    return [(m.role, m.content) for m in ctx.as_list()]


def clear_body_keep_prefix():
    """``size=0`` drops the body entirely while keeping the pinned prefix."""
    ctx = ChatContext(window_size=10_000)
    ctx = ctx.add(Message("system", "You are helpful."))
    for i in range(5):
        ctx = ctx.add(Message("user", f"msg {i}"))
    cleared = WindowCompactor(size=0).compact(ctx)
    return [(m.role, m.content) for m in cleared.as_list()]
    # → [('system', 'You are helpful.')]


if __name__ == "__main__":
    for fn in [
        basic_window,
        window_size_sugar,
        system_prefix_pinned,
        pin_initial_user_too,
        pure_last_n,
        clear_body_keep_prefix,
    ]:
        print(f"--- {fn.__name__} ---")
        print(fn())


def test_window_compactor_examples():
    """Smoke test all examples — invariants documented in each docstring."""
    assert basic_window() == ["msg 3", "msg 4", "msg 5", "msg 6", "msg 7"]
    assert window_size_sugar() == ["msg 3", "msg 4", "msg 5"]
    assert system_prefix_pinned()[0] == ("system", "You are a helpful assistant.")
    pinned = pin_initial_user_too()
    assert pinned[0] == ("system", "You are helpful.")
    assert pinned[1] == ("user", "What is the capital of France?")
    assert all(role == "user" for role, _ in pure_last_n())
    assert clear_body_keep_prefix() == [("system", "You are helpful.")]
