# pytest: unit
"""Implementing the Compactor protocol — anything with ``compact()`` works.

The protocol is structurally typed: a class with a ``compact(ctx, *,
backend=None) -> ChatContext`` method is a valid Compactor. No
inheritance is required.
"""

from mellea.stdlib.components.chat import Message
from mellea.stdlib.context import ChatContext, Compactor
from mellea.stdlib.context.chat import _rebuild_chat_context


class TruncateOldest:
    """Drop only the very first body component each call.

    Demonstrates the smallest possible Compactor implementation. Pattern
    1 (wired into ``ChatContext``) means each ``add()`` removes the
    oldest item then appends — net result: the context never grows.
    """

    def compact(self, ctx, *, backend=None):
        items = ctx.as_list()
        if len(items) <= 1:
            return ctx
        return _rebuild_chat_context(items[1:], compactor=ctx._compactor)


def pattern_1_wired_into_context():
    """Pattern 1: compactor lives on the context, runs in ``add()``."""
    ctx = ChatContext(compactor=TruncateOldest())
    for i in range(4):
        ctx = ctx.add(Message("user", f"msg {i}"))
    return [m.content for m in ctx.as_list()]
    # → ['msg 3']  (oldest dropped before each append)


def pattern_2_manual_call():
    """Pattern 2: caller invokes ``compact()`` directly between turns."""
    ctx = ChatContext(window_size=10_000)  # permissive — no auto-compaction
    for i in range(5):
        ctx = ctx.add(Message("user", f"msg {i}"))
    truncated = TruncateOldest().compact(ctx)
    return [m.content for m in truncated.as_list()]


def structural_typing_check():
    """The Compactor protocol is satisfied structurally, no inheritance."""
    c: Compactor = TruncateOldest()  # mypy-checked Protocol assignment
    return type(c).__name__


if __name__ == "__main__":
    for fn in [pattern_1_wired_into_context, pattern_2_manual_call]:
        print(f"--- {fn.__name__} ---")
        print(fn())
    print(f"structural typing: {structural_typing_check()} satisfies Compactor")


def test_custom_compactor_examples():
    assert pattern_1_wired_into_context() == ["msg 3"]
    assert pattern_2_manual_call() == ["msg 1", "msg 2", "msg 3", "msg 4"]
    assert structural_typing_check() == "TruncateOldest"
