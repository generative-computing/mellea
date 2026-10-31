# pytest: unit
"""Implementing custom compactors with structural and marker typing.

The `Compactor` protocol is structurally typed: any class with a
`compact(ctx, *, backend=None) -> ChatContext` method satisfies it.

Two patterns for using custom compactors with `ChatContext`:

- **Pattern 1 (wired into `ChatContext.add`)**: Inherit from `InlineCompactor`
  for lightweight compactors that don't make backend calls. Backend-calling
  compactors should instead be wrapped in `ThresholdCompactor` before being attached.
- **Pattern 2 (manual call)**: Invoke `compact()` directly between turns. Any
  object satisfying the structural `Compactor` protocol works without inheritance.

Usage:
    uv run python docs/examples/context/custom_compactor.py
"""

from mellea.stdlib.components.chat import Message
from mellea.stdlib.context import ChatContext, Compactor
from mellea.stdlib.context.chat import _rebuild_chat_context
from mellea.stdlib.context.compactor import InlineCompactor


class TruncateOldest(InlineCompactor):
    """Drop only the very first body component each call.

    Smallest possible Pattern-1 compactor — inherits `InlineCompactor`
    so it can be wired directly into `ChatContext`. Each `add()`
    removes the oldest item then appends — net result: the context
    never grows.

    Note:
        `items[1:]` drops index 0 unconditionally, so a leading system
        message gets evicted on the first compaction. This is intentional
        for the smallest-possible-example shape; for the more common
        "keep the system prompt, drop oldest body" behaviour use
        :class:`mellea.stdlib.context.WindowCompactor` with the
        :func:`mellea.stdlib.context.pin_system` predicate instead.
    """

    def compact(self, ctx, *, backend=None):
        items = ctx.as_list()
        if len(items) <= 1:
            return ctx
        return _rebuild_chat_context(items[1:], source=ctx, compactor=ctx._compactor)


def pattern_1_wired_into_context():
    """Pattern 1: compactor lives on the context, runs in `add()`."""
    ctx = ChatContext(compactor=TruncateOldest())
    for i in range(4):
        ctx = ctx.add(Message("user", f"msg {i}"))
    return [m.content for m in ctx.as_list()]
    # → ['msg 3']  (oldest dropped before each append)


def pattern_2_manual_call():
    """Pattern 2: caller invokes `compact()` directly between turns."""
    ctx = ChatContext(window_size=10_000)  # permissive — no auto-compaction
    for i in range(5):
        ctx = ctx.add(Message("user", f"msg {i}"))
    truncated = TruncateOldest().compact(ctx)
    return [m.content for m in truncated.as_list()]


def structural_typing_check():
    """The bare Compactor protocol is satisfied structurally — no inheritance."""

    class JustCompact:  # no base class — pure structural Compactor
        def compact(self, ctx, *, backend=None):
            return ctx

    c: Compactor = JustCompact()  # mypy-checked Protocol assignment
    return type(c).__name__


if __name__ == "__main__":
    for fn in [pattern_1_wired_into_context, pattern_2_manual_call]:
        print(f"--- {fn.__name__} ---")
        print(fn())
    print(f"structural typing: {structural_typing_check()} satisfies Compactor")


def test_custom_compactor_examples():
    assert pattern_1_wired_into_context() == ["msg 3"]
    assert pattern_2_manual_call() == ["msg 1", "msg 2", "msg 3", "msg 4"]
    assert structural_typing_check() == "JustCompact"
