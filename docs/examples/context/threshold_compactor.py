# pytest: unit
"""ThresholdCompactor — gate an inner Compactor on conversation size.

Reads ``ModelOutputThunk.generation.usage`` from the most recent thunk
in the context. For a chat backend, ``total_tokens`` on that thunk is
``prompt_tokens`` (full conversation history sent to the model) plus
``completion_tokens`` (the reply), so it tracks *cumulative* context
size — not just one call's isolated tokens. The inner compactor fires
once that running size exceeds the configured threshold.
"""

from mellea.core.base import ModelOutputThunk
from mellea.stdlib.components.chat import Message
from mellea.stdlib.context import ChatContext, ThresholdCompactor, WindowCompactor


def _thunk(total_tokens: int) -> ModelOutputThunk:
    """Build a ModelOutputThunk with a populated usage dict (test helper)."""
    mot = ModelOutputThunk(value="")
    mot.generation.usage = {
        "prompt_tokens": total_tokens,
        "completion_tokens": 0,
        "total_tokens": total_tokens,
    }
    return mot


def below_threshold_passthrough():
    """Token usage is below threshold → inner compactor is NOT invoked."""
    gated = ThresholdCompactor(WindowCompactor(size=2), threshold=1000)
    ctx = ChatContext(window_size=10_000)
    for i in range(5):
        ctx = ctx.add(Message("user", f"msg {i}"))
    ctx = ctx.add(_thunk(50))  # only 50 tokens — below 1000
    out = gated.compact(ctx)
    return len(out.as_list())  # 6 (5 messages + thunk) — unchanged


def above_threshold_compacts():
    """Token usage exceeds threshold → inner compactor runs."""
    gated = ThresholdCompactor(WindowCompactor(size=2), threshold=1000)
    ctx = ChatContext(window_size=10_000)
    for i in range(5):
        ctx = ctx.add(Message("user", f"msg {i}"))
    ctx = ctx.add(_thunk(2000))  # 2000 tokens — over the gate
    out = gated.compact(ctx)
    return len(out.as_list())  # 2 — WindowCompactor(size=2) ran


if __name__ == "__main__":
    print(f"below_threshold_passthrough: {below_threshold_passthrough()}")
    print(f"above_threshold_compacts:    {above_threshold_compacts()}")


def test_threshold_compactor_examples():
    assert below_threshold_passthrough() == 6
    assert above_threshold_compacts() == 2
