"""Live multi-turn reasoning-replay verification against a real thinking model.

This is the "quick verification after the fact" for issue #1201: the mechanical
strip/round-trip plumbing is unit-tested in ``test_reasoning_replay.py`` without
a model; this module confirms the same behaviour holds against a real reasoning
endpoint, and pins the one empirical unknown — which side of the tool-call
contract the served model lands on.

Model-agnostic by design. Configure via environment variables (skipped entirely
when unset, so it never runs in default CI):

- ``MELLEA_REASONING_TEST_BASE_URL`` — OpenAI-compatible endpoint, e.g.
  ``http://localhost:8000/v1`` (vLLM) or ``http://localhost:11434/v1`` (Ollama).
- ``MELLEA_REASONING_TEST_MODEL`` — served model id. Defaults to a Granite
  reasoning model; point it at Qwen3 / DeepSeek-R1 / etc. to verify those.
- ``MELLEA_REASONING_TEST_API_KEY`` — optional; defaults to ``"ollama"``.

Run directly (bypasses the default marker filter):

    MELLEA_REASONING_TEST_BASE_URL=http://localhost:8000/v1 \\
    MELLEA_REASONING_TEST_MODEL=Qwen/Qwen3-8B \\
    uv run pytest test/backends/test_reasoning_replay_e2e.py -v
"""

import os

import pytest

from mellea.backends import ModelOption
from mellea.backends.openai import OpenAIBackend
from mellea.stdlib.components import Message
from mellea.stdlib.components.chat import as_chat_history
from mellea.stdlib.context import ChatContext, Context

pytestmark = [
    pytest.mark.openai,
    pytest.mark.e2e,
    pytest.mark.qualitative,
    pytest.mark.skipif(
        not os.environ.get("MELLEA_REASONING_TEST_BASE_URL"),
        reason="Set MELLEA_REASONING_TEST_BASE_URL to run the live reasoning-replay check",
    ),
]

_DEFAULT_MODEL = "granite4:micro-h"


@pytest.fixture(scope="module")
def backend() -> OpenAIBackend:
    return OpenAIBackend(
        model_id=os.environ.get("MELLEA_REASONING_TEST_MODEL", _DEFAULT_MODEL),
        base_url=os.environ["MELLEA_REASONING_TEST_BASE_URL"],
        api_key=os.environ.get("MELLEA_REASONING_TEST_API_KEY", "ollama"),
    )


async def test_reasoning_captured_and_carried_across_turns(backend: OpenAIBackend):
    """End-to-end: reasoning is captured on turn 1 and reaches Message.thinking.

    Verifies the capture -> parse -> as_chat_history plumbing against a live
    reasoning model. The turn-2 replay contract (strip vs. round-trip) is
    asserted structurally in the unit test; here we confirm the trace survives
    into the parsed assistant Message so the policy has something to act on.
    """
    ctx: Context = ChatContext()
    mot, ctx = await backend.generate_from_chat_context(
        Message("user", "What is 17 * 23? Think step by step."),
        ctx,
        model_options={ModelOption.THINKING: True},
    )
    await mot.avalue()

    if not mot.thinking:
        pytest.skip(
            "Model produced no reasoning trace — ensure the configured model is a "
            "reasoning model and THINKING is honoured by the endpoint."
        )

    # The captured reasoning must survive parsing into the assistant Message and
    # the chat-history round-trip.
    history = as_chat_history(ctx)
    assistant_turns = [m for m in history if m.role == "assistant"]
    assert assistant_turns, "expected an assistant turn in the history"
    assert assistant_turns[-1].thinking, (
        "reasoning captured on the MOT must carry onto the parsed assistant Message"
    )

    # A plain follow-up turn should complete without error (consensus rule strips
    # the prior reasoning; a model that hard-400s here is the DeepSeek-style
    # exception worth flagging on the issue).
    mot2, ctx = await backend.generate_from_chat_context(
        Message("user", "Now double that result."),
        ctx,
        model_options={ModelOption.THINKING: True},
    )
    value2 = await mot2.avalue()
    assert value2 is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
