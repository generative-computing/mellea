# Testing plugins — how to unit-test hook functions without a live session.
#
# This example shows how to:
#   1. Construct a payload manually
#   2. Call a hook function directly (await my_hook(payload, ctx))
#   3. Assert the return value for both pass-through and blocking cases
#
# Run:
#   uv run python docs/examples/plugins/testing_plugins.py

import asyncio
import logging
from datetime import UTC, datetime, timezone
from unittest.mock import MagicMock

from mellea.plugins import HookType, PluginMode, block, hook

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("testing_plugins")


# ---------------------------------------------------------------------------
# The plugin under test
# ---------------------------------------------------------------------------

MAX_DESCRIPTION_LENGTH = 500


@hook(HookType.COMPONENT_PRE_EXECUTE, mode=PluginMode.SEQUENTIAL, priority=10)
async def enforce_description_length(payload, ctx):
    """Block components whose action text exceeds the maximum length."""
    action_text = str(payload.action)
    if len(action_text) > MAX_DESCRIPTION_LENGTH:
        return block(
            f"Action text is {len(action_text)} chars, max is {MAX_DESCRIPTION_LENGTH}",
            code="DESC_TOO_LONG",
            details={"length": len(action_text), "max": MAX_DESCRIPTION_LENGTH},
        )


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def make_payload(action_text: str):
    """Construct a minimal ComponentPreExecutePayload for testing.

    In a real test suite you would import the payload class directly:
        from mellea.plugins.hooks.component import ComponentPreExecutePayload
    Here we use a simple mock to keep the example dependency-free.
    """
    payload = MagicMock()
    payload.action = MagicMock()
    payload.action.__str__ = lambda self: action_text
    payload.action.description = action_text
    payload.component_type = "Instruction"
    payload.timestamp = datetime.now(UTC)
    payload.hook = HookType.COMPONENT_PRE_EXECUTE.value
    return payload


def make_ctx():
    """Construct a minimal PluginContext mock."""
    return MagicMock()


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


async def test_pass_through():
    """Short action text should pass through (return None)."""
    payload = make_payload("What is 2 + 2?")
    ctx = make_ctx()

    result = await enforce_description_length(payload, ctx)

    assert result is None, f"Expected None (pass-through), got {result}"
    log.info("[PASS] test_pass_through: short text passes through")


async def test_blocks_long_description():
    """Long action text should be blocked."""
    long_text = "x" * (MAX_DESCRIPTION_LENGTH + 100)
    payload = make_payload(long_text)
    ctx = make_ctx()

    result = await enforce_description_length(payload, ctx)

    assert result is not None, "Expected a blocking PluginResult, got None"
    assert result.continue_processing is False, "Expected continue_processing=False"
    assert result.violation is not None, "Expected a violation"
    assert result.violation.code == "DESC_TOO_LONG"
    log.info(
        "[PASS] test_blocks_long_description: long text is blocked with code=%s",
        result.violation.code,
    )


async def test_exact_boundary():
    """Text at exactly the max length should pass through."""
    boundary_text = "y" * MAX_DESCRIPTION_LENGTH
    payload = make_payload(boundary_text)
    ctx = make_ctx()

    result = await enforce_description_length(payload, ctx)

    assert result is None, f"Expected None at boundary, got {result}"
    log.info("[PASS] test_exact_boundary: boundary-length text passes through")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------


async def run_all():
    await test_pass_through()
    await test_blocks_long_description()
    await test_exact_boundary()
    log.info("")
    log.info("All tests passed!")


if __name__ == "__main__":
    log.info("--- Testing plugins example ---")
    log.info("")
    asyncio.run(run_all())
