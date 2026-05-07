"""Test ReACT framework handling of direct answers without tool calls."""

import pydantic
import pytest

from mellea.backends.tools import tool
from mellea.stdlib import functional as mfuncs
from mellea.stdlib.context import ChatContext
from mellea.stdlib.frameworks.react import react
from mellea.stdlib.session import start_session


class TrueOrFalse(pydantic.BaseModel):
    """Response indicating whether the ReACT agent has completed its task."""

    answer: bool = pydantic.Field(
        description="True if you have enough information to answer the user's question, False if you need more tool calls"
    )


async def last_loop_completion_check(
    goal, context, backend, step, model_options, turn_num, loop_budget
):
    """Completion check that asks the model if it has the answer on the last iteration.

    Note: step.value is guaranteed to exist when this is called.
    """
    # Only check on last iteration (and not for unlimited budget)
    if loop_budget == -1 or turn_num < loop_budget:
        return False

    content = mfuncs.chat(
        content=f"Do you know the answer to the user's original query ({goal})? If so, respond with True. If you need to take more actions, then respond False.",
        context=context,
        backend=backend,
        format=TrueOrFalse,
    )[0].content
    have_answer = TrueOrFalse.model_validate_json(content).answer
    return have_answer


@pytest.mark.ollama
@pytest.mark.e2e
@pytest.mark.qualitative
async def test_react_direct_answer_without_tools():
    """Test that ReACT handles direct answers when model doesn't call tools.

    This tests the case where the model provides a direct answer in step.value
    without making any tool calls. The fix ensures the loop terminates properly
    instead of continuing until loop_budget is exhausted.
    """
    m = start_session()

    # Ask a simple question that doesn't require tools
    # The model should provide a direct answer without calling any tools
    out, _ = await react(
        goal="What is 2 + 2?",
        context=ChatContext(),
        backend=m.backend,
        tools=[],  # No tools provided
        loop_budget=3,  # Should complete in 1 iteration, not exhaust budget
        answer_check=last_loop_completion_check,
    )

    # Verify we got an answer
    assert out.value is not None
    assert len(out.value) > 0

    # The answer should contain "4" or "four"
    answer_lower = out.value.lower()
    assert "4" in answer_lower or "four" in answer_lower


@pytest.mark.ollama
@pytest.mark.e2e
@pytest.mark.qualitative
async def test_react_direct_answer_with_unused_tools():
    """Test that ReACT handles direct answers even when tools are available.

    This tests the case where tools are provided but the model chooses to
    answer directly without using them.
    """
    m = start_session()

    # Create a dummy tool that won't be needed
    @tool
    def search_web(query: str) -> str:
        """Search the web for information."""
        return "Search results"

    # Ask a question that doesn't need the tool
    out, _ = await react(
        goal="What is the capital of France?",
        context=ChatContext(),
        backend=m.backend,
        tools=[search_web],
        loop_budget=3,
        answer_check=last_loop_completion_check,
    )

    # Verify we got an answer
    assert out.value is not None
    assert len(out.value) > 0

    # The answer should mention Paris
    answer_lower = out.value.lower()
    assert "paris" in answer_lower
