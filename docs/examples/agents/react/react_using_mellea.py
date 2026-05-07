# pytest: ollama, e2e, qualitative, slow

"""React examples using the Mellea library's framework."""

import asyncio

import pydantic
from langchain_community.tools import DuckDuckGoSearchResults

from mellea.backends.tools import MelleaTool
from mellea.stdlib import functional as mfuncs
from mellea.stdlib.context import ChatContext
from mellea.stdlib.frameworks.react import react
from mellea.stdlib.session import start_session

m = start_session()

# Simple tool for searching. Requires the langchain-community package.
# Mellea allows you to interop with langchain defined tools.
lc_ddg_search = DuckDuckGoSearchResults(output_format="list")
search_tool: MelleaTool = MelleaTool.from_langchain(lc_ddg_search)


class Email(pydantic.BaseModel):
    """An email."""

    to: str
    subject: str
    body: str


class TrueOrFalse(pydantic.BaseModel):
    """Response indicating whether the ReACT agent has completed its task."""

    answer: bool = pydantic.Field(
        description="True if you have enough information to answer the user's question, False if you need more tool calls"
    )


async def last_loop_completion_check(
    goal, context, backend, step, model_options, turn_num, loop_budget
):
    """Completion check that asks the model if it has the answer on the last iteration.

    Only checks on the last iteration (when turn_num == loop_budget) to avoid
    unnecessary LLM calls. Returns False for all other iterations.

    Note: step.value is guaranteed to exist when this is called.
    """
    # Only check on last iteration (and not for unlimited budget)
    if loop_budget == -1 or turn_num < loop_budget:
        return False

    message, _ = await mfuncs.achat(
        content=f"Do you know the answer to the user's original query ({goal})? If so, respond with True. If you need to take more actions, then respond False.",
        context=context,
        backend=backend,
        format=TrueOrFalse,
    )
    have_answer = TrueOrFalse.model_validate_json(message.content).answer

    return have_answer


async def custom_completion_check(
    goal, context, backend, step, model_options, turn_num, loop_budget
):
    """Custom completion check combining keyword detection and fallback to last-loop check.

    This runs every iteration:
    1. First checks if response contains "final answer" for early termination
    2. On the last iteration, falls back to asking the model if it has the answer

    Note: step.value is guaranteed to exist when this is called.
    """
    # Check every iteration for "final answer" keyword (early termination)
    if "final answer" in step.value.lower():
        return True

    # On last iteration, fall back to asking the model if it has the answer
    if loop_budget != -1 and turn_num >= loop_budget:
        return await last_loop_completion_check(
            goal, context, backend, step, model_options, turn_num, loop_budget
        )

    return False


async def main():
    """Example."""
    # Version with custom answer check that terminates early
    # when the model says "final answer" and queries the LLM
    # if it reaches the loop_budget.
    out, _ = await react(
        goal="What is the Mellea python library?",
        context=ChatContext(),
        backend=m.backend,
        tools=[search_tool],
        loop_budget=12,
        answer_check=custom_completion_check,
    )
    print(out)

    # Version that looks up info and formats the final response as an Email object.
    # out, _ = await react(
    #     goal="Write an email about the Mellea python library to Jake with the subject 'cool library'.",
    #     context=ChatContext(),
    #     backend=m.backend,
    #     tools=[search_tool],
    #     format=Email,
    #     answer_check=custom_completion_check,
    #     loop_budget=20,
    # )
    # print(out)


asyncio.run(main())
