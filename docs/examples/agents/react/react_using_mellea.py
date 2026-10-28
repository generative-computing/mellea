# pytest: ollama, e2e, qualitative, slow
"""React examples using the Mellea library's framework.

Demonstrates the ReACT (Reasoning and Acting) loop with tool integration,
using Mellea's stdlib react() function and LangChain tool interoperability.

Prerequisites:
    uv pip install langchain-community ddgs
    ollama serve
    ollama pull granite4.2:3b

Usage:
    uv run python docs/examples/agents/react/react_using_mellea.py
"""

import asyncio

import pydantic
from langchain_community.tools import DuckDuckGoSearchResults

from mellea.backends.tools import MelleaTool
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


async def main():
    """Example."""
    # Simple version that just searches for an answer.
    out, _ = await react(
        goal="What is the Mellea python library?",
        context=ChatContext(),
        backend=m.backend,
        tools=[search_tool],
        loop_budget=12,
    )
    print(out)

    # Version that looks up info and formats the final response as an Email object.
    # out, _ = await react(
    #     goal="Write an email about the Mellea python library to Jake with the subject 'cool library'.",
    #     context=ChatContext(),
    #     backend=m.backend,
    #     tools=[search_tool],
    #     format=Email,
    #     loop_budget=20,
    # )
    # print(out)


asyncio.run(main())
