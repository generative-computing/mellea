"""React examples using the Mellea library's framework."""

import asyncio

import pydantic
from langchain_community.tools import DuckDuckGoSearchResults

from mellea.backends.tools import MelleaTool
from mellea.stdlib.context import ChatContext
from mellea.stdlib.frameworks.react import react
from mellea.stdlib.session import start_session

m = start_session()

# Simple tool for searching. Requires the langchain_communities package.
# Mellea allows you to interop with langchain defined tools.
lc_ddg_search = DuckDuckGoSearchResults(output_format="list")
search_tool = MelleaTool.from_langchain(lc_ddg_search)


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
    )
    print(out)

    # Version that looks up info and formats the final response as an Email object.
    # out, _ = await react(
    #     goal="Write an email about the Mellea python library to Jake with the subject 'cool library'.",
    #     context=ChatContext(),
    #     backend=m.backend,
    #     tools=[search_tool],
    #     format=Email
    # )
    # print(out)


asyncio.run(main())
