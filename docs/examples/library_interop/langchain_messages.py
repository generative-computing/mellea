# pytest: ollama, e2e, qualitative
"""Continuing a LangChain conversation in a Mellea session.

Demonstrates converting messages from another library into a Mellea `ChatContext`:
LangChain messages are converted to OpenAI format with `convert_to_openai_messages`,
each one is added to a `ChatContext` as a Mellea `Message`, and the context seeds
`start_session(ctx=...)` so the model can answer a question about the earlier turns.

The pattern works with any library that can export OpenAI-format messages (for
example LlamaIndex, Haystack or Semantic Kernel), and the same conversion can be
reversed to send Mellea messages back out. If your messages carry images or
documents, extract those fields explicitly; the loop below copies only `role` and
`content`. For more on LangChain interop, see docs/docs/integrations/langchain.md.

Prerequisites:
    uv sync --extra tools
    ollama pull granite4.2:3b
    ollama serve

Usage:
    uv run python docs/examples/library_interop/langchain_messages.py
"""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Messages from a different library.
messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Hello!"),
    AIMessage(content="Hi there!"),
]

# Some libraries have conversion functions that make it easier to ingest into Mellea.
from langchain_core.messages import convert_to_openai_messages

openai_messages: list[dict[str, Any]] = convert_to_openai_messages(messages=messages)

# Import Mellea.
from mellea import start_session
from mellea.backends import ModelOption
from mellea.stdlib.components import Message
from mellea.stdlib.context import ChatContext

# Mellea uses explicit contexts. Cast the OpenAI formatted messages into
# Mellea messages and add them to the context.
ctx = ChatContext()
for msg in openai_messages:
    ctx = ctx.add(
        # NOTE: If your messages aren't in OpenAI format or have additional data like
        #       documents / images, you need to explicitly grab those fields as well.
        Message(role=msg["role"], content=msg["content"])
    )

# Utilize that new ChatContext to ask the assistant about its past messages.
m = start_session(ctx=ctx)
response = m.chat(
    "What exact words did the AI assistant use in its most recent response?",
    model_options={
        ModelOption.SEED: 123
    },  # Utilizing a seed for consistency in the example.
).content

assert "Hi there!" in response
