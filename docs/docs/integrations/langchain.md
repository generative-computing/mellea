---
title: "LangChain"
description: "Use LangChain tools inside Mellea and seed a Mellea session with LangChain message history."
# diataxis: how-to
---

# LangChain

Mellea integrates with LangChain in two ways:

1. **Tool bridging** — wrap existing LangChain tools as [`MelleaTool`](../guide/glossary#tool)
   objects and pass them to any [`MelleaSession`](../guide/glossary#melleasession) call.
2. **Message history** — seed a Mellea [`ChatContext`](../guide/glossary#context) with
   conversation history from a LangChain session.

## Using LangChain tools

**Prerequisites:** `pip install langchain-core` (or `pip install langchain-community`
for community tools).

`MelleaTool.from_langchain()` wraps any LangChain `BaseTool` so it can be passed to
`instruct()` or `chat()` via [`ModelOption.TOOLS`](../guide/glossary#modeloption):

```python
from mellea import start_session
from mellea.backends import ModelOption
from mellea.backends.tools import MelleaTool

# Import any LangChain BaseTool subclass
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Wrap for use in Mellea
wiki = MelleaTool.from_langchain(WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()))

m = start_session()
result = m.instruct(
    "What year was the Eiffel Tower completed? Use the Wikipedia tool.",
    model_options={ModelOption.TOOLS: [wiki]},
    tool_calls=True,
)

print(result)

# The model chose to call a tool — execute it
if result.tool_calls:
    tool_output = result.tool_calls[wiki.name].call_func()
    print(tool_output)
```

`from_langchain()` reads the tool's name and schema directly from the `BaseTool`
instance, so any tool that follows the LangChain `BaseTool` interface works without
further configuration.

> **Backend note:** Tool calling requires a backend and model that support function
> calling (e.g., Ollama with `granite4:micro`, OpenAI with `gpt-4o`). The default
> Ollama setup supports this.

## Seeding a session with LangChain message history

When migrating from LangChain or building a system that spans both libraries, you may
want to start a Mellea session from an existing LangChain conversation. Mellea uses
explicit `ChatContext` objects; the bridge is to convert LangChain messages to OpenAI
format first, then build the context:

```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import convert_to_openai_messages

from mellea import start_session
from mellea.stdlib.components import Message
from mellea.stdlib.context import ChatContext

# Existing LangChain conversation history
lc_messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Hello!"),
    AIMessage(content="Hi there!"),
]

# 1. Convert to OpenAI format (a common interchange)
openai_messages = convert_to_openai_messages(messages=lc_messages)

# 2. Build a Mellea ChatContext from the converted messages
ctx = ChatContext()
for msg in openai_messages:
    # NOTE: if messages contain images or documents, extract those fields too
    ctx = ctx.add(Message(role=msg["role"], content=msg["content"]))

# 3. Continue the conversation in Mellea
m = start_session(ctx=ctx)
response = m.chat("What exact words did the AI assistant use in its most recent response?")
print(str(response))
# Output will vary — LLM responses depend on model and temperature.
# Expected: the model reports back "Hi there!" from the seeded context
```

`convert_to_openai_messages` normalises all LangChain message subtypes (system, human,
AI, tool) into `{"role": ..., "content": ...}` dicts. Any library that exports to
OpenAI chat format — LlamaIndex, Haystack, Semantic Kernel — works with the same pattern.

> **Full example:** [`docs/examples/library_interop/langchain_messages.py`](../../examples/library_interop/langchain_messages.py)

## Which approach to use

| Scenario | Use |
| -------- | --- |
| Your tool exists as a LangChain `BaseTool` | `MelleaTool.from_langchain(tool)` |
| Your tool exists as a smolagents `Tool` | [`MelleaTool.from_smolagents(tool)`](./smolagents) |
| You have a plain Python function to expose | [`@tool` decorator](../guide/tools-and-agents) |
| You have LangChain message history to continue | `convert_to_openai_messages` → `ChatContext` |
| You want Mellea as an OpenAI endpoint for another framework | [`m serve`](./m-serve) |

---


**See also:** [Tools and Agents](../guide/tools-and-agents) |
[Context and Sessions](../concepts/context-and-sessions)
