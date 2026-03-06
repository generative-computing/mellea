---
title: "LangChain and smolagents"
description: "Use LangChain and smolagents tools inside Mellea, and bring LangChain message history into a Mellea session."
# diataxis: how-to
---

# LangChain and smolagents

Mellea integrates with the broader Python LLM ecosystem in two ways:

1. **Tool bridging** — wrap existing LangChain or smolagents tools as [`MelleaTool`](../guide/glossary#tool) objects and pass them to any [`MelleaSession`](../guide/glossary#melleasession) call.
2. **Message history** — seed a Mellea [`ChatContext`](../guide/glossary#context) with conversation history from another library.

---

## Using LangChain tools

**Prerequisites:** `pip install langchain-core` (or `pip install langchain-community` for community tools).

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

`from_langchain()` reads the tool's name and schema directly from the `BaseTool` instance,
so any tool that follows the LangChain `BaseTool` interface works without further
configuration.

> **Backend note:** Tool calling requires a backend and model that support function
> calling (e.g., Ollama with `granite4:micro`, OpenAI with `gpt-4o`). The default
> Ollama setup supports this.

---

## Using smolagents tools

**Prerequisites:** `pip install 'mellea[smolagents]'` (installs smolagents as a dependency).

`MelleaTool.from_smolagents()` wraps any smolagents `Tool` instance. The HuggingFace
ecosystem provides many pre-built tools — `PythonInterpreterTool`, `DuckDuckGoSearchTool`,
`WikipediaSearchTool`, and others:

```python
from mellea import start_session
from mellea.backends import ModelOption
from mellea.backends.tools import MelleaTool

from smolagents import PythonInterpreterTool

# Wrap the smolagents tool
python_tool = MelleaTool.from_smolagents(PythonInterpreterTool())

m = start_session()
result = m.instruct(
    "Calculate the sum of numbers from 1 to 10 using Python",
    model_options={ModelOption.TOOLS: [python_tool]},
    tool_calls=True,
)

print(result)

if result.tool_calls:
    try:
        calc_result = result.tool_calls[python_tool.name].call_func()
        print(f"Calculation result: {calc_result}")
    except Exception as e:
        print(f"Tool execution failed: {e}")
```

`from_smolagents()` uses smolagents' own JSON schema conversion, so the tool's
description and parameter types are preserved exactly.

> **Full example:** [`docs/examples/tools/smolagents_example.py`](../../examples/tools/smolagents_example.py)

---

## Seeding a session with LangChain message history

When migrating from LangChain or building a system that spans both libraries, you may
want to start a Mellea session from an existing LangChain conversation. Mellea uses
explicit [`ChatContext`](../guide/glossary#context) objects; the bridge is to convert
LangChain messages to OpenAI format first, then build the context:

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

`convert_to_openai_messages` is provided by LangChain and normalises all message
subtypes (system, human, AI, tool) into `{"role": ..., "content": ...}` dicts. Any
library that can export to OpenAI chat format — LlamaIndex, Haystack, Semantic Kernel —
works with the same pattern.

> **Full example:** [`docs/examples/library_interop/langchain_messages.py`](../../examples/library_interop/langchain_messages.py)

---

## Which approach to use

| Scenario | Use |
| -------- | --- |
| Your tool exists as a LangChain `BaseTool` | `MelleaTool.from_langchain(tool)` |
| Your tool exists as a smolagents `Tool` | `MelleaTool.from_smolagents(tool)` |
| You have a plain Python function to expose | [`@tool` decorator](../guide/tools-and-agents.md) |
| You have LangChain message history to continue | `convert_to_openai_messages` → `ChatContext` |
| You want Mellea as an OpenAI endpoint for another framework | [`m serve`](./m-serve.md) |

---

**Previous:** [m serve](./m-serve.md) |
**Next:** [Metrics and Telemetry](../evaluation-and-observability/metrics-and-telemetry.md)

**See also:** [Tools and Agents](../guide/tools-and-agents.md) |
[Context and Sessions](../concepts/context-and-sessions.md)
