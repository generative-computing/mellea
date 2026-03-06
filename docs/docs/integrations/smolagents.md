---
title: "smolagents"
description: "Use HuggingFace smolagents tools inside a Mellea session."
# diataxis: how-to
---

# smolagents

`MelleaTool.from_smolagents()` wraps any [smolagents](https://huggingface.co/docs/smolagents)
`Tool` instance so it can be passed to any [`MelleaSession`](../guide/glossary#melleasession)
call. The HuggingFace ecosystem provides many pre-built tools — `PythonInterpreterTool`,
`DuckDuckGoSearchTool`, `WikipediaSearchTool`, and others.

**Prerequisites:** `pip install 'mellea[smolagents]'`

## Using smolagents tools

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

> **Backend note:** Tool calling requires a backend and model that support function
> calling (e.g., Ollama with `granite4:micro`, OpenAI with `gpt-4o`). The default
> Ollama setup supports this.

> **Full example:** [`docs/examples/tools/smolagents_example.py`](../../examples/tools/smolagents_example.py)

## Which approach to use

| Scenario | Use |
| -------- | --- |
| Your tool exists as a LangChain `BaseTool` | [`MelleaTool.from_langchain(tool)`](./langchain) |
| Your tool exists as a smolagents `Tool` | `MelleaTool.from_smolagents(tool)` |
| You have a plain Python function to expose | [`@tool` decorator](../guide/tools-and-agents) |
| You have LangChain message history to continue | [`convert_to_openai_messages` → `ChatContext`](./langchain.md#seeding-a-session-with-langchain-message-history) |
| You want Mellea as an OpenAI endpoint for another framework | [`m serve`](./m-serve) |

---


**See also:** [Tools and Agents](../guide/tools-and-agents) |
[Context and Sessions](../concepts/context-and-sessions)
