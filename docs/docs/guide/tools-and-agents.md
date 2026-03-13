---
title: "Tools and Agents"
description: "Give LLMs access to tools, build ReACT agents, and validate tool call arguments."
# diataxis: how-to
---

**Prerequisites:** [Quick Start](../getting-started/quickstart) complete, `pip install mellea`,
Ollama running locally. LangChain interop requires `pip install langchain-community`.

> **Note:** An _agent_ is a generative program in which an LLM determines the control
> flow of the program. The patterns in this page range from simple one-shot tool use
> to goal-driven agentic loops.

## Defining tools with `@tool`

The `@tool` decorator turns a regular Python function into a tool the LLM can call.
Mellea uses the function's docstring and type hints to build the tool schema:

```python
from mellea.backends import tool

@tool
def get_weather(location: str, days: int = 1) -> dict:
    """Get weather forecast for a location.

    Args:
        location: City name.
        days: Number of days to forecast.
    """
    return {"location": location, "days": days, "forecast": "sunny", "temperature": 72}
```

Use `@tool(name="...")` to override the tool name as it appears to the model:

```python
from mellea.backends import tool

@tool(name="calculator")
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate.
    """
    return str(eval(expression))  # noqa: S307 — use only with trusted input
```

Decorated tools expose a `.run()` method for direct invocation without going through
the LLM:

```python
weather = get_weather.run("Boston", days=3)
```

You can also construct a tool from any callable manually:

```python
from mellea.backends.tools import MelleaTool

def double(x: int) -> int:
    """Double the input. Args: x: Input value."""
    return x * 2

my_tool = MelleaTool.from_callable(double)
```

## Passing tools to `instruct()`

Pass tools via `ModelOption.TOOLS`. The model can then choose to call them:

```python
from mellea import start_session
from mellea.backends import ModelOption, tool

@tool
def get_weather(location: str, days: int = 1) -> dict:
    """Get weather forecast for a location.

    Args:
        location: City name.
        days: Number of days to forecast.
    """
    return {"location": location, "days": days, "forecast": "sunny", "temperature": 72}

m = start_session()
response = m.instruct(
    "What is the weather like in San Francisco?",
    model_options={ModelOption.TOOLS: [get_weather]},
)
print(str(response))
# Output will vary — LLM responses depend on model and temperature.
```

### Requiring a tool call

Use the `uses_tool` requirement to enforce that the model actually calls a specific
tool:

```python
from mellea import start_session
from mellea.backends import ModelOption
from mellea.backends.tools import MelleaTool
from mellea.stdlib.requirements import uses_tool
from mellea.stdlib.tools import local_code_interpreter

m = start_session()
response = m.instruct(
    "Use the code interpreter tool to compute 7 factorial.",
    requirements=[uses_tool(local_code_interpreter)],
    model_options={ModelOption.TOOLS: [MelleaTool.from_callable(local_code_interpreter)]},
    tool_calls=True,
)
```

With `tool_calls=True`, the result exposes a `.tool_calls` dict you can inspect and
execute:

```python
code = response.tool_calls["local_code_interpreter"].args["code"]
exec_result = response.tool_calls["local_code_interpreter"].call_func()
print(exec_result)
```

### Validating tool arguments

`tool_arg_validator` adds fine-grained validation over the arguments the model
generates for a tool call:

```python
from mellea import start_session
from mellea.backends import ModelOption
from mellea.backends.tools import MelleaTool
from mellea.stdlib.requirements import tool_arg_validator, uses_tool
from mellea.stdlib.tools import local_code_interpreter

m = start_session()
response = m.instruct(
    "Use the code interpreter to plot y=x². Save the plot to /tmp/output.png.",
    requirements=[
        uses_tool(local_code_interpreter),
        tool_arg_validator(
            "The plot must be saved to /tmp/output.png and must not call plt.show()",
            tool_name=local_code_interpreter,
            arg_name="code",
            validation_fn=lambda code: (
                "/tmp/output.png" in code and "plt.show()" not in code
            ),
        ),
    ],
    model_options={ModelOption.TOOLS: [MelleaTool.from_callable(local_code_interpreter)]},
    tool_calls=True,
)
```

## LangChain and smolagents interop

Import tools directly from LangChain or smolagents:

```python
from langchain_community.tools import DuckDuckGoSearchResults
from mellea.backends.tools import MelleaTool

search_tool = MelleaTool.from_langchain(DuckDuckGoSearchResults(output_format="list"))
```

`MelleaTool.from_smolagents()` works the same way for smolagents tools.

## ReACT agent

`react()` is a built-in goal-driven agentic loop. It iteratively selects and calls
tools until the goal is met or a step budget is reached:

```python
import asyncio
from mellea import start_session
from mellea.backends.tools import MelleaTool
from mellea.stdlib.context import ChatContext
from mellea.stdlib.frameworks.react import react
from langchain_community.tools import DuckDuckGoSearchResults

m = start_session()
search_tool = MelleaTool.from_langchain(DuckDuckGoSearchResults(output_format="list"))

async def main():
    result, _ = await react(
        goal="What is the Mellea Python library?",
        context=ChatContext(),
        backend=m.backend,
        tools=[search_tool],
    )
    print(result)

asyncio.run(main())
# Output will vary — LLM responses depend on model and temperature.
```

`react()` can return a structured Pydantic object by passing a `format` parameter:

```python
import asyncio
import pydantic
from mellea import start_session
from mellea.backends.tools import MelleaTool
from mellea.stdlib.context import ChatContext
from mellea.stdlib.frameworks.react import react
from langchain_community.tools import DuckDuckGoSearchResults

class Email(pydantic.BaseModel):
    to: str
    subject: str
    body: str

m = start_session()
search_tool = MelleaTool.from_langchain(DuckDuckGoSearchResults(output_format="list"))

async def main():
    result, _ = await react(
        goal="Write an email about Mellea to Jake with subject 'cool library'.",
        context=ChatContext(),
        backend=m.backend,
        tools=[search_tool],
        format=Email,
    )
    print(result.parsed_repr.body)

asyncio.run(main())
# Output will vary — LLM responses depend on model and temperature.
```

> **Advanced:** The core idea of ReACT is to alternate between reasoning ("Thought")
> and acting ("Action") in a loop: generate a thought, choose an action, supply
> arguments, observe the tool output, then check whether the goal is achieved.
> Mellea's `react()` implements this loop using `chat()` with structured output at
> each step, backed by `@generative` for constrained argument selection. You can
> build a custom ReACT-style loop by hand using the same primitives — see
> `mellea.stdlib.components.react` for reference.

## Code interpreter

Mellea includes a built-in Python code interpreter tool:

```python
from mellea.stdlib.tools import code_interpreter

result = code_interpreter("print(1 + 1)")
print(result)  # "2"
```

Pass `local_code_interpreter` as a tool to `instruct()` to let the LLM write and
execute code. Combine with `uses_tool` and `tool_arg_validator` to constrain what
gets generated (see examples above).

> **Warning:** `local_code_interpreter` executes Python code in the current process.
> Do not use it in production contexts without sandboxing.

---

**See also:** [Tutorial 04: Making Agents Reliable](../tutorials/04-making-agents-reliable) | [Instruct, Validate, Repair](../concepts/instruct-validate-repair)
