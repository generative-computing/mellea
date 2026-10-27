---
title: "Tools and Agents"
description: "Give LLMs access to tools, build ReACT agents, and validate tool call arguments."
# diataxis: how-to
---

**Prerequisites:** [Quick Start](../getting-started/quickstart.md) complete, `pip install mellea`,
Ollama running locally. LangChain interop requires `pip install langchain-community`.

> **Note:** An _agent_ is a generative program in which an LLM determines the control
> flow of the program. The patterns in this page range from simple one-shot tool use
> to goal-driven agentic loops.

## Defining tools with `@tool`

The `@tool` decorator turns a regular Python function into a tool the LLM can call.
Mellea uses the function's docstring and type hints to build the tool schema:

```python
# Requires: mellea
# Returns: dict
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
# Requires: mellea
# Returns: str
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
# Requires: mellea
# Returns: MelleaTool
from mellea.backends.tools import MelleaTool

def double(x: int) -> int:
    """Double the input. Args: x: Input value."""
    return x * 2

my_tool = MelleaTool.from_callable(double)
```

## Passing tools to `instruct()`

Pass tools via `ModelOption.TOOLS`. The model can then choose to call them:

```python
# Requires: mellea
# Returns: str
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
# Requires: mellea
# Returns: ModelOutputThunk
from mellea import start_session
from mellea.backends import ModelOption
from mellea.stdlib.requirements import uses_tool
from mellea.stdlib.tools import python_tool

tool = python_tool(tier="local_unsafe", name="python")
m = start_session()
response = m.instruct(
    "Use the code interpreter tool to compute 7 factorial.",
    requirements=[uses_tool("python")],
    model_options={ModelOption.TOOLS: [tool]},
    tool_calls=True,
)

# tool_calls=True makes .tool_calls available on the result
tool_call = next((tc for tc in response.tool_calls if tc.name == "python"), None)
if tool_call:
    code = tool_call.args["code"]
    exec_result = tool_call.call_func()
    print(exec_result)
```

### Validating tool arguments

`tool_arg_validator` adds fine-grained validation over the arguments the model
generates for a tool call:

```python
# Requires: mellea
# Returns: ModelOutputThunk
from mellea import start_session
from mellea.backends import ModelOption
from mellea.stdlib.requirements import tool_arg_validator, uses_tool
from mellea.stdlib.tools import python_tool

tool = python_tool(tier="local_unsafe", name="python")
m = start_session()
response = m.instruct(
    "Use the code interpreter to plot y=x². Save the plot to /tmp/output.png.",
    requirements=[
        uses_tool("python"),
        tool_arg_validator(
            "The plot must be saved to /tmp/output.png and must not call plt.show()",
            tool_name="python",
            arg_name="code",
            validation_fn=lambda code: (
                "/tmp/output.png" in code and "plt.show()" not in code
            ),
        ),
    ],
    model_options={ModelOption.TOOLS: [tool]},
    tool_calls=True,
)
```

## LangChain and smolagents interop

Import tools directly from LangChain or smolagents. Install the required
packages first: `uv pip install langchain-community ddgs`.

```python
# Requires: langchain-community
# Returns: MelleaTool
from langchain_community.tools import DuckDuckGoSearchResults
from mellea.backends.tools import MelleaTool

search_tool = MelleaTool.from_langchain(DuckDuckGoSearchResults(output_format="list"))
```

`MelleaTool.from_smolagents()` works the same way for smolagents tools.

## ReACT agent

`react()` is a built-in goal-driven agentic loop. It iteratively selects and calls
tools until the goal is met or a step budget is reached:

```python
# Requires: mellea, langchain-community
# Returns: str
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
# Requires: mellea, langchain-community, pydantic
# Returns: Email
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
# Requires: mellea
# Returns: ExecutionResult
from mellea.stdlib.tools import python_tool

tool = python_tool(tier="local_unsafe")
result = tool.run(code="print(1 + 1)")
print(result.stdout)  # "2"
```

Pass the tool to `instruct()` to let the LLM write and execute code. Combine with
`uses_tool` and `tool_arg_validator` to constrain what gets generated (see examples
above).

> **Warning:** `tier="local_unsafe"` executes model-generated code as an
> unrestricted subprocess with no container isolation or resource limits.
> Do not use it in production contexts without sandboxing. Use
> `tier="docker"` or `tier="docker_unsafe"` for real process isolation.

### Execution tiers

The `tier` argument chooses where and how code runs. Pass it explicitly: omitting it
resolves to `"local_unsafe"` and emits a `UserWarning` so the trust decision is not
implicit. The same five tiers are available to the
[`PythonExecutionReq`](../api/mellea/stdlib/requirements/python_reqs.mdx) requirement,
which calls the parameter `execution_tier` and defaults it to `"static"`.

| Tier | Isolation | Speed | Use case |
| ---- | --------- | ----- | -------- |
| `"static"` | Analysis only (no execution) | Fastest | Fast validation before execution; the `PythonExecutionReq` default |
| `"local_unsafe"` | None (direct subprocess) | Fast | Trusted code only, max performance |
| `"local"` | None (policy control only) | Fast | Trusted code with resource limits (timeout, stdout/stderr limits) |
| `"docker_unsafe"` | Container isolation | Slower | Untrusted code, no policy restrictions |
| `"docker"` | Container isolation + policy | Slower | **Recommended:** untrusted or LLM-generated code with resource limits |

All executing tiers enforce the timeout and the `allowed_imports` allowlist, and they
can install `packages=` before the first run. Only the Docker tiers can copy files in
and out of the environment, and they need a running Docker daemon plus the `sandbox`
extra (`pip install 'mellea[sandbox]'`).

#### When to use each tier

**Use `"local"` or `"local_unsafe"`:**

- Code source is trusted (internal development)
- Running in an already-sandboxed environment (CI/CD, Docker container)
- Performance is critical and isolation overhead is unacceptable
- Development or demonstration purposes

**Use `"docker"` (recommended for production):**

- Code is LLM-generated or from untrusted sources
- Running in shared environments or production
- Security is more important than speed
- You need container-level process isolation and resource limits

### Capability policies

A `CapabilityPolicy` declares what an execution environment may do. Only some of its
fields are enforced at runtime; the rest are declarations that do not block anything.
Each field has a matching `ENFORCED_*` class attribute on `CapabilityPolicy` that
reports its status.

| Field | Enforced |
| ----- | -------- |
| `timeout`, `stdout_max_bytes`, `stderr_max_bytes` | Yes |
| `artifact_export_paths`, `packages` | Yes |
| `network_access`, `package_installation`, `subprocess_execution`, `env_var_access` | No |
| `filesystem_read_roots`, `filesystem_write_roots` | No (a `UserWarning` is emitted when set) |

The boolean restrictions are declarative on the local tiers regardless of the policy
you pass, including the default policy that `tier="local"` uses. Treat `"local"` as
resource limits, not isolation.

For production with LLM-generated code:

```python
# Requires: mellea[sandbox]
# Returns: Requirement
from mellea.stdlib.requirements.python_reqs import PythonExecutionReq
from mellea.stdlib.tools.execution_policy import CapabilityPolicy

# Safe: container isolation + timeout control
# Note: most policy fields are declared only (not enforced at runtime).
# The docker tier enforces: timeout, resource limits, and artifact export paths.
req = PythonExecutionReq(
    execution_tier="docker",  # Container-isolated execution
    policy=CapabilityPolicy(
        timeout=30,  # Kill after 30 seconds (enforced)
    ),
    max_output_chars=10_000,  # Limit output size (enforced)
)
```

#### How code execution works

When `m.instruct()` runs with a `PythonExecutionReq`, the requirement validation
executes the generated code during validation. This happens as a **side effect** of
validation, before the result is returned to the caller. The requirement runs the code
in an execution environment built from the `execution_tier` and `CapabilityPolicy` you
pass, and that environment enforces the timeout and tier behavior.

**Key point:** after `m.instruct()` returns, the code has _already been executed_. To
verify success, check the outcome (for example, that an output file exists). There is
no need for a separate execution step.

### Collecting files

Pass `artifact_dir=<Path>` to `python_tool()` to collect files the code writes:

- **Local tiers:** only files from a successful run (exit code 0) are returned, as
  `Artifact` objects on `result.artifacts`.
- **Docker tiers:** there is no host-shared directory to scan, so artifacts are
  returned only when the policy lists the container paths in `artifact_export_paths`.
  `artifact_dir` then becomes the host destination.
- **No `artifact_dir`:** a per-call temporary directory is used. It is kept alive while
  the returned `ExecutionResult` holds artifacts and cleaned up otherwise.

If the executed code imports `matplotlib`, `matplotlib.use('Agg')` is injected
automatically so plots are written to files. Pass `suppress_agg=True` to disable this.

## MCP tools

Mellea can consume tools from any [MCP](https://modelcontextprotocol.io/) server
and drop them into an agent loop. Install with `pip install 'mellea[tools]'`.

The workflow is two steps: discover what the server offers, then instantiate the
tools you want.

```python
# Requires: mellea[tools]
# Returns: list[MelleaTool]
from mellea.stdlib.tools.mcp import discover_mcp_tools, http_connection

connection = http_connection("https://api.example.com/mcp/", api_key="...")

specs = await discover_mcp_tools(connection)
tools = [s.as_mellea_tool() for s in specs if s.name in {"search", "fetch"}]
```

`http_connection`, `sse_connection`, and `stdio_connection` build the transport
config. Each tool invocation opens a short-lived session, so callers do not need
to manage the connection lifetime.

Once built, MCP tools work like any other `MelleaTool`: pass them via
`ModelOption.TOOLS` to `instruct()` or to `react()`:

```python
# Requires: mellea[tools]
# Returns: str
result, _ = await react(
    goal="Find recent pull requests I authored.",
    context=ChatContext(),
    backend=m.backend,
    tools=tools,
)
```

See [`docs/examples/mcp/github_activity_summary.py`](https://github.com/generative-computing/mellea/blob/main/docs/examples/mcp/github_activity_summary.py)
for a complete example against the hosted GitHub MCP server.

---

**See also:** [Tutorial 04: Making Agents Reliable](../tutorials/04-making-agents-reliable.md) | [Instruct, Validate, Repair](../concepts/instruct-validate-repair.md)
