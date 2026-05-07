---
canonical: "https://docs.mellea.ai/concepts/context-and-sessions"
title: "Context and Sessions"
description: "How Component, Backend, Context, and Session fit together in Mellea's architecture."
# diataxis: explanation
---

Every call to an LLM in Mellea passes through four layers: [**Component**](../reference/glossary#component), [**Backend**](../reference/glossary#backend),
[**Context**](../reference/glossary#context), and **Session**. Understanding how these fit together explains both why
Mellea is structured the way it is and how to extend it effectively.

> **Looking to use this in code?** See [Context and Sessions](../how-to/use-context-and-sessions) for practical examples and session extension patterns.

## The four layers

### Components

A `Component` is the structured representation of a single interaction with an LLM.
When you call `m.instruct(...)`, Mellea creates an `Instruction` component — a
composite data structure that holds the description, requirements, user variables,
grounding context, and ICL examples for that call.

Components are composable: a component can contain other components. This is how
Mellea keeps prompts modular. An `Instruction` contains `Requirement` objects;
a `Requirement` is itself a component. The composition forms a directed acyclic
graph (DAG) that the backend renders into a prompt.

The leaf nodes of the DAG are `CBlock` objects — atomic content blocks that hold
raw text or a parsed representation of a model output.

### Backends

A `Backend` takes a `Component`, formats it into a prompt, sends it to an LLM, and
returns the model output as a [`ModelOutputThunk`](../reference/glossary#modeloutputthunk). The `Thunk` is a lazy wrapper: it
holds the raw model output and parses it on access (via `.value` or `str()`).

The backend is responsible for:

- Rendering the component tree into the prompt format the model expects (chat
  messages, template strings, etc.)
- Making the network or process call to the LLM
- Parsing the response into a typed representation where applicable

Different backends — Ollama, OpenAI, HuggingFace, WatsonX — share the same
component interface. A `Component` does not know which backend will render it.

### Contexts

A `Context` records the history of interactions during a session. It is a linked
list (or tree, when you clone a session) of components and their outputs.

The context serves two purposes:

1. **Prompt construction** — the backend calls `ctx.view_for_generation()` to get
   the components that should appear in the prompt. For `ChatContext`, this includes
   all prior turns. For [`SimpleContext`](../reference/glossary#simplecontext), it includes only the current instruction.

2. **Validation** — during the IVR loop, requirement validators receive the
   `Context` object. They can call `ctx.last_output()` to inspect the most recent
   model output, or examine the full history for more complex checks.

### Sessions

[`MelleaSession`](../reference/glossary#melleasession) is the developer-facing layer. It wraps a backend and a context,
exposes the `instruct()`, `chat()`, `validate()`, and other methods you use in your
code, and handles the bookkeeping that ties components, context updates, and backend
calls together.

`start_session()` returns a `MelleaSession` with defaults: Ollama backend, Granite 4
Micro model, and `SimpleContext`.

## Session creation

There are two ways to create a session in Mellea:

1. `start_session(...)` — a convenience factory that creates and configures a
   `MelleaSession`
2. `MelleaSession(backend, ctx)` — explicit construction when you want to provide
   the backend and context objects yourself

For most application code, prefer `start_session()`. It is shorter, clearer, and
uses sensible defaults. Reach for direct `MelleaSession(...)` construction when
you need full control over backend instantiation or want to inject a preconfigured
backend object.

| Pattern | What you provide | Defaults included | Best for |
| --- | --- | --- | --- |
| `start_session()` | Backend name, model ID, optional context, model options, backend kwargs | Yes — backend class resolution, default model, and `SimpleContext` | Quick prototyping, tutorials, most production app code |
| `MelleaSession(backend, ctx)` | A fully constructed backend object and a context object | No | Advanced usage, custom backend setup, dependency injection, testing |

### `start_session()`: recommended default

Use `start_session()` when you want the standard Mellea entry point:

```python
from mellea import start_session

m = start_session()
result = m.instruct("Summarise this paragraph in one sentence.")
```

Use it when:

- You want the shortest path from example code to a working session
- The built-in backend selection by name is sufficient
- You want to configure the model with `model_id`, `model_options`, or backend kwargs
  without manually instantiating backend objects

You can still customize the session while keeping the convenience factory:

```python
from mellea import start_session
from mellea.stdlib.context import ChatContext

m = start_session(
    backend_name="ollama",
    ctx=ChatContext(),
    model_options={"temperature": 0.2},
)
```

### `MelleaSession(backend, ctx)`: explicit construction

Use direct construction when you already have a backend instance or need explicit
control over how it is created:

```python
from mellea import MelleaSession
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.context import SimpleContext

backend = OllamaModelBackend(
    "granite4.1:3b",
    model_options={"temperature": 0.2},
)
m = MelleaSession(backend, SimpleContext())
```

Use it when:

- You are wiring Mellea into a larger application with dependency injection
- You need backend setup that happens before session creation
- You want tests to construct or swap backend instances explicitly

The two patterns are not different session types: `start_session()` is the
higher-level convenience API, and it returns a regular `MelleaSession`.

## `SimpleContext` vs `ChatContext`

The two built-in context types implement very different history policies.

### `SimpleContext`

`SimpleContext` is stateless between calls. Each `instruct()` or `chat()` call sees
only the current instruction — no prior turns. The prompt is entirely determined by
the current component.

Use `SimpleContext` (the default) when:

- Calls are logically independent (a batch of classification tasks, extraction from
  different documents)
- You are composing `@generative` functions whose results flow through Python code,
  not through chat history
- You want predictable, isolated calls with no context accumulation

### `ChatContext`

`ChatContext` preserves the full message history across calls. The model sees all
prior turns on every new request.

```python
from mellea import start_session
from mellea.stdlib.context import ChatContext

m = start_session(ctx=ChatContext())
m.chat("Make up a math problem.")
m.chat("Now solve the problem you just made up.")

print(str(m.ctx.last_output()))
# The model's answer to the second question, referencing the first.
```

Use `ChatContext` when:

- You are building a stateful conversation (a chat assistant, an interactive
  planning session)
- The model needs to refer back to prior turns to give a coherent response
- You are implementing agentic loops where each step builds on previous results

### The context window trade-off

`ChatContext` accumulates history indefinitely. As history grows, prompts become
larger, latency increases, and cost rises. For long sessions, consider using
`ctx.reset_to_new()` or `m.reset()` to clear history at a natural breakpoint.

The `ChatContext` constructor accepts a `window_size` parameter to limit how many
prior turns are retained:

```python
from mellea.stdlib.context import ChatContext

# Keep only the last 10 turns
ctx = ChatContext(window_size=10)
```

For most structured extraction or transformation tasks, `SimpleContext` (the default)
is the right choice. Reserve `ChatContext` for applications where conversational
coherence is genuinely required.

## Why explicit context management matters

Implicit context — a global chat history that grows without bounds — is a common
source of subtle failures in generative programs:

- **Prompt degradation:** A very long history can cause the model to lose focus on
  the current instruction, producing outputs that drift from what was asked.
- **Context window overflow:** Every LLM has a maximum token budget. Exceeding it
  causes truncation or errors.
- **Hard-to-debug behaviour:** When context is implicit and global, it is hard to
  reproduce failures — the same instruction can produce different results depending
  on what happened earlier in the session.

Mellea's response is to make context explicit and local. Components encapsulate
the context they need; `SimpleContext` ensures independence by default; `ChatContext`
is opt-in for cases where history is genuinely needed.

## Session cloning

`m.clone()` creates a copy of a session at its current context state. Both the
original and the clone start from the same history and then diverge independently:

```python
import asyncio
from mellea import start_session
from mellea.stdlib.context import ChatContext

async def main():
    m = start_session(ctx=ChatContext())
    m.instruct("Multiply 2 × 2.")

    m1 = m.clone()
    m2 = m.clone()

    # Both branches see the "Multiply 2 × 2" exchange in their history.
    r1 = await m1.ainstruct("Multiply that result by 3.")
    r2 = await m2.ainstruct("Multiply that result by 5.")

    print(str(r1))  # 12
    print(str(r2))  # 20

asyncio.run(main())
```

Cloning is useful for:

- Exploring multiple continuations of the same context (tree-structured reasoning)
- Running parallel comparisons with the same conversational history
- Implementing best-of-N sampling at the conversation level rather than the
  single-turn level

## Inspecting context

The `ctx` object exposes helpers for reading the current session state:

```python
from mellea import start_session
from mellea.stdlib.context import ChatContext

m = start_session(ctx=ChatContext())
m.chat("What is the capital of France?")
m.chat("And its population?")

# Most recent model output
last = m.ctx.last_output()
print(last.value)

# Full last turn: user message + model output
turn = m.ctx.last_turn()
```

`last_turn()` returns a [`ContextTurn`](../reference/glossary#contextturn) with `.input` and `.output` fields. It is
useful for observability or when you need to log exactly what the model received and
produced.

## Extending sessions

`MelleaSession` is a regular Python class. Subclassing it lets you inject custom
behaviour — input filtering, output validation, logging, rate limiting — into
every call. See [Context and Sessions how-to](../how-to/use-context-and-sessions)
for a worked example.

---

**See also:** [Context and Sessions how-to](../how-to/use-context-and-sessions) |
[Async and Streaming](../how-to/use-async-and-streaming)
