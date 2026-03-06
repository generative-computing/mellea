---
title: "Mellea Core Internals"
description: "The three core data structures and abstraction layers underlying every Mellea program."
sidebarTitle: "Core Internals"
# diataxis: explanation
---

> **Advanced:** This page is for contributors, backend developers, and anyone who
> wants to understand what happens when Mellea executes a request. If you are
> building applications with Mellea, you do not need this material.

Mellea's high-level API (`m.chat()`, `m.instruct()`, `@generative`) is built on three
core data structures. Understanding these structures and the abstraction layers above
them explains how Mellea achieves lazy evaluation, parallel dispatch, and composable
context management.

## The three core data structures

### `CBlock`

A `CBlock` (content block) is a wrapper around a string that marks a tokenisation
and KV caching boundary:

```python
from mellea.core import CBlock

block = CBlock("What is 1+1?")
```

`CBlock`s are the leaf nodes of every data dependency graph in Mellea. Importantly,
`CBlock` boundaries affect tokenisation:

```text
tokenise(CBlock(a) + CBlock(b)) == tokenise(a) + tokenise(b)
```

This may differ from `tokenise(a + b)`. When you care about KV cache reuse, CBlock
boundaries let you control exactly where the tokeniser makes splits.

### `Component`

A `Component` is a declarative structure that can depend on other `Component`s or
`CBlock`s. Components are the unit of composition in Mellea. `Message`,
[`Instruction`](../guide/glossary#instruction), `@mify` objects, and `@generative` functions all produce `Component`s.

### `ModelOutputThunk`

A `ModelOutputThunk` is a lazy reference to a computation result. It represents the
_future_ output of an LLM call — the call may or may not have been dispatched yet
when you receive the thunk. You can pass a thunk as an input to another `Component`
before the underlying computation has completed.

```python
thunk.is_computed()     # True if the value is already available
await thunk.avalue()    # Force evaluation; returns the actual value
```

This lazy evaluation model lets the backend see the full dependency graph of a
request before executing anything, enabling batching and optimisation.

## The abstraction layers

Each layer below is a thinner wrapper around the one beneath it. You work at
whatever level of abstraction the task requires.

### Layer 1: `MelleaSession`

The entry point for most programs. The session bundles a backend, a context, and
high-level methods. Everything is handled for you:

```python
from mellea import MelleaSession
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.context import SimpleContext

m = MelleaSession(backend=OllamaModelBackend("granite4:latest"), ctx=SimpleContext())
response = m.chat("What is 1+1?")
print(response.content)
```

When you call `m.chat()`, the session:

1. Wraps your string in a `Message` component
2. Passes the component and context to the backend
3. Updates the context with the result
4. Returns the response as a `Message`

### Layer 2: Functional API with explicit context

The functional API (`mfuncs`) exposes the same operations as stateless functions.
Context is threaded explicitly — you pass it in and get a new context back:

```python
import mellea.stdlib.functional as mfuncs
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.context import SimpleContext

response, next_context = mfuncs.chat(
    "What is 1+1?",
    context=SimpleContext(),
    backend=OllamaModelBackend("granite4:latest"),
)
print(response.content)
```

This is useful when you need to fork, merge, or snapshot context explicitly.

### Layer 3: Direct component construction with `mfuncs.act()`

`mfuncs.act()` accepts any component or `CBlock` directly. All other `mfuncs`
functions (`chat`, `instruct`, etc.) are thin wrappers that construct a component
and then call `act()`:

```python
import mellea.stdlib.functional as mfuncs
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.components import Instruction
from mellea.stdlib.context import SimpleContext

response, next_context = mfuncs.act(
    action=Instruction("What is 1+1?"),
    context=SimpleContext(),
    backend=OllamaModelBackend("granite4:latest"),
)
print(response.value)
```

### Layer 4: Async execution with `mfuncs.aact()`

Mellea's core is async. The synchronous API wraps the async operations with
`asyncio.run()`. For each method in `mfuncs` there is an `a*` async version:

```python
import asyncio
import mellea.stdlib.functional as mfuncs
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.components import Instruction
from mellea.stdlib.context import SimpleContext

async def main():
    response, _ = await mfuncs.aact(
        Instruction("What is 1+1?"),
        context=SimpleContext(),
        backend=OllamaModelBackend("granite4:latest"),
    )
    print(response.value)

asyncio.run(main())
```

### Layer 5: Lazy computation via `backend.generate_from_context()`

`mfuncs.aact()` is itself a convenience wrapper around the backend's
`generate_from_context()` method. Calling it directly returns a `ModelOutputThunk`
rather than an evaluated response:

```python
import asyncio
from mellea.backends.ollama import OllamaModelBackend
from mellea.core import CBlock
from mellea.stdlib.context import SimpleContext

async def main():
    backend = OllamaModelBackend("granite4:latest")
    ctx = SimpleContext()

    response, _ = await backend.generate_from_context(CBlock("What is 1+1?"), ctx=ctx)

    print(f"Computed: {response.is_computed()}")  # may be False
    print(await response.avalue())                 # forces evaluation
    print(f"Computed: {response.is_computed()}")  # True

asyncio.run(main())
```

### Layer 6: Composing lazy computations

Because thunks are lazy, you can pass a thunk as an input to a second computation
_before_ the first one has been evaluated. This lets the backend optimise across
the full dependency graph:

```python
import asyncio
from mellea.backends.ollama import OllamaModelBackend
from mellea.core import Backend, CBlock, Context
from mellea.stdlib.components import SimpleComponent
from mellea.stdlib.context import SimpleContext

async def main(backend: Backend, ctx: Context):
    x, _ = await backend.generate_from_context(CBlock("What is 1+1?"), ctx=ctx)
    y, _ = await backend.generate_from_context(CBlock("What is 2+2?"), ctx=ctx)

    # x and y may not have been computed yet — we can still use them as inputs
    z, _ = await backend.generate_from_context(
        SimpleComponent(instruction="What is x+y?", x=x, y=y),
        ctx=ctx,
    )

    print(f"x computed: {x.is_computed()}")
    print(f"y computed: {y.is_computed()}")
    print(await z.avalue())   # forces evaluation of the whole graph

asyncio.run(main(OllamaModelBackend("granite4:latest"), SimpleContext()))
```

The backend sees `z`'s dependency on `x` and `y`, evaluates them in order (or
in parallel if the backend supports it), and returns `z`'s result.

## Layer summary

| Layer | Entry point | Who uses it |
| ----- | ----------- | ----------- |
| `MelleaSession` | `m.chat()`, `m.instruct()` | Application developers |
| `mfuncs` synchronous | `mfuncs.chat()`, `mfuncs.act()` | Application developers needing context control |
| `mfuncs` async | `mfuncs.aact()`, `mfuncs.achat()` | Mellea contributors |
| `backend.generate_from_context()` | Thunks, `is_computed()`, `avalue()` | Backend developers, advanced users |
| Composition | `SimpleComponent` with thunk inputs | Backend developers |

## Template and prompt engineering

### TemplateFormatter

Mellea formats Python objects into LLM-readable text using a [`TemplateFormatter`](../guide/glossary#templateformatter).
It uses Jinja2 templates stored in a `templates/prompts/` directory. Each
component class can have its own template, looked up by class name.

The formatter resolves templates in this order:

1. Cached templates (from recent lookups)
2. The formatter's configured template path
3. The package that owns the component (`mellea` or a third-party package)

Within a template path, the formatter traverses subdirectories matching the model
ID before falling back to `default/`:

```text
templates/prompts/
├── default/
│   └── Instruction.jinja2    ← fallback for all models
└── granite/
    └── granite-3-2/
        └── instruct/
            └── Instruction.jinja2   ← used for ibm-granite/granite-3.2-8b-instruct
```

The formatter returns the template from the deepest matching directory. A model ID
of `ibm-granite/granite-3.2-8b-instruct` matches `granite/granite-3-2/instruct`
but not `ibm/` — only one path should match in any given templates directory.

### [`TemplateRepresentation`](../guide/glossary#templaterepresentation)

Each component's `format_for_llm()` method returns either a string or a
`TemplateRepresentation`. The `TemplateRepresentation` specifies:

- A reference to the component instance
- A dictionary of arguments passed to the template renderer
- A list of tools or functions related to the component
- Either a `template` (inline Jinja2 string) or a `template_order` (list of
  template file names to look up, where `*` means the class name)

The simplest approach is to return a string directly — this bypasses templating
entirely:

```python
def format_for_llm(self) -> str:
    return f"Summarise: {self.text}"
```

### Customising templates for an existing class

To change how an existing component is rendered, subclass it and override
`format_for_llm()`. Then create a new template file at the appropriate path.
See [`docs/examples/mify/rich_document_advanced.py`](../../examples/mify/rich_document_advanced.py)
for a worked example.

---

**See also:**
[Generative Programming](../concepts/generative-programming) |
[Working with Data](../guide/working-with-data) |
[Async and Streaming](../how-to/use-async-and-streaming)
