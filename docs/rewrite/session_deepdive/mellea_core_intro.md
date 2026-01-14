## Introduction to Mellea's Core

This chapter introduces Mellea's core primitives by progressively peeling away abstractions. We'll start with the familiar `MelleaSession` API and work our way down through the layers to understand what's actually happening underneath.

Mellea's core is designed to support innovation at the interface of models and the inference stack. Our goal is to provide a flexible set of primitives that allow model developers to extend their models in arbitrary ways and then surface these model extensions to programmers via Mellea's standard library.

The best way to understand these layers is to see them in action. Let's trace through six progressively more detailed implementations of the same simple task: asking a model "What is 1+1?". With each step, we'll peel away one layer of abstraction to see what's happening underneath.

## Step 0: The MelleaSession API

Most Mellea programs start here. The `MelleaSession` is a convenient wrapper that bundles together a backend, a context, and a set of high-level methods like `chat()`, `instruct()`, and `query()`:

```python
from mellea import MelleaSession
from mellea.stdlib.base import SimpleContext
from mellea.backends.ollama import OllamaModelBackend


m = MelleaSession(
    backend=OllamaModelBackend("granite4:latest"), context=SimpleContext()
)
response = m.chat("What is 1+1?")
print(response.content)
```

This is clean and convenient. The session manages the state of both the context and the backend for you. When you call `m.chat()`, it:
1. Takes your string input
2. Wraps it in a `Message` component
3. Passes it to the backend
4. Updates the context with the result
5. Returns the response

The magic here is that the session handles threading context and backend together. You don't need to think about either one separately.

## Step 1: Functional API with Explicit Context

Now let's peel away the session wrapper. Instead of using a session object, we can call functions directly from the functional API, which requires us to explicitly manage context:

```python
import mellea.stdlib.functional as mfuncs
from mellea.stdlib.base import SimpleContext
from mellea.backends.ollama import OllamaModelBackend

response, next_context = mfuncs.chat(
    "What is 1+1?",
    context=SimpleContext(),
    backend=OllamaModelBackend("granite4:latest"),
)

print(response.content)
```

What changed? Now we're calling `mfuncs.chat()` instead of `m.chat()`. The functional API is stateless—it takes a context, performs an operation, and returns a new context. Notice that we get back `next_context`: this is the updated context after the operation.

This level of abstraction makes it clear that context management is explicit. You pass in a context, you get out a new context. This is useful when you want more control, or when you're building systems that need to fork or merge contexts.

## Step 2: Working with `mfuncs.act` and CBlocks

Let's continue unwrapping layers. The functional API accepts strings, but underneath it converts them to `CBlock`s (content blocks). A `CBlock` is Mellea's atomic unit of content:

```python
import mellea.stdlib.functional as mfuncs
from mellea.stdlib.base import SimpleContext, CBlock
from mellea.backends.ollama import OllamaModelBackend

response, next_context = mfuncs.act(
    CBlock("What is 1+1?"),
    context=SimpleContext(),
    backend=OllamaModelBackend("granite4:latest"),
)

print(response.value)
```

There are two changes in this next step.

First, we switched from a `Message` `Component` to a `CBlock`. The `CBlock` type is lower-level than the `Message` used by `chat()`. A `CBlock` is a thin wrapper around raw strings and is the fundamental unit that the backend understands. When you use `m.chat()`, Mellea wraps your string in a `Message` component, which then gets formatted into `CBlock`s for the backend. Here, we're skipping that intermediate step. You should think of `Components` as a compound data type that contains may other `CBlocks` and `Components`. The leaf nodes of that compound data type are always `CBlocks`.

One important note that will become important later: `CBlocks` are tokenziation (and therefore KV caching) boundaries. So, the tokenization of the `concatenate(CBlock(str_a), CBlock(str_b))` is `concatenate(tokenize(str_a), tokenize(str_b))`. This may be different from `tokenize(concatenate(str_a, str_b))`.

Second, we're now calling `mfuncs.act()` instead of `mfuncs.chat()`. The `act()` method can take _any_ component or `CBlock`. The `mfuncs.chat()` function is just a thin wrapper around `act()`. This is true for all of the other functions defined in `mfuncs` as well -- under the hood, each constructs a `Component` and the ncalls `mfuncs.act()` on that `Component`.


## Step 3: Async Operations

Mellea's core is asynchronous. The synchronous API you've been using (like `m.chat()`) is just a wrapper around the async operations, using `asyncio.run()` internally. For each method in `mfuncs`, there's a corresponding async version named by prepending an `a` to the synchronous version. For example, the async version of `act()` is `aact()` and the async version of `chat()` is `achat()`.

By moving to the `a*` methods, we can start to see Mellea's actual execution model.

```python
import mellea.stdlib.functional as mfuncs
from mellea.stdlib.base import SimpleContext, CBlock, Context
from mellea.backends.ollama import OllamaModelBackend
from mellea.backends import Backend
import asyncio


async def main(backend: Backend, ctx: Context):
    response, next_context = await mfuncs.aact(
        CBlock("What is 1+1?"), context=ctx, backend=backend
    )

    print(response.value)


asyncio.run(main(OllamaModelBackend("granite4:latest"), SimpleContext()))
```

Notice that we now have an `async` function and we're using `await`. The functional API's `aact()` is the asynchronous version of `act()`. This shows that at Mellea's core, operations are async.

## Step 4: Lazy Computation and Thunks

Now we're getting into Mellea's more sophisticated features. The async `mfuncs` versions are themselves also just convienance wrappers around a two-step process where we first dispatch an LLM call and then wait for its response. Let's call `backend.generate_from_context()` directly instead of using the functional API:

```python
import mellea.stdlib.functional as mfuncs
from mellea.stdlib.base import SimpleContext, CBlock, Context
from mellea.backends.ollama import OllamaModelBackend
from mellea.backends import Backend
import asyncio


async def main(backend: Backend, ctx: Context):
    # This is not actually an async function; the computation ends immediately. It must be awaited because we create the thunk.
    response, next_context = await backend.generate_from_context(
        CBlock("What is 1+1?"),
        ctx=ctx,
    )

    print(f"Currently computed: {response.is_computed()}")
    print(await response.avalue())
    print(f"Currently computed: {response.is_computed()}")


asyncio.run(main(OllamaModelBackend("granite4:latest"), SimpleContext()))
```

Here's where it gets interesting. The `response` returned from `backend.generate_from_context()` is not the actual value. It's a **thunk**, a lazy reference to a computation that may or may not have been performed yet.

Notice the calls to `is_computed()` and `avalue()`. The first time we print `is_computed()`, it likely returns `False` (though this depends on the backend and timing). Then we call `await response.avalue()` to actually compute the value. After that, `is_computed()` will always `True`.

This lazy evaluation is crucial to Mellea's design. By returning thunks, the backend can:
- Delay computation until it's actually needed
- Allow multiple thunks to be created before any computation happens
- Enable automatic merging and optimization of computations

## Step 5: Composing Computations with Components

Finally, let's see what happens when we work with multiple computations together. This shows the real power of lazy evaluation:

```python
import mellea.stdlib.functional as mfuncs
from mellea.stdlib.base import (
    SimpleContext,
    CBlock,
    Context,
    SimpleComponent,
    Component,
)
from mellea.backends.ollama import OllamaModelBackend
from mellea.backends import Backend
import asyncio


async def main(backend: Backend, ctx: Context):
    x, _ = await backend.generate_from_context(CBlock("What is 1+1?"), ctx=ctx)

    y, _ = await backend.generate_from_context(CBlock("What is 2+2?"), ctx=ctx)

    # here, x and y have not necessarily been computed!

    response, _ = await backend.generate_from_context(
        SimpleComponent(instruction="What is x+y?", x=x, y=y),
        ctx=ctx,
    )

    print(f"x currently computed: {x.is_computed()}")
    print(f"y currently computed: {y.is_computed()}")
    print(f"response currently computed: {response.is_computed()}")
    print(await response.avalue())
    print(f"response currently computed: {response.is_computed()}")


asyncio.run(main(OllamaModelBackend("granite4:latest"), SimpleContext()))
```

Now we're seeing the full power of Mellea's design. We create `x` and `y` as thunks by calling `generate_from_context()` twice. At this point, neither computation has necessarily run.

Then we create a `SimpleComponent` that depends on `x` and `y`. We pass this component to `generate_from_context()` to get `response`. Still, none of these computations may have run yet!

Only when we call `await response.avalue()` do we actually force computation. At that point, the backend can see that `response` depends on `x` and `y`, so it computes those first, and then computes `response` using the results.

This is the key abstraction that makes Mellea powerful:

1. **CBlocks** are a wrapper around strings denoting tokenization and KV caching boundaries.
2. **Components** are declarative structures that can depend on other `Component`s or `CBlock`s.
3. **ModelOutputThunks** are lazy references to computation results emitted from an inference engine.

For each request, the Mellea Backend sees the full computation graph before any actual computation happens. This allows for per-request optimizations, batching, and context sharing that wouldn't be possible if each call were evaluated immediately.

## Understanding the Layers

Let's summarize what we've learned by peeling away abstractions:

| Abstraction level | Description | Example |
|-------|---------|---------|
| **MelleaSession** | Extremely simple high-level interface | `m.chat("...")` |
| **mfuncs.chat(), .instruct(), etc.** | Explicit context threading | `mfuncs.chat(..., context=ctx, backend=backend)` |
| **mfuncs.act()** | Direct component creation | `CBlock(...)` |
| **mfuncs.aact()** | Async execution model | `await mfuncs.aact(...)` |
| **backend.generate_from_context()** | Lazy evaluation with thunks | `response.is_computed()`, `await response.avalue()` |
| **ModelOutputThunks** | Computation graphs with components | `SimpleComponent(...)` with thunk dependencies |

The reason these layers exist is that different use cases need different levels of control:

- **Application developers** usually work at Step 0 or 1, using the session or functional API. This handles the common cases cleanly.
- **Framework developers** might work at Step 2 or 3, creating components and managing async operations.
- **Advanced users** building complex inference pipelines work at Steps 4 and 5, directly manipulating thunks and computation graphs.

Each layer is built on top of the previous one. When you call `m.chat()`, you're ultimately using the same backend infrastructure as the raw `backend.generate_from_context()` call -- you're just using more convenient abstractions on top of it.
