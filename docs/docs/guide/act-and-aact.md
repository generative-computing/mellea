---
title: "act() and aact()"
description: "Work directly with Components using act(), aact(), and the functional API."
# diataxis: how-to
---

# act() and aact()

**Prerequisites:** [Instruct, Validate, Repair](../concepts/instruct-validate-repair.md) complete,
`pip install mellea`, Ollama running locally.

`act()` is the generic method on `MelleaSession` that runs any `Component` and
returns a result. Every other session method is built on it:

- `instruct()` creates an `Instruction` component and passes it to `act()`
- `chat()` creates a `Message` component and passes it to `act()` with `strategy=None`
- `query()` and `transform()` wrap mified objects into components and pass them to `act()`

Use `act()` when you need to work directly with a component — for custom components,
fine-grained control, or building your own inference loops.

## Three levels of abstraction

These three snippets all produce the same result:

```python
import mellea
from mellea import start_session
from mellea.stdlib import functional as mfuncs
from mellea.stdlib.components import Instruction
from mellea.stdlib.context import SimpleContext

# Level 1: instruct() — builds the Instruction for you
m = start_session()
result = m.instruct("Write a haiku about the ocean.")

# Level 2: act() — you build the Instruction, session threads context
m = start_session()
instruction = Instruction(description="Write a haiku about the ocean.")
result = m.act(instruction)

# Level 3: mfuncs.act() — you manage context and backend directly
ctx = SimpleContext()
backend = mellea.start_session().backend
instruction = Instruction(description="Write a haiku about the ocean.")
result, new_ctx = mfuncs.act(instruction, context=ctx, backend=backend)
```

## Basic usage

Pass any `Component` to `act()`. It returns a `ModelOutputThunk`:

```python
from mellea import start_session
from mellea.stdlib.components import Instruction

m = start_session()
instruction = Instruction(
    description="List three facts about Mars.",
    requirements=["Each fact must be on its own line."],
)
result = m.act(instruction)
print(str(result))
# Output will vary — LLM responses depend on model and temperature.
```

## Working with Messages

`Message` is a component with a role and content string. Pass `strategy=None` to
skip the IVR loop — this is what `chat()` does internally:

```python
from mellea import start_session
from mellea.stdlib.components import Message

m = start_session()
result = m.act(Message("user", "What is the capital of France?"), strategy=None)
print(str(result))
# Output will vary — LLM responses depend on model and temperature.
```

## Working with Documents

Use `Document` to pass structured text with optional title and ID metadata:

```python
from mellea import start_session
from mellea.stdlib.components import Document, Message

m = start_session()
doc = Document(
    "Mellea is a framework for structured LLM programming.",
    title="Mellea Overview",
    doc_id="doc-1",
)
msg = Message("user", "Summarize this document.", documents=[doc])
result = m.act(msg, strategy=None)
print(str(result))
# Output will vary — LLM responses depend on model and temperature.
```

For rich document processing (PDFs, tables), see
[Working with Data](./working-with-data.md).

## Validation and sampling strategies

`act()` accepts the same `requirements` and `strategy` parameters as `instruct()`.
The default is `RejectionSamplingStrategy(loop_budget=2)`:

```python
from mellea import start_session
from mellea.core import Requirement
from mellea.stdlib.components import Instruction
from mellea.stdlib.sampling import RejectionSamplingStrategy

m = start_session()
instruction = Instruction(description="List three facts about Mars.")

candidate = m.act(
    instruction,
    requirements=[Requirement("Each fact must be on its own line.")],
    strategy=RejectionSamplingStrategy(loop_budget=3),
    return_sampling_results=True,
)

if candidate.success:
    print(str(candidate.result))
else:
    print(str(candidate.sample_generations[0].value))
```

See [Instruct, Validate, Repair](../concepts/instruct-validate-repair.md) and
[Inference-Time Scaling](../advanced/inference-time-scaling.md) for full details on requirements
and validation.

## Structured output

Pass a Pydantic `BaseModel` as the `format` parameter for constrained decoding:

```python
from pydantic import BaseModel
from mellea import start_session
from mellea.stdlib.components import Instruction

class Planet(BaseModel):
    name: str
    diameter_km: float
    has_rings: bool

m = start_session()
instruction = Instruction(description="Describe Saturn.")
result = m.act(instruction, format=Planet)
print(result.value)  # A Planet instance
# Output will vary — LLM responses depend on model and temperature.
```

## The functional API

> **Advanced:** `mellea.stdlib.functional` exposes `act()` and `aact()` as
> standalone functions. You pass `context` and `backend` explicitly instead of
> relying on a session to thread them.

```python
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib import functional as mfuncs
from mellea.stdlib.components import Instruction
from mellea.stdlib.context import SimpleContext

backend = OllamaModelBackend(model_id="phi4-mini:latest")
ctx = SimpleContext()

instruction = Instruction(description="Explain gravity in one sentence.")
result, new_ctx = mfuncs.act(instruction, context=ctx, backend=backend)
print(str(result))
# Output will vary — LLM responses depend on model and temperature.
```

The functional `act()` returns a `(ModelOutputThunk, Context)` tuple. With
`return_sampling_results=True` it returns `(SamplingResult, Context)`.

Use the functional API when you need to branch context for parallel explorations or
build custom inference loops. For most use cases, the session API (`m.act()`) is
simpler.

## Async with `aact()`

`aact()` is the async counterpart. Same signature, same return types:

```python
import asyncio
from mellea import start_session
from mellea.stdlib.components import Instruction

async def main():
    m = start_session()
    instruction = Instruction(description="Write a limerick about debugging.")
    result = await m.aact(instruction)
    print(str(result))
    # Output will vary — LLM responses depend on model and temperature.

asyncio.run(main())
```

The functional async version is `mfuncs.aact()`:

```python
result, new_ctx = await mfuncs.aact(instruction, context=ctx, backend=backend)
```

For parallel generation and streaming patterns, see
[Async and Streaming](../how-to/use-async-and-streaming.md).

---

**Previous:** [Backends and Configuration](./backends-and-configuration.md) |
**Next:** [Async and Streaming](../how-to/use-async-and-streaming.md)
