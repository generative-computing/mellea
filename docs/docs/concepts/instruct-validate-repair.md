---
title: "The Instruction Model"
description: "How instruct(), requirements, and the IVR loop work in Mellea."
# diataxis: how-to
---

# The Instruction Model

**Prerequisites:** [Quick Start](../getting-started/quickstart.md) complete,
`pip install mellea`, Ollama running locally.

`instruct()` is the primary API in Mellea. It builds a structured `Instruction`
component — not a raw chat message — with a description, requirements, user variables,
grounding context, few-shot examples, and images. The instruction is rendered through
Jinja2 templates and run through an instruct–validate–repair (IVR) loop by default.

## Basic `instruct()`

```python
import mellea

m = mellea.start_session()
email = m.instruct("Write an email inviting interns to an office party at 3:30pm.")
print(str(email))
# Output will vary — LLM responses depend on model and temperature.
```

`instruct()` returns a `ModelOutputThunk`. Access the result as a string with
`str(email)` or via `email.value`.

## User variables

Embed dynamic values in your description using `{{double_braces}}`. The description
is a Jinja2 template; values are injected at generation time via `user_variables`:

```python
import mellea

def write_email(m: mellea.MelleaSession, name: str, notes: str) -> str:
    email = m.instruct(
        "Write an email to {{name}} using the notes following: {{notes}}.",
        user_variables={"name": name, "notes": notes},
    )
    return str(email)

m = mellea.start_session()
print(write_email(m, name="Olivia", notes="Organized intern events."))
# Output will vary — LLM responses depend on model and temperature.
```

Variables work in requirements too — you can use the same `{{var}}` syntax anywhere
in the instruction description or requirement strings.

## Requirements

Requirements are declarative constraints. They serve two purposes:

1. They are embedded in the prompt so the model knows what to aim for.
2. They are checked after generation; if any fail, the IVR loop asks the model to
   repair its output.

Pass plain strings for LLM-checked requirements:

```python
import mellea

m = mellea.start_session()
email = m.instruct(
    "Write an email inviting the team to a meeting.",
    requirements=[
        "The email should have a salutation.",
        "Use only lower-case letters.",
    ],
)
print(str(email))
# Output will vary — LLM responses depend on model and temperature.
```

## Custom validation functions

For deterministic checks, attach a `validation_fn` to a `Requirement`:

```python
from mellea import start_session
from mellea.core import Requirement
from mellea.stdlib.requirements import simple_validate

word_limit_req = Requirement(
    "Use fewer than 100 words.",
    validation_fn=simple_validate(lambda output: len(output.split()) < 100),
)

m = start_session()
email = m.instruct(
    "Write an email inviting the team to a meeting.",
    requirements=["Be formal.", word_limit_req],
)
print(str(email))
```

`simple_validate` wraps a callable that returns a `bool` (or a `(bool, str)` tuple
with a failure reason) into a validation function.

### Shorthand helpers

`req()` and `check()` are concise constructors for `Requirement`:

```python
from mellea import start_session
from mellea.stdlib.requirements import check, req, simple_validate

m = start_session()
email = m.instruct(
    "Write an email to {{name}}.",
    requirements=[
        req("The email should have a salutation."),
        req(
            "Use only lower-case letters.",
            validation_fn=simple_validate(lambda x: x.lower() == x),
        ),
        check("Do not mention purple elephants."),
    ],
    user_variables={"name": "Olivia"},
)
print(str(email))
# Output will vary — LLM responses depend on model and temperature.
```

- `req(description)` — creates a `Requirement` with an optional `validation_fn`
- `check(description)` — alias for `req()`, reads naturally for boolean constraints

## Sampling strategies and the IVR loop

By default, `instruct()` uses `RejectionSamplingStrategy(loop_budget=2)`: it
generates once, validates all requirements, and retries up to two times if any fail.

Configure the loop explicitly with `strategy`:

```python
from mellea import start_session
from mellea.stdlib.requirements import req, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

m = start_session()
result = m.instruct(
    "Write an email to {{name}}.",
    requirements=[
        req(
            "Use only lower-case letters.",
            validation_fn=simple_validate(lambda x: x.lower() == x),
        ),
    ],
    strategy=RejectionSamplingStrategy(loop_budget=5),
    user_variables={"name": "Olivia"},
    return_sampling_results=True,
)

if result.success:
    print(str(result.result))
else:
    # All attempts failed — fall back to the first generation
    print(str(result.sample_generations[0].value))
```

With `return_sampling_results=True`, `instruct()` returns a `SamplingResult` instead
of a `ModelOutputThunk`. This lets you inspect whether validation passed and access
all intermediate generations.

> **Advanced:** SOFAI (`SOFAISamplingStrategy`) is a dual-model strategy that routes
> between a fast and a slow model based on confidence. See
> [Inference-Time Scaling](../advanced/inference-time-scaling.md).

## Grounding context

Attach reference documents to an instruction for retrieval-augmented generation:

```python
from mellea import start_session

m = start_session()
answer = m.instruct(
    "Given the documents in the context, answer: {{query}}",
    user_variables={"query": "What is the capital of France?"},
    grounding_context={"doc0": "France is a country in Western Europe. Its capital is Paris."},
)
print(str(answer))
# Output will vary — LLM responses depend on model and temperature.
```

`grounding_context` maps string keys to document text. These are injected as
reference material in the prompt. See [Working with Data](../guide/working-with-data.md)
for richer document handling using MObjects and `RichDocument`.

## ICL examples

In-context learning (ICL) examples provide few-shot demonstrations. They are rendered
as input–output pairs inside the `Instruction` component's Jinja2 template, giving the
model concrete examples to follow.

> **Note (review needed):** The `instruct()` `icl_examples` parameter API needs
> verification against the current source before documenting the full signature here.

## Images

Pass images to `instruct()` with the `images` parameter. Accepts both Mellea
`ImageBlock` and PIL images:

```python
from PIL import Image
from mellea import start_session
from mellea.core import ImageBlock

m = start_session()  # requires a vision-capable backend and model
pil_image = Image.open("photo.jpg")
img_block = ImageBlock.from_pil_image(pil_image)

response = m.instruct(
    "Describe what is in this image.",
    images=[img_block],
)
print(str(response))
# Output will vary — LLM responses depend on model and temperature.
```

> **Backend note:** Vision requires a model that supports image inputs (e.g.,
> `qwen2.5vl:7b` via the OpenAI backend). The default Ollama/Granite setup does not
> support images.

## Multi-turn with `ChatContext`

`instruct()` works with `ChatContext` for stateful multi-turn conversations:

```python
from mellea import start_session
from mellea.stdlib.context import ChatContext

m = start_session(ctx=ChatContext())
m.chat("Make up a simple math problem.")
m.chat("Now solve the problem you just made up.")

print(str(m.ctx.last_output()))
# Output will vary — LLM responses depend on model and temperature.
```

`ChatContext` accumulates turns. `SimpleContext` (the default) discards the previous
turn on each call.

## `chat()` vs `instruct()`

`chat()` is a lighter-weight alternative that sends a plain message with no
requirements and no sampling strategy:

```python
from mellea import start_session
from mellea.stdlib.context import ChatContext

m = start_session(ctx=ChatContext())
response = m.chat("What is 2 + 2?")
print(str(response))
```

Use `chat()` for conversational back-and-forth where you don't need the IVR machinery.
Use `instruct()` when you want requirements, validation, or structured output.

---

**Previous:** [Generative Programming](./generative-programming.md) |
**Next:** [Generative Functions](../guide/generative-functions.md)
