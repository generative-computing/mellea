---
title: "Configure model options"
description: "Set temperature, seed, max tokens, system prompts, and other backend parameters at session level or per call."
# diataxis: how-to
---

# Configure model options

Most LLM APIs accept parameters such as temperature, max tokens, and seed. Mellea exposes
these through the `ModelOption` enum, which works uniformly across all backends, and also
lets you pass backend-native keys directly.

**Prerequisites:** `pip install mellea` complete, a backend available (see
[Installation](../getting-started/installation.md)).

## The ModelOption enum

Import `ModelOption` from `mellea.backends`. The enum provides cross-backend names
for the most common parameters:

```python
import mellea
from mellea.backends import ModelOption, model_ids
from mellea.backends.ollama import OllamaModelBackend

m = mellea.MelleaSession(
    backend=OllamaModelBackend(
        model_id=model_ids.IBM_GRANITE_4_HYBRID_SMALL,
        model_options={ModelOption.SEED: 42},
    )
)

answer = m.instruct(
    "What is 2x2?",
    model_options={
        ModelOption.TEMPERATURE: 0.5,
        ModelOption.MAX_NEW_TOKENS: 10,
    },
)
print(str(answer))
# Output will vary — LLM responses depend on model and temperature.
```

Options set on the backend apply to every call on that session. Options passed to a specific
`m.*` call apply only to that call and take precedence over the session-level values.

You can also pass backend-native key names directly — Mellea forwards any key it does not
recognize to the underlying API unchanged. This means you can copy model option dicts from
existing codebases without translation:

```python
answer = m.instruct(
    "Summarize this in one sentence.",
    model_options={
        "temperature": 0.3,
        "num_predict": 50,   # Ollama-native key
    },
)
```

## Precedence rules

When the same option is set in multiple places, the following rules apply:

1. A `ModelOption` key always takes precedence over its backend-native equivalent.
2. Options passed to a `m.*` call override the corresponding session-level options for that
   call only.

```python
# Backend initialised with these options
backend_options = {
    "seed": 1,
    ModelOption.MAX_NEW_TOKENS: 100,
    "temperature": 1.0,
}

# Options passed at call time
call_options = {
    "seed": 2,
    ModelOption.SEED: 3,   # takes precedence over "seed": 2
    "num_predict": 50,
}

# Options actually sent to the model for this call:
# seed = 3  (ModelOption.SEED wins)
# max_new_tokens = 100  (from backend; not overridden)
# temperature = 1.0  (from backend; not overridden)
# num_predict = 50  (new key from call)
```

## Pushing and popping model state

Sessions support temporarily overriding model options for a series of calls, then restoring
the original state:

```python
m = mellea.start_session()

m.push_model_options({ModelOption.TEMPERATURE: 0.0, ModelOption.SEED: 99})

# These calls use temperature=0.0, seed=99
result1 = m.instruct("List three capitals of South America.")
result2 = m.instruct("List three capitals of Europe.")

m.pop_model_options()

# Back to original session options
result3 = m.instruct("Write a short poem.")
```

This is useful when you need deterministic output for a batch of calls within a larger,
non-deterministic session.

## System prompts

Set a system prompt with `ModelOption.SYSTEM_PROMPT`. At session level it applies to all
subsequent calls; at call level it applies only to that call.

```python
m = mellea.MelleaSession(
    backend=OllamaModelBackend(
        model_id=model_ids.IBM_GRANITE_4_HYBRID_MICRO,
        model_options={
            ModelOption.SYSTEM_PROMPT: "You are a concise technical assistant. Never use bullet points."
        },
    )
)

answer = m.instruct("Explain what a context manager is in Python.")
```

Using `ModelOption.SYSTEM_PROMPT` is recommended over constructing a system-role message
manually. Some backend APIs do not serialize system-role messages correctly and expect the
system prompt as a separate parameter — `ModelOption.SYSTEM_PROMPT` handles this correctly
across all backends.

---

**Previous:** [Write Custom Verifiers](./write-custom-verifiers.md) |
**Next:** [Use Images and Vision Models](./use-images-and-vision.md)
