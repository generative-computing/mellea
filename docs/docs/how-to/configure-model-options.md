---
canonical: "https://docs.mellea.ai/how-to/configure-model-options"
title: "Configure model options"
description: "Set temperature, seed, max tokens, system prompts, and other backend parameters at session level or per call."
# diataxis: how-to
---

Most LLM APIs accept parameters such as temperature, max tokens, and seed. Mellea exposes
these through the `ModelOption` enum, which works uniformly across all backends, and also
lets you pass backend-native keys directly.

**Prerequisites:** `pip install mellea` complete, a backend available (see
[Installation](../getting-started/installation)).

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

## Reference: all ModelOption keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `ModelOption.TEMPERATURE` | `float` | backend default | Sampling temperature. |
| `ModelOption.MAX_NEW_TOKENS` | `int` | backend default ⚠️ | Maximum tokens to generate. Backend defaults vary widely — set this explicitly in production code. |
| `ModelOption.SEED` | `int` | `None` | Random seed for reproducible output. |
| `ModelOption.SYSTEM_PROMPT` | `str` | `None` | System prompt prepended to every call on the session. |
| `ModelOption.STREAM` | `bool` | `False` | Enable streaming output. |
| `ModelOption.STREAM_TIMEOUT` | `float \| None` | `60.0` | Timeout in seconds applied to every chunk, including time-to-first-token. If no chunk arrives within this window the stream aborts with a `TimeoutError`. Set to `None` to disable. Increase for slow local inference. |
| `ModelOption.STOP_SEQUENCES` | `list[str]` | `None` | Strings that halt generation when produced by the model. |
| `ModelOption.THINKING` | varies | `None` | Enable or configure reasoning/thinking mode (model-dependent). |
| `ModelOption.CONTEXT_WINDOW` | `int` | backend default | Context window size override. |
| `ModelOption.TOOLS` | `list[MelleaTool]` | `None` | Tools exposed to the model for tool calling. |
| `ModelOption.TOOL_CHOICE` | `str` | `"auto"` | Tool selection strategy (`"none"`, `"auto"`, or a specific tool name). |

Keys marked with a backend default are forwarded to the underlying API unchanged; the
value the model sees depends on the backend's own defaults.

> **Warning:** `MAX_NEW_TOKENS` backend defaults vary widely and some are very low — for
> example vLLM defaults to 16 tokens, which will silently truncate most real responses.
> Always set `ModelOption.MAX_NEW_TOKENS` explicitly in production code:
>
> ```python
> m.instruct(
>     "Summarise this document.",
>     model_options={ModelOption.MAX_NEW_TOKENS: 1024},
> )
> ```
>
> A value of 512–2048 covers most chat and instruction use cases. For code generation
> or long-form output, set a higher value to match your expected output length.

## Streaming timeout

By default Mellea waits up to 60 seconds for each chunk, including the first
(time-to-first-token). If the backend stops sending without closing the connection
the stream aborts with a `TimeoutError` rather than hanging indefinitely.

> **Note for slow or local backends:** The 60 s default covers time-to-first-token.
> Large models on CPU, long prompts, or heavily loaded servers can take longer than
> this before producing the first token. Use a higher value or `None` for those
> deployments.

```python
from mellea.backends import ModelOption

# Tighter bound for a known-fast remote endpoint
mot = await m.ainstruct(
    "Summarise this document.",
    model_options={ModelOption.STREAM: True, ModelOption.STREAM_TIMEOUT: 10},
)

# Larger value for slow local inference (e.g. large model on CPU)
mot = await m.ainstruct(
    "Write a long analysis.",
    model_options={ModelOption.STREAM: True, ModelOption.STREAM_TIMEOUT: 300},
)

# Disable entirely — original unbounded behaviour
mot = await m.ainstruct(
    "Write a long analysis.",
    model_options={ModelOption.STREAM: True, ModelOption.STREAM_TIMEOUT: None},
)
```

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
