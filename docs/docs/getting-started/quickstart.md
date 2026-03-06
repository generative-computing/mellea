---
title: "Quick Start"
description: "Run your first generative program in minutes."
# diataxis: tutorial
---

# Quick Start

**Prerequisites:** [Ollama](https://ollama.ai) installed and running locally,
[Installation](./installation.md) complete.

## Hello world

By default, `start_session()` connects to Ollama and uses **IBM Granite 4 Micro**
(`granite4:micro`). Make sure Ollama is running before you run this:

```python
import mellea

m = mellea.start_session()
email = m.instruct("Write an email inviting interns to an office party at 3:30pm.")
print(str(email))
# Output will vary — LLM responses depend on model and temperature.
```

Three lines: create a session, instruct, print. The `instruct()` call returns a
[`ModelOutputThunk`](../guide/glossary#modeloutputthunk); call `str()` on it (or access `.value`) to get the string.

> **Full example:** [`docs/examples/tutorial/simple_email.py`](../../examples/tutorial/simple_email.py)

## User variables

Embed dynamic values in instructions using `{{double_braces}}`. The description is
treated as a [Jinja2](https://jinja.palletsprojects.com/) template:

```python
import mellea

def write_email(m: mellea.MelleaSession, name: str, notes: str) -> str:
    email = m.instruct(
        "Write an email to {{name}} using the notes following: {{notes}}.",
        user_variables={"name": name, "notes": notes},
    )
    return str(email)

m = mellea.start_session()
print(write_email(
    m,
    name="Olivia",
    notes="Organized intern events and handled issues with snack delivery.",
))
# Output will vary — LLM responses depend on model and temperature.
```

## Requirements

Pass a list of plain-English requirements to constrain the output. Mellea runs an
instruct–validate–repair loop: if any requirement fails, it asks the model to fix
its output:

```python
import mellea

def write_email(m: mellea.MelleaSession, name: str, notes: str) -> str:
    email = m.instruct(
        "Write an email to {{name}} using the notes following: {{notes}}.",
        requirements=[
            "The email should have a salutation.",
            "Use only lower-case letters.",
        ],
        user_variables={"name": name, "notes": notes},
    )
    return str(email)

m = mellea.start_session()
print(write_email(m, name="Olivia", notes="Organized intern events."))
# Output will vary — LLM responses depend on model and temperature.
```

The repair loop retries up to two times by default. See
[Instruct, Validate, Repair](../concepts/instruct-validate-repair.md) for control
over loop budget, custom validators, and the full `instruct()` API.

## Core concepts

**Sessions** — [`MelleaSession`](../guide/glossary#melleasession) is the main entry point. `start_session()` creates one
with defaults: Ollama backend, Granite 4 Micro, [`SimpleContext`](../guide/glossary#context) (single-turn).

**Instructions** — `instruct()` builds a structured `Instruction` component, not a
raw chat message. It supports a description, requirements, user variables, grounding
context, and few-shot examples.

**Contexts** — `SimpleContext` holds a single turn. [`ChatContext`](../guide/glossary#context) accumulates turns for
multi-turn conversations. Pass `ctx=ChatContext()` to `start_session()` for stateful
chat.

**Backends** — Pluggable model providers. Ollama is the default. OpenAI, [LiteLLM](../guide/glossary#litellm--litellmbackend),
HuggingFace, and WatsonX are also supported. See
[Backends and Configuration](../guide/backends-and-configuration.md).

## Troubleshooting

**`granite4:micro` not found** — run `ollama pull granite4:micro` before starting.

**Python 3.13 `outlines` install failure** — `outlines` requires a Rust compiler.
Either install [Rust](https://www.rust-lang.org/tools/install) or pin Python to 3.12.

**Intel Mac torch errors** — create a conda environment and run
`conda install 'torchvision>=0.22.0'`, then `uv pip install mellea` inside it.

---

**Previous:** [Installation](./installation.md) |
**Next:** [Tutorial: Your First Generative Program](../tutorials/01-your-first-generative-program.md)
