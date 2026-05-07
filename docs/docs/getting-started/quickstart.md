---
canonical: "https://docs.mellea.ai/getting-started/quickstart"
title: "Quick Start"
description: "Run your first generative program in minutes."
# diataxis: tutorial
---

**Prerequisites:** [Ollama](https://ollama.ai) installed and running locally,
[Installation](./installation) complete.

## Hello world

By default, `start_session()` connects to Ollama and uses **IBM Granite 4 Micro**
(`granite4.1:3b`). Make sure Ollama is running before you run this:

```python
import mellea

m = mellea.start_session()
email = m.instruct("Write an email inviting interns to an office party at 3:30pm.")
print(str(email))
```

```text Sample output
Subject: Invitation to the Office Party

Dear [Intern's Name],

I hope this message finds you well.

We would like to extend a warm invitation to all interns for our
upcoming Office Party, which is scheduled for today at 3:30 PM in
the main conference room.

The event will provide an excellent opportunity for us to get to
know each other better and create lasting memories together. There
will be light refreshments, music, and networking opportunities.

Kindly let us know if you are able to attend by replying to this
email at your earliest convenience. If any special accommodations
are needed due to health or religious reasons, please inform us as
soon as possible so we can make the necessary arrangements.

We look forward to seeing many of you there!

Best Regards,

[Your Name]
[Your Position]
```

> **Note:** LLM output is non-deterministic. Your exact results will vary by model and temperature, but you should see a well-formed email in a similar format.

Three lines: create a session, instruct, print. The `instruct()` call returns a
[`ModelOutputThunk`](../reference/glossary#modeloutputthunk); call `str()` on it (or access `.value`) to get the string.

> **Full example:** [`docs/examples/tutorial/simple_email.py`](https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/simple_email.py)

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
```

```text Sample output
Subject: Appreciation for Your Efforts in Managing Intern Events
         and Snack Delivery

Dear Olivia,

I hope this message finds you well. I am writing to express my
sincere appreciation for your outstanding efforts in organizing our
recent intern events as well as handling the issue related to snack
deliveries.

Your ability to organize these events is truly commendable. You have
managed to create an environment that is not only productive but also
enjoyable for all participants, including our interns. Your dedication
and hard work are evident in how seamlessly each event was executed,
ensuring a smooth flow of activities and maintaining high levels of
engagement among attendees.

Furthermore, I must acknowledge your proactive approach when it came
to addressing the issue concerning snack deliveries. The fact that
you were able to swiftly resolve this problem speaks volumes about
your organizational skills, attention to detail, and ability to
manage unexpected challenges effectively.

Your contributions have been invaluable in making these events
successful and enjoyable for everyone involved. Your commitment
towards maintaining a positive work environment reflects highly on
our team culture.

Thank you once again for all that you do. We are fortunate to have
someone as dedicated and capable as you within our organization.

Best Regards,

[Your Name]
[Your Position]
```

> **Note:** LLM output is non-deterministic. Your exact results will vary.

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
```

```text Sample output
subject: organized intern events

hello olivia,

i hope this message finds you well.

recently, we've been working diligently on organizing some exciting
intern events for our team members to engage with and learn from
each other. these activities are designed not just for fun but also
as a platform for interns to showcase their skills and ideas.

please feel free to reach out if you need any further information
or wish to participate in planning some of the upcoming events.
your input would be greatly appreciated.

thank you for being an integral part of our team, and we look
forward to seeing everyone at these gatherings.

best regards,

[your name]
```

> **Note:** LLM output is non-deterministic. With the `"Use only lower-case letters"` requirement, the output will be all lowercase. If the first attempt isn't successful, Mellea retries automatically up to a specified limit.

The repair loop retries up to two times by default. See
[Instruct, Validate, Repair](../concepts/instruct-validate-repair) for control
over loop budget, custom validators, and the full `instruct()` API.

## Core concepts

**Sessions** — [`MelleaSession`](../reference/glossary#melleasession) is the main entry point. `start_session()` creates one
with defaults: Ollama backend, Granite 4 Micro, [`SimpleContext`](../reference/glossary#context) (single-turn).

**Instructions** — `instruct()` builds a structured `Instruction` component, not a
raw chat message. It supports a description, requirements, user variables, grounding
context, and few-shot examples.

**Contexts** — `SimpleContext` holds a single turn. [`ChatContext`](../reference/glossary#context) accumulates turns for
multi-turn conversations. Pass `ctx=ChatContext()` to `start_session()` for stateful
chat.

**Backends** — Pluggable model providers. Ollama is the default. OpenAI, [LiteLLM](../reference/glossary#litellm--litellmbackend),
HuggingFace, and WatsonX are also supported. See
[Backends and Configuration](../how-to/backends-and-configuration).

## Troubleshooting

**`granite4.1:3b` not found** — run `ollama pull granite4.1:3b` before starting.

**Python 3.13 `outlines` install failure** — `outlines` requires a Rust compiler.
Either install [Rust](https://www.rust-lang.org/tools/install) or pin Python to 3.12.

**Intel Mac torch errors** — create a conda environment and run
`conda install 'torchvision>=0.22.0'`, then `uv pip install mellea` inside it.
