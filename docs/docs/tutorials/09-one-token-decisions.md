---
title: "Tutorial: One-Token Decisions"
description: "Route support tickets with decide, choose, and score. Each answer costs one forward pass."
---

**Prerequisites:** a Granite Switch model with the decision adapter,
served via vLLM on `localhost:8000` (see the
[adapter-functions page](../advanced/intrinsics.md#decisions-typed-one-token-answers)
if you have not set this up). This tutorial takes about 10 minutes.

## What you are building

You run a support desk. Tickets arrive as free-text messages. For each
one you need to answer three questions: is it urgent, which team
handles it, and how bad is it.

You could send each question to a chat model and parse the response,
but that is slow and fragile. The decision adapter takes a different
approach: it was trained to reply with a single letter (A, B, C, ...)
representing the best answer. Mellea sends your question, reads the
model's probability for each letter, and hands you back a typed result.
No parsing, no prose to interpret, one forward pass.

## Set up the backend

```python
from mellea.backends.openai import OpenAIBackend
from mellea.formatters import TemplateFormatter
from mellea.stdlib.components.intrinsic import decisions

MODEL = "ibm-granite/granite-switch-4.1-8b"  # change to your model

backend = OpenAIBackend(
    model_id=MODEL,
    formatter=TemplateFormatter(model_id=MODEL),
    base_url="http://localhost:8000/v1",  # change to your server
    api_key="EMPTY",
    load_embedded_adapters=True,
)
```

`load_embedded_adapters=True` tells the backend to look for adapter
functions built into the checkpoint. The decision adapter is one of
these. `TemplateFormatter` handles prompt assembly. You set this up
once and reuse the backend for every call.

The adapter's weights ship inside the Granite Switch checkpoint, so
there is no separate adapter file to download; you serve the
checkpoint and the adapter is there. (The same decision adapter is
also published as a standalone LoRA in the granitelib repositories
for use with non-Switch checkpoints.)

## Ask a yes/no question

```python
ticket = (
    "My payouts have failed three times. The bank says everything "
    "is fine. Can someone please fix this?"
)

answer = decisions.decide(ticket, backend, "Does this require urgent attention?")
print(answer.probability)  # 0.953
```

`decide()` returns a `PredicateAnswer` with a `probability` field: the
probability that the answer is yes. You can compare it directly against
a number:

```python
if answer >= 0.7:
    print("Escalating: this ticket needs same-day attention")
```

The comparison works because `>=`, `<`, etc. delegate to
`.probability`. A bare `if answer:` is always true, so always compare
against a threshold.

### Customizing what "yes" and "no" mean

By default the model sees "Yes / true" and "No / false" as the two
option descriptions. You can make these more specific to your domain:

```python
answer = decisions.decide(
    ticket,
    backend,
    "Does this require urgent attention?",
    true="Same-day human action is required",
    false="Routine handling is fine",
)
print(answer.probability)  # 0.948
```

The descriptions are part of the prompt, so more specific wording can
improve accuracy. The ordering is fixed: the `false` description
always comes first, `true` second.

## Pick from options

```python
route = decisions.choose(
    ticket,
    backend,
    "Which team should handle this?",
    {
        "payments": "Payout failures and payment processing",
        "account": "Login and account access",
        "other": "Something else",
    },
)
print(route.choice)         # "payments"
print(route.probabilities)  # {"payments": 0.982, "account": 0.003, "other": 0.014}
```

`choose()` returns the best option key and the full distribution over
all options. Only the descriptions are sent to the model; the keys are
yours. Ties break to the first key in insertion order.

You can pass between 2 and 10 options. If you need more than 10
categories, consider splitting into a two-level hierarchy.

## Rate on a scale

```python
severity = decisions.score(
    ticket,
    backend,
    "How urgent is this?",
    ["can wait a week", "this week", "today"],
)
print(severity.score)          # 1.89
print(severity.probabilities)  # {"0": 0.015, "1": 0.084, "2": 0.902}
```

`score()` returns the expected level and the distribution. Levels go
from lowest to highest. The score may land between levels (1.89 means
"leaning toward today").

The `confidence` field tells you how concentrated the model's answer
was: 1.0 means the model was certain about a single level, 0.0 means
it split its probability evenly between the extremes. It does not tell
you whether the answer is correct.

## Put it together

```python
def triage(ticket_text):
    """Route a support ticket and return a priority dict."""
    urgent = decisions.decide(
        ticket_text, backend, "Does this require urgent attention?"
    ) >= 0.7

    if not urgent:
        return {"team": "general", "severity": 0, "urgent": False}

    team = decisions.choose(
        ticket_text,
        backend,
        "Which team should handle this?",
        {
            "payments": "Payout failures and payment processing",
            "account": "Login and account access",
            "other": "Something else",
        },
    )
    severity = decisions.score(
        ticket_text,
        backend,
        "How urgent is this?",
        ["this week", "today", "immediately"],
    )
    return {
        "team": team.choice,
        "severity": severity.score,
        "urgent": True,
    }


result = triage(ticket)
print(result)
# {'team': 'payments', 'severity': 1.66, 'urgent': True}
```

Each call generates exactly one token, so there is no decode phase.
Latency depends on your hardware and serving stack; the
`latency_ms` field in the log entry tells you what you are getting.

## Using a conversation as the state

Sometimes you are judging a conversation rather than a block of text.
Pass a `ChatContext` and its messages go straight to the model:

```python
from mellea.stdlib.components import Message
from mellea.stdlib.context import ChatContext

conversation = (
    ChatContext()
    .add(Message("user", "My payouts have failed three times."))
    .add(Message("assistant", "I can help you with the payment issue."))
)

answer = decisions.decide(
    conversation, backend, "Is this conversation about payments?"
)
print(answer.probability)  # 0.940
```

The conversation's messages go directly to the model as the state.
Mellea still adds the trained system prompt and the decision turn
around them, but no separate state block is inserted. One caveat: the adapter was trained to judge a single block of
text, not a multi-turn dialogue. For long conversations, you may get
better results by extracting the key facts into a summary string and
passing that instead.

## What gets logged

Every decision emits a log entry. Here is what you see at INFO level:

```text
decision: capability=decision model=ibm-granite/granite-switch-4.1-8b
  question='Which team handles this?' option_labels=['A', 'B', 'C']
  answer_token=A distribution={'A': 0.982, 'B': 0.003, 'C': 0.014}
  label_mass=0.9998 missing_labels=[] returned_labels=3/3
  latency_ms=308.1 state_chars=98 state_sha256_8=129b4f91
```

The most useful field is `label_mass`: it says how much of the model's
probability landed on the option labels (as opposed to other tokens in
the vocabulary). Values above 0.95 are normal. If it drops below 0.5,
the call would have raised an error.

The state text is not logged at INFO level (only its length and a
hash), so sensitive content stays out of logs. Set
`MELLEA_LOGS_LEVEL=DEBUG` to see the full state and option
descriptions.

## When things go wrong

**Label-mass floor.** The model answered with something other than an
option label. This is a backstop for serving or template problems; if
you see it, check your endpoint is running the correct model with the
right chat template. It does not fire for off-topic questions (the
model picks a label even for nonsensical input).

**Server rejected the logprobs request.** vLLM needs `--max-logprobs
20` (the default). If set lower, the server returns an HTTP 400 when
the decision call asks for 20 logprobs.

**Invocation marker in your text.** The adapter activates at the
string `### DECISION` in the prompt. If that string appears in your
state, question, or option descriptions, the call raises
`AloraActivationError`. Reword the offending text.

## Summary

| Helper | Returns | Good for |
| ------ | ------- | -------- |
| `decide(state, backend, question)` | `.probability` (float) | Yes/no questions |
| `choose(state, backend, question, options)` | `.choice` (str) and `.probabilities` | Routing, picking |
| `score(state, backend, question, levels)` | `.score` (float) and `.probabilities` | Ratings, severity |

All three accept a string, a dict or list, or a `ChatContext` as the
state. Any other `Context` type raises a `TypeError`.

A few things are fixed and cannot be changed: the model always
generates exactly one token, decodes greedily (`temperature` is pinned to
0.0), never streams, and is never given tools to call (the decision
prompt has no tool block). Logit-shaping options such as `logit_bias` and
the repetition, frequency and presence penalties are rejected as well,
because a served backend may take its logprobs after applying them. If a
`model_options` key, whether passed per call or set on the backend,
conflicts with any of these, the call raises rather than silently
producing a broken result. On the local Hugging Face backend the
probabilities are read from the raw logits at the label token ids, so
generation options cannot move them.

For the full API reference, see the
[adapter-functions page](../advanced/intrinsics.md#decisions-typed-one-token-answers).
For runnable scripts, see
[`decisions.py`](https://github.com/generative-computing/mellea/blob/main/docs/examples/intrinsics/decisions.py)
and
[`decisions_llamacpp.py`](https://github.com/generative-computing/mellea/blob/main/docs/examples/intrinsics/decisions_llamacpp.py).
