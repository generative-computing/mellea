---
title: "Generative Functions"
description: "How the @generative decorator turns a Python function signature into an LLM-backed implementation."
# diataxis: explanation
---

In classical programming, a pure function takes inputs and produces outputs deterministically.
In a generative program, a function can have the same interface but delegate its implementation
to an LLM. Mellea calls these [**generative functions**](../reference/glossary#generative-function) and provides the [`@generative`](../reference/glossary#generative) decorator
to define them.

> **Looking to use this in code?** See [Generative Functions](../how-to/generative-functions) for practical examples and API details.

## The @generative decorator

Decorate a function with `@generative` and give it a return type annotation. The function body
is replaced by the LLM at call time — the signature and docstring guide the model in producing
the output.

```python
from typing import Literal
from mellea import generative, start_session

@generative
def classify_sentiment(text: str) -> Literal["positive", "negative"]:
    """Classify the sentiment of the input text as 'positive' or 'negative'."""
    ...

m = start_session()
sentiment = classify_sentiment(m, text="I love this!")
print(sentiment)
# Output will vary — LLM responses depend on model and temperature.
```

The session `m` is always the first argument when calling a generative function. Mellea
constructs the prompt automatically from the function name, parameters, docstring, and return
type. The `Literal` annotation constrains the output to exactly two values — the model cannot
return anything else.

Generative functions can also return Pydantic models for structured multi-field output:

```python
from typing import Literal
from pydantic import BaseModel
from mellea import generative, start_session

class FeedbackSummary(BaseModel):
    summary: str
    sentiment: Literal["positive", "negative", "mixed"]
    key_issue: str

@generative
def analyze_feedback(text: str) -> FeedbackSummary:
    """Analyze customer feedback and extract a summary, sentiment, and the main issue raised."""
    ...

m = start_session()
result = analyze_feedback(m, text="Onboarding took too long but support was excellent.")
print(result.sentiment)
# Output will vary — LLM responses depend on model and temperature.
```

## Compositionality

One of the key benefits of generative functions is that they compose the same way ordinary
functions do. Independent libraries can each expose generative functions, and those functions
can be combined without either library knowing about the other.

Consider two independent libraries: one that summarizes documents, and one that proposes
decisions or risks from summaries.

```python
from mellea import generative

# Summarizer library
@generative
def summarize_meeting(transcript: str) -> str:
    """Summarize the meeting transcript into a concise paragraph of main points."""
    ...

@generative
def summarize_contract(contract_text: str) -> str:
    """Produce a natural language summary of contract obligations and risks."""
    ...

# Decision aides library
@generative
def propose_business_decision(summary: str) -> str:
    """Given a structured summary with clear recommendations, propose a business decision."""
    ...

@generative
def generate_risk_mitigation(summary: str) -> str:
    """If the summary contains risk elements, propose mitigation strategies."""
    ...
```

These two libraries do not always compose meaningfully — a meeting transcript may or may not
contain actionable risks. Calling `generate_risk_mitigation` on a summary that contains no
risks produces noise.

## Guarded nondeterminism

To compose libraries safely without coupling them, use generative functions as contracts — small
classifiers that gate whether a composition makes sense:

```python
from typing import Literal
from mellea import generative

@generative
def contains_actionable_risks(summary: str) -> Literal["yes", "no"]:
    """Check whether the summary contains references to business risks or exposure."""
    ...

@generative
def has_structured_conclusion(summary: str) -> Literal["yes", "no"]:
    """Determine whether the summary contains a clearly marked conclusion or recommendation."""
    ...
```

These contracts let you write dynamic composition logic in ordinary Python:

```python
from mellea import start_session

m = start_session()

transcript = "... meeting transcript text ..."
summary = summarize_meeting(m, transcript=transcript)

if contains_actionable_risks(m, summary=summary) == "yes":
    mitigation = generate_risk_mitigation(m, summary=summary)
    print(f"Mitigation: {mitigation}")
else:
    print("No actionable risks found.")

if has_structured_conclusion(m, summary=summary) == "yes":
    decision = propose_business_decision(m, summary=summary)
    print(f"Decision: {decision}")
else:
    print("Summary lacks a structured conclusion.")
```

This pattern — using generative functions as boolean guards on composition — is sometimes called
**guarded nondeterminism**. It keeps the two libraries fully decoupled while still making
nonsensical compositions impossible at runtime.

Without these guards, your only options are to tightly couple the libraries (rewrite one to
satisfy the other's interface) or add requirements to the decision function that silently fail
if unmet. Neither approach scales. With contracts, the coupling logic lives in the guard
functions, which can be maintained and tested independently.

## Generative functions vs instruct()

`@generative` and `m.instruct()` serve different purposes:

| | `@generative` | `m.instruct()` |
| --- | --- | --- |
| Interface | Named function with typed signature | Inline prompt string |
| Return type | Python type annotation | String (or constrained by requirements) |
| Reusability | High — call like any function | Low — prompt embedded at call site |
| Composability | Natural Python composition | Manual |

Use `@generative` when you want a named, typed, reusable LLM-backed operation. Use
`m.instruct()` for one-off generation where a function abstraction would be overhead.

**See also:** [Instruct, Validate, Repair](./instruct-validate-repair) |
[The Requirements System](./requirements-system) |
[Tools and Agents](../how-to/tools-and-agents)
