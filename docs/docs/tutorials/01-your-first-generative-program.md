---
title: "Tutorial: Your First Generative Program"
description: "Build a document analysis pipeline step by step — from a single instruct() call to a composed, typed, validated generative program."
# diataxis: tutorial
---

In this tutorial you build a document analysis pipeline that extracts a summary,
classifies sentiment, and surfaces key issues from customer feedback. You start
with the simplest possible Mellea program and add reliability and structure at each
step.

By the end you will have covered:

- `instruct()` with user variables and requirements
- Rejection sampling and `SamplingResult`
- [`@generative`](../guide/glossary#generative) with `Literal` and [Pydantic](https://docs.pydantic.dev/) return types
- Composing generative functions into a pipeline

**Prerequisites:** [Quick Start](../getting-started/quickstart) complete,
`pip install mellea`, Ollama running locally with `granite4:micro` downloaded.

---

## Step 1: One instruction

Start with the smallest possible program: a single call to `instruct()`.

```python
import mellea

m = mellea.start_session()
summary = m.instruct(
    "Summarise this customer feedback in one sentence: "
    "The onboarding was confusing and took far too long. "
    "Support was helpful once I got through."
)
print(str(summary))
# Output will vary — LLM responses depend on model and temperature.
```

`instruct()` returns a [`ModelOutputThunk`](../guide/glossary#modeloutputthunk). Calling `str()` on it (or accessing
`.value`) gives you the string. This is already a generative program: it calls an
LLM and returns structured text.

The problem is reliability. The model might return two sentences, or three, or
include a preamble. Move to the next step to enforce the format.

---

## Step 2: Adding user variables

Hardcoding the text in the instruction string makes the function impossible to reuse.
Use `user_variables` and `{{double_braces}}` template syntax:

```python
import mellea

def summarize_feedback(m: mellea.MelleaSession, text: str) -> str:
    result = m.instruct(
        "Summarise this customer feedback in one sentence: {{text}}",
        user_variables={"text": text},
    )
    return str(result)


m = mellea.start_session()
feedback = (
    "The onboarding was confusing and took far too long. "
    "Support was helpful once I got through."
)
print(summarize_feedback(m, feedback))
# Output will vary — LLM responses depend on model and temperature.
```

The description is now a [Jinja2](https://jinja.palletsprojects.com/) template. Variables are rendered at generation time,
not embedded in the source code.

---

## Step 3: Enforcing constraints with requirements

Pass a list of plain-English requirements to constrain the output. Mellea checks
each requirement after generation and retries if any fail:

```python
import mellea

def summarize_feedback(m: mellea.MelleaSession, text: str) -> str:
    result = m.instruct(
        "Summarise this customer feedback in one sentence: {{text}}",
        requirements=[
            "The summary must be a single sentence.",
            "Include both positive and negative aspects if both are present.",
        ],
        user_variables={"text": text},
    )
    return str(result)


m = mellea.start_session()
feedback = (
    "The onboarding was confusing and took far too long. "
    "Support was helpful once I got through."
)
print(summarize_feedback(m, feedback))
# Output will vary — LLM responses depend on model and temperature.
```

Requirements are validated by LLM-as-a-judge by default. If a requirement fails,
Mellea sends the model the failure reason and asks it to repair the output.

---

## Step 4: Deterministic validation

For facts you can check in code — word counts, format, length — use
`simple_validate`:

```python
import mellea
from mellea.stdlib.requirements import req, simple_validate

def summarize_feedback(m: mellea.MelleaSession, text: str) -> str:
    result = m.instruct(
        "Summarise this customer feedback in one sentence: {{text}}",
        requirements=[
            req(
                "The summary must be a single sentence.",
            ),
            req(
                "Fewer than 30 words.",
                validation_fn=simple_validate(
                    lambda x: (
                        len(x.split()) < 30,
                        f"Summary has {len(x.split())} words; must be under 30.",
                    )
                ),
            ),
        ],
        user_variables={"text": text},
    )
    return str(result)


m = mellea.start_session()
feedback = (
    "The onboarding was confusing and took far too long. "
    "Support was helpful once I got through."
)
print(summarize_feedback(m, feedback))
# Output will vary — LLM responses depend on model and temperature.
```

The word-count check is deterministic: it runs in microseconds. The "single
sentence" check is left for LLM-as-a-judge since counting sentences is harder
to code reliably.

---

## Step 5: Rejection sampling and inspecting results

By default, `instruct()` retries up to twice if any requirement fails. Use
[`RejectionSamplingStrategy`](../guide/glossary#sampling-strategy) to control the budget and inspect results:

```python
import mellea
from mellea.stdlib.requirements import req, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

def summarize_feedback(m: mellea.MelleaSession, text: str) -> str:
    result = m.instruct(
        "Summarise this customer feedback in one sentence: {{text}}",
        requirements=[
            req(
                "Fewer than 30 words.",
                validation_fn=simple_validate(
                    lambda x: (
                        len(x.split()) < 30,
                        f"Summary has {len(x.split())} words; must be under 30.",
                    )
                ),
            ),
        ],
        strategy=RejectionSamplingStrategy(loop_budget=5),
        user_variables={"text": text},
        return_sampling_results=True,
    )

    if result.success:
        return str(result.result)
    else:
        # All attempts failed — use the first generation anyway
        print(f"Warning: failed after {len(result.sample_generations)} attempts")
        return str(result.sample_generations[0].value)


m = mellea.start_session()
print(summarize_feedback(m, "The onboarding was confusing and took far too long."))
```

With `return_sampling_results=True`, `instruct()` returns a [`SamplingResult`](../guide/glossary#samplingresult) with
`.success`, `.result`, and `.sample_generations`. This gives you programmatic
control over what to do when the model can not satisfy your requirements.

---

## Step 6: Typed classification with `@generative`

Switch to [`@generative`](../guide/glossary#generative) when you want the return type enforced at the Python level.
Add a sentiment classification step to the pipeline:

```python
from typing import Literal
from mellea import generative, start_session

@generative
def classify_sentiment(summary: str) -> Literal["positive", "negative", "mixed"]:
    """Classify the overall sentiment of the customer feedback summary."""

m = start_session()
sentiment = classify_sentiment(m, summary="Onboarding was confusing; support was helpful.")
print(sentiment)
# Output will vary — LLM responses depend on model and temperature.
# Expected one of: "positive", "negative", "mixed"
```

`@generative` generates the prompt from the function signature and docstring.
The model is constrained to return exactly one of the three allowed values.
`sentiment` is a Python string — no parsing needed.

---

## Step 7: Structured extraction with Pydantic

For richer structured output, use a Pydantic model as the return type:

```python
from pydantic import BaseModel
from mellea import generative, start_session

class FeedbackIssues(BaseModel):
    main_complaint: str
    positive_aspect: str | None
    urgency: str  # "low", "medium", "high"

@generative
def extract_issues(feedback: str) -> FeedbackIssues:
    """Extract the main complaint, any positive aspect, and urgency level from the feedback."""

m = start_session()
issues = extract_issues(
    m,
    feedback=(
        "The onboarding was confusing and took far too long. "
        "Support was helpful once I got through."
    ),
)
print(issues.main_complaint)
print(issues.positive_aspect)
print(issues.urgency)
# Output will vary — LLM responses depend on model and temperature.
```

The model output is automatically parsed into a `FeedbackIssues` instance.
Attribute access replaces manual JSON parsing.

---

## Step 8: Composing the pipeline

Assemble all the pieces into a complete pipeline:

```python
from typing import Literal
from pydantic import BaseModel

from mellea import MelleaSession, generative, start_session
from mellea.stdlib.requirements import req, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy


class FeedbackIssues(BaseModel):
    main_complaint: str
    positive_aspect: str | None
    urgency: str


@generative
def classify_sentiment(summary: str) -> Literal["positive", "negative", "mixed"]:
    """Classify the overall sentiment of the customer feedback summary."""


@generative
def extract_issues(feedback: str) -> FeedbackIssues:
    """Extract the main complaint, any positive aspect, and urgency from the feedback."""


def summarize_feedback(m: MelleaSession, text: str) -> str:
    result = m.instruct(
        "Summarise this customer feedback in one sentence: {{text}}",
        requirements=[
            req(
                "Fewer than 30 words.",
                validation_fn=simple_validate(
                    lambda x: (
                        len(x.split()) < 30,
                        f"Summary is {len(x.split())} words; must be under 30.",
                    )
                ),
            ),
        ],
        strategy=RejectionSamplingStrategy(loop_budget=5),
        user_variables={"text": text},
        return_sampling_results=True,
    )
    if result.success:
        return str(result.result)
    return str(result.sample_generations[0].value)


def analyze_feedback(feedback: str) -> None:
    m = start_session()

    summary = summarize_feedback(m, feedback)
    sentiment = classify_sentiment(m, summary=summary)
    issues = extract_issues(m, feedback=feedback)

    print(f"Summary:   {summary}")
    print(f"Sentiment: {sentiment}")
    print(f"Complaint: {issues.main_complaint}")
    print(f"Positive:  {issues.positive_aspect}")
    print(f"Urgency:   {issues.urgency}")


analyze_feedback(
    "The onboarding was confusing and took far too long. "
    "Support was helpful once I got through."
)
# Output will vary — LLM responses depend on model and temperature.
```

Each step in the pipeline is an independent LLM call with a typed interface. The
output of `summarize_feedback` feeds `classify_sentiment`; the original feedback
feeds `extract_issues`. There is no global state, no prompt accumulation — each
call is self-contained.

> **Full example:** [`docs/examples/instruct_validate_repair/101_email_with_requirements.py`](../../examples/instruct_validate_repair/101_email_with_requirements.py)

---

## What you have built

| Step | What it does |
| ---- | ------------ |
| `instruct()` | Calls the LLM with a structured instruction |
| User variables | Injects dynamic values into the prompt template |
| Requirements | Enforces plain-English constraints via IVR |
| `simple_validate` | Adds deterministic checks (word count, format) |
| `RejectionSamplingStrategy` | Controls retry budget and exposes `SamplingResult` |
| `@generative` + `Literal` | Type-safe classification with constrained output |
| `@generative` + Pydantic | Structured extraction with attribute access |
| Composition | Independent typed functions wired into a pipeline |

## Next steps

- [Instruct, Validate, Repair](../concepts/instruct-validate-repair) — deep dive
  into the IVR loop and sampling strategies
- [The Requirements System](../concepts/requirements-system) — advanced validators,
  preconditions, and debugging
- [Generative Functions](../guide/generative-functions) — `@generative` in depth
- [Working with Data](../guide/working-with-data) — passing documents and images
  into generative programs
