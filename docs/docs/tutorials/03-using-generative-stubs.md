---
title: "Tutorial: Using Generative Stubs"
description: "Replace ad-hoc instruct() calls with typed, composable @generative functions."
# diataxis: tutorial
---

This tutorial shows how to build composable LLM-backed functions using the
[`@generative`](../guide/glossary#generative) decorator — functions with typed return values, docstring-driven
prompts, and consistent behaviour that you can reuse across a codebase.

By the end you will have covered:

- Defining `@generative` functions with typed returns
- Composing multiple generative functions into a pipeline
- Controlling behaviour via [`ChatContext`](../guide/glossary#chatcontext) and context injection
- Precondition and postcondition validation patterns

**Prerequisites:** [Tutorial 01](./01-your-first-generative-program) complete,
`pip install mellea`, Ollama running locally with `granite4:micro` downloaded.

---

## Step 1: Your first @generative function

A `@generative` function uses its name, type annotation, and docstring as the
prompt. Call it by passing a `MelleaSession` as the first argument:

```python
import mellea
from mellea import generative

@generative
def classify_sentiment(text: str) -> str:
    """Classify the sentiment of the text as 'positive', 'negative', or 'neutral'."""

m = mellea.start_session()
result = classify_sentiment(m, text="The product arrived damaged and support ignored me.")
print(str(result))
# Output will vary — LLM responses depend on model and temperature.
```

The return type annotation shapes the output. With `-> str`, the model returns
free text. For constrained output, use `Literal`:

```python
from typing import Literal
from mellea import generative

@generative
def classify_sentiment(text: str) -> Literal["positive", "negative", "neutral"]: ...
```

Now the output is guaranteed to be one of those three strings.

## Step 2: Typed and structured returns

Generative functions support any JSON-serialisable return type — `str`, `int`,
`bool`, `list`, `dict`, and Pydantic models:

```python
from typing import Literal

import mellea
from mellea import generative
from pydantic import BaseModel

class FeedbackAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    key_issue: str
    actionable: bool

@generative
def analyse_feedback(text: str) -> FeedbackAnalysis:
    """Extract sentiment, the main issue, and whether it is actionable."""

m = mellea.start_session()
result = analyse_feedback(
    m,
    text="The onboarding took two hours and nothing was explained clearly.",
)
print(result.sentiment, result.key_issue, result.actionable)
# Output will vary — LLM responses depend on model and temperature.
```

The return value is a validated `FeedbackAnalysis` instance. If the model output
doesn't conform, Mellea retries.

## Step 3: Compose generative functions

Because each `@generative` function is just a Python function, you compose them
the same way as any other code:

```python
import mellea
from mellea import generative

# FeedbackAnalysis is the Pydantic model from Step 2 above.

@generative
def analyse_feedback(text: str) -> FeedbackAnalysis:
    """Extract sentiment, the main issue, and whether it is actionable."""

@generative
def draft_response(issue: str) -> str:
    """Draft a polite, empathetic customer service response addressing this issue."""

@generative
def translate(text: str, target_language: str) -> str:
    """Translate the text into the target language."""

def handle_ticket(m, feedback: str, language: str = "English") -> str:
    analysis = analyse_feedback(m, text=feedback)
    if not analysis.actionable:
        return "Logged for review."
    response = draft_response(m, issue=analysis.key_issue)
    if language != "English":
        response = translate(m, text=response, target_language=language)
    return str(response)

m = mellea.start_session()
print(handle_ticket(m, "The app crashes on login every time.", "French"))
# Output will vary — LLM responses depend on model and temperature.
```

Each function is an independent LLM call. The composition logic stays in
ordinary Python.

> **Full example:** [`docs/examples/generative_stubs/generate_with_context.py`](https://github.com/generative-computing/mellea/blob/main/docs/examples/generative_stubs/generate_with_context.py)

## Step 4: Steer all functions via context

A key advantage of `@generative` functions over direct `instruct()` calls: you can
change the behaviour of every function in a session by injecting context once.

```python
from mellea import generative, start_session
from mellea.stdlib.context import ChatContext
from mellea.core import CBlock

@generative
def grade_essay(essay: str) -> int:
    """Grade the essay and return a score from 1 to 100."""

@generative
def give_feedback(essay: str) -> list[str]:
    """Return a list of specific improvement suggestions for the essay."""

essay = "The cat sat on the mat. It was a nice mat. The cat liked it."

m = start_session(ctx=ChatContext())

# No context — grader decides independently.
grade = grade_essay(m, essay=essay)
feedback = give_feedback(m, essay=essay)
print(f"Grade: {grade}")
print(f"Feedback: {feedback}")
# Output will vary — LLM responses depend on model and temperature.

# Inject a persona — both functions now behave as this grader.
m.ctx = m.ctx.add(CBlock(
    "You are an encouraging primary school teacher. "
    "Keep grades above 70 unless there is a serious problem. "
    "Frame all feedback kindly."
))

grade = grade_essay(m, essay=essay)
feedback = give_feedback(m, essay=essay)
print(f"Grade with teacher context: {grade}")
print(f"Feedback with teacher context: {feedback}")
# Output will vary — LLM responses depend on model and temperature.

# Reset and try a different persona.
m.reset()
m.ctx = m.ctx.add(CBlock(
    "You are a grammar specialist. Focus entirely on sentence structure, "
    "punctuation, and vocabulary. Ignore content quality."
))

grade = grade_essay(m, essay=essay)
print(f"Grade with grammar context: {grade}")
# Output will vary — LLM responses depend on model and temperature.
```

`m.reset()` clears injected context while keeping the session and backend alive.

## Step 5: Pre- and postcondition validation

For production pipelines, validate inputs before the LLM call and outputs
afterwards using plain Python:

```python
from typing import Literal
from mellea import generative, start_session, MelleaSession

@generative
def analyse_client_profile(profile: str) -> dict:
    """Extract risk_tolerance, time_horizon, and liquidity_needs from the profile."""

@generative
def detect_prohibited_language(text: str) -> Literal["clean", "prohibited"]:
    """Detect whether the text contains phrases like 'guaranteed returns' or 'no risk'."""

@generative
def generate_advice_letter(profile: str) -> str:
    """Generate a personalised financial advice letter based on the client profile."""

def check_preconditions(analysis: dict) -> None:
    required = ["risk_tolerance", "time_horizon", "liquidity_needs"]
    missing = [f for f in required if not analysis.get(f)]
    if missing:
        raise ValueError(f"Incomplete profile — missing: {', '.join(missing)}")

def check_postconditions(letter: str, lang_flag: str) -> None:
    if lang_flag == "prohibited":
        raise ValueError("Letter contains prohibited compliance language.")
    if len(letter.split()) < 50:
        raise ValueError("Letter is too short to be a valid advice document.")

def render_advice(m: MelleaSession, profile: str) -> str:
    analysis = analyse_client_profile(m, profile=profile)
    check_preconditions(analysis)

    letter = generate_advice_letter(m, profile=profile)
    lang_flag = detect_prohibited_language(m, text=letter)
    check_postconditions(str(letter), str(lang_flag))

    return str(letter)

m = start_session()
profile = (
    "Client is 62, conservative risk tolerance, "
    "needs liquidity within 3 years, concerned about volatility."
)
try:
    print(render_advice(m, profile))
except ValueError as e:
    print(f"Validation failed: {e}")
# Output will vary — LLM responses depend on model and temperature.
```

The precondition check runs before the expensive letter generation. The
postcondition check uses a second `@generative` call as a lightweight verifier.

> **Full example:** [`docs/examples/generative_stubs/investment_advice.py`](https://github.com/generative-computing/mellea/blob/main/docs/examples/generative_stubs/investment_advice.py)

## What you built

A pattern for replacing ad-hoc `instruct()` calls with reusable, typed,
context-steerable generative functions:

| Pattern | What it gives you |
| --- | --- |
| `@generative` with `Literal` return | Constrained output, no parsing |
| `@generative` with Pydantic return | Structured output, validated schema |
| Multiple `@generative` functions | Composable pipeline in plain Python |
| `ChatContext` + `CBlock` injection | Shared persona or policy across all functions |
| Pre/postcondition checks | Input validation and output compliance |

---

**See also:** [Generative Functions](../how-to/generative-functions) | [The Requirements System](../concepts/requirements-system) | [Write Custom Verifiers](../how-to/write-custom-verifiers)
