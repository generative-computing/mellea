---
title: "Generative Functions"
description: "Define type-safe LLM functions with @generative and Pydantic structured output."
# diataxis: how-to
---

**Prerequisites:** [Quick Start](../getting-started/quickstart) complete,
`pip install mellea`, Ollama running locally.

> **Concept overview:** [Generative functions](../concepts/generative-functions) explains the design and trade-offs.

`@generative` is the idiomatic way to define type-safe LLM functions in Mellea. You
write a function signature with type hints and a docstring — Mellea generates the
implementation, calls the backend, and parses the output into the declared return type.

## Basic `@generative`

```python
from typing import Literal
from mellea import generative, start_session

@generative
def classify_sentiment(text: str) -> Literal["positive", "negative"]:
    """Classify the sentiment of the input text as 'positive' or 'negative'."""

m = start_session()
sentiment = classify_sentiment(m, text="I love this!")
print(sentiment)
# Output will vary — LLM responses depend on model and temperature.
# Expected: "positive"
```

The function body is empty (or `...`). The decorator generates a prompt from the
signature and docstring, calls the backend, and returns a value of the declared type.
The first argument is always the `MelleaSession`.

`Literal` types constrain the model to output one of the allowed values.

## Pydantic structured output

Return complex structured objects using Pydantic models:

```python
from pydantic import BaseModel
from mellea import generative, start_session

class Thought(BaseModel):
    step_name: str
    step_content: str

class ChainOfThought(BaseModel):
    chain_name: str
    step_by_step_solution: list[Thought]

@generative
def solve_step_by_step(question: str) -> ChainOfThought:
    """Generate a chain-of-thought solution for the question,
    decomposing reasoning into named, detailed steps."""

m = start_session()
response = solve_step_by_step(m, question="If I have $50 and spend $12, how much is left?")
for step in response.step_by_step_solution:
    print(f"{step.step_name}: {step.step_content}")
# Output will vary — LLM responses depend on model and temperature.
```

The model output is automatically parsed and validated against the Pydantic schema.
If parsing fails, the IVR loop retries.

## Pre- and post-conditions

Add runtime constraints with `precondition_requirements` (checked before generation)
and `requirements` (checked after). Both accept the same requirement types as
`instruct()`:

```python
from typing import Literal
from mellea import generative, start_session
from mellea.stdlib.sampling import RejectionSamplingStrategy

@generative
def classify_sentiment(text: str) -> Literal["positive", "negative", "unknown"]:
    """Classify the sentiment of the text."""

m = start_session()
result = classify_sentiment(
    m,
    text="I love this!",
    precondition_requirements=["the text argument should be fewer than 100 words"],
    requirements=["avoid classifying as unknown"],
    strategy=RejectionSamplingStrategy(),
)
print(result)
# Output will vary — LLM responses depend on model and temperature.
```

If a precondition fails, `PreconditionException` is raised immediately — the model
is never called:

```python
from mellea import generative, start_session
from mellea.core import Requirement
from mellea.stdlib.components.genstub import PreconditionException
from mellea.stdlib.requirements import simple_validate
from typing import Literal

@generative
def classify_sentiment(text: str) -> Literal["positive", "negative"]:
    """Classify the sentiment of the text."""

m = start_session()
try:
    result = classify_sentiment(
        m,
        text="I love this!",
        precondition_requirements=[
            Requirement(
                "text must be a single word",
                validation_fn=simple_validate(
                    lambda x: (len(x.split()) == 1, "Input has more than one word.")
                ),
            )
        ],
    )
except PreconditionException as e:
    print(f"Precondition failed: {e}")
    for val_result in e.validation:
        print(f"  - {val_result.reason}")
```

## Composing generative functions

Chain multiple `@generative` functions to build typed pipelines. The output of one
call becomes the input to the next:

```python
from typing import Literal
from mellea import generative, start_session

@generative
def summarize_meeting(transcript: str) -> str:
    """Summarize the key points of the meeting transcript."""

@generative
def contains_actionable_risks(summary: str) -> Literal["yes", "no"]:
    """Determine whether the summary references business risks."""

@generative
def generate_risk_mitigation(summary: str) -> str:
    """Generate risk mitigation recommendations based on the summary."""

transcript = "..."  # your meeting transcript

m = start_session()
summary = summarize_meeting(m, transcript=transcript)
if contains_actionable_risks(m, summary=summary) == "yes":
    mitigation = generate_risk_mitigation(m, summary=summary)
    print(mitigation)
# Output will vary — LLM responses depend on model and temperature.
```

Each call is an independent LLM invocation. The typed interface enforces that each
step receives and produces valid data, making pipelines easier to test and debug.

## Chain-of-thought reasoning

> **Advanced:** This section shows a performance-oriented pattern for math and
> reasoning tasks.

The Pydantic structured output pattern works well for explicit chain-of-thought (CoT)
reasoning. Separating the reasoning step from the answer extraction step can
significantly improve accuracy on tasks like GSM8K.

```python
from pydantic import BaseModel
from mellea import generative, start_session

class Thought(BaseModel):
    step_name: str
    step_content: str

class ChainOfThought(BaseModel):
    chain_name: str
    step_by_step_solution: list[Thought]

@generative
def compute_chain_of_thought(question: str) -> ChainOfThought:
    """Generate a comprehensive chain-of-thought solution for the question,
    tracking cumulative state at every step."""

@generative
def extract_final_answer(question: str, chain_of_thought: ChainOfThought) -> int:
    """Extract the final numeric answer from the chain-of-thought solution."""

m = start_session()
question = "If I have $50 and spend $12, how much is left?"
cot = compute_chain_of_thought(m, question=question)
answer = extract_final_answer(m, question=question, chain_of_thought=cot)
print(answer)
# Output will vary — LLM responses depend on model and temperature.
# Expected: 38
```

The structured `Thought` titles can be surfaced in a UI for observability into the
model's reasoning process.

---

**See also:** [Generative Functions](../concepts/generative-functions) | [Enforce Structured Output](../how-to/enforce-structured-output) | [Write Custom Verifiers](../how-to/write-custom-verifiers)
