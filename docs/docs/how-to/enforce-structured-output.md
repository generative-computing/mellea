---
title: "Enforce Structured Output"
description: "Get JSON, Pydantic models, and typed values from LLM calls using @generative and instruct(format=...)."
# diataxis: how-to
---

**Prerequisites:** [Quick Start](../getting-started/quickstart) complete,
`pip install mellea`, Ollama running locally.

Mellea provides two paths to structured output. Choose based on how the call fits
into your code:

| Pattern | When to use |
| ------- | ----------- |
| `@generative` with return type | You want a named, reusable function. The return type is declared in the signature. |
| `instruct(format=...)` | You are building the prompt dynamically or combining structured output with `grounding_context` or `user_variables`. |

Both paths enforce the declared schema at generation time using constrained decoding
where the backend supports it, and retry with the IVR loop if parsing fails.

## Pattern 1: `@generative` with typed returns

### Classification with `Literal`

```python
from typing import Literal
from mellea import generative, start_session

@generative
def classify_priority(issue: str) -> Literal["critical", "high", "medium", "low"]:
    """Classify the priority level of a support issue."""

m = start_session()
priority = classify_priority(m, issue="Production database is unreachable.")
print(priority)
# Output will vary — LLM responses depend on model and temperature.
# Expected: "critical"
```

The model is constrained to return exactly one of the four allowed values.

### Simple Pydantic extraction

```python
from pydantic import BaseModel
from mellea import generative, start_session

class PersonInfo(BaseModel):
    name: str
    role: str
    department: str

@generative
def extract_person(bio: str) -> PersonInfo:
    """Extract the person's name, role, and department from their biography."""

m = start_session()
bio = "Sarah Chen joined the engineering team in 2021 as a senior backend developer."
person = extract_person(m, bio=bio)
print(person.name, person.role)
# Output will vary — LLM responses depend on model and temperature.
```

### List returns

Return a list of typed values or Pydantic models:

```python
from mellea import generative, start_session

@generative
def extract_person_names(doc: str) -> list[str]:
    """Extract the names of all people mentioned in the document."""

m = start_session()
names = extract_person_names(
    m,
    doc="The report was co-authored by Alice Johnson and Bob Lee.",
)
print(names)
# Output will vary — LLM responses depend on model and temperature.
# Expected: ["Alice Johnson", "Bob Lee"]
```

### Nested models

Complex structured extraction works naturally with nested Pydantic models:

```python
from pydantic import BaseModel
from mellea import generative, start_session

class Address(BaseModel):
    street: str
    city: str
    country: str

class Company(BaseModel):
    name: str
    industry: str
    headquarters: Address

@generative
def extract_company(text: str) -> Company:
    """Extract company details from the text."""

m = start_session()
company = extract_company(
    m,
    text="Acme Corp is a manufacturing company headquartered at 123 Main St, Springfield, USA.",
)
print(company.headquarters.city)
# Output will vary — LLM responses depend on model and temperature.
```

## Pattern 2: `instruct(format=...)`

When you need structured output alongside dynamic prompts, grounding context, or
user variables, use the `format` parameter on `instruct()`:

```python
from pydantic import BaseModel
from mellea import start_session
from mellea.stdlib.requirements import check, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy


class NameResponse(BaseModel):
    names: list[str]


m = start_session()
result = m.instruct(
    "Extract ALL person names from the document (doc1).",
    grounding_context={
        "doc1": (
            "Leaders banded together to press Germany to back pro-growth policies. "
            "President Obama gained support for his argument that Europe cannot "
            "afford Chancellor Merkel's austerity approach."
        )
    },
    format=NameResponse,
)

parsed = NameResponse.model_validate_json(str(result))
print(parsed.names)
# Output will vary — LLM responses depend on model and temperature.
# Expected: ["President Obama", "Angela Merkel"]
```

The `format` parameter triggers constrained decoding. The result is a
`ModelOutputThunk` whose `.value` is a JSON string matching the schema. Parse it
with `PydanticModel.model_validate_json(str(result))`.

## Validating structured output content

Constrained decoding enforces schema validity — the output is always parseable JSON
matching your model. To enforce semantic constraints (e.g., "the list must contain at
least 2 names"), combine `format` with a custom validation function:

```python
from collections.abc import Callable
from pydantic import BaseModel, ValidationError
from mellea import start_session
from mellea.stdlib.requirements import check, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy


class NameResponse(BaseModel):
    names: list[str]


def at_least_n_names(n: int) -> Callable[[str], tuple[bool, str]]:
    """Factory: returns a validator that checks the names list has >= n entries."""
    def _validate(text: str) -> tuple[bool, str]:
        try:
            parsed = NameResponse.model_validate_json(text)
        except ValidationError:
            return (False, "Output is not valid JSON matching the NameResponse schema.")
        if len(parsed.names) >= n:
            return (True, "")
        return (False, f"Found {len(parsed.names)} name(s); expected at least {n}.")
    return _validate


m = start_session()
result = m.instruct(
    "Extract ALL person names from the document (doc1).",
    grounding_context={"doc1": "...your document text..."},
    requirements=[
        check(
            None,
            validation_fn=simple_validate(at_least_n_names(2)),
        )
    ],
    strategy=RejectionSamplingStrategy(loop_budget=5),
    format=NameResponse,
    return_sampling_results=True,
)

if result.success:
    names = NameResponse.model_validate_json(str(result.result)).names
    print(names)
else:
    print("Could not extract the required names after retries.")
```

The `check(None, ...)` idiom creates a validation-only requirement that is never
embedded in the prompt. This avoids biasing the model while still gating the output
on your semantic constraint.

## Requirements on `@generative` output

You can also apply requirements to `@generative` output. When the return type is a
Pydantic model, the requirements operate on the JSON string representation:

```python
from pydantic import BaseModel
from mellea import generative, start_session
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import RejectionSamplingStrategy

class Summary(BaseModel):
    title: str
    bullets: list[str]

@generative
def summarize(text: str) -> Summary:
    """Summarize the text as a titled bullet list."""

m = start_session()
summary = summarize(
    m,
    text="...",
    requirements=[req("Include at least 3 bullet points.")],
    strategy=RejectionSamplingStrategy(loop_budget=3),
)
# summary is already a Summary instance — no manual parsing needed
print(summary.title)
for bullet in summary.bullets:
    print(f"  - {bullet}")
# Output will vary — LLM responses depend on model and temperature.
```

With `@generative`, the output is parsed into the Pydantic model automatically.
You receive a `Summary` instance, not a JSON string.

## Choosing between the two patterns

**Use `@generative`** when:

- The function is reusable and called from multiple places.
- The input and output types are stable.
- You want a clean function signature with IDE type-checking.
- You prefer direct attribute access (`person.name`) over manual JSON parsing.

**Use `instruct(format=...)`** when:

- The prompt is built dynamically with `user_variables` or `grounding_context`.
- You are retrofitting structured output onto an existing `instruct()` call.
- You need fine-grained control over requirements and sampling alongside formatting.

Both patterns support the full IVR loop, requirements, sampling strategies, and
`SamplingResult` inspection.

---


**See also:** [Generative Functions](../guide/generative-functions) |
[The Requirements System](../concepts/requirements-system)
