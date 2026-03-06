---
title: "Handling Exceptions and Failures"
description: "Handle SamplingResult failures, PreconditionException, and parse errors gracefully in Mellea programs."
# diataxis: how-to
---

# Handling Exceptions and Failures

**Prerequisites:** [The Requirements System](../concepts/requirements-system.md),
[Quick Start](../getting-started/quickstart.md) complete, `pip install mellea`.

Mellea programs encounter two categories of failure: **expected failures** (IVR
exhaustion, precondition violations) that are part of normal operation, and
**unexpected errors** (backend connectivity, parse failures) that indicate
configuration or implementation problems.

## Expected failures

### IVR loop exhaustion: `SamplingResult.success = False`

When `instruct()` is called with `return_sampling_results=True` and the IVR loop
exhausts its budget without satisfying all requirements, `SamplingResult.success` is
`False`. This is not a Python exception — it is a normal return value that your code
should handle.

```python
from mellea import start_session
from mellea.stdlib.requirements import req, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

m = start_session()
result = m.instruct(
    "Write a haiku about the ocean.",
    requirements=[
        req(
            "Must have exactly 17 syllables (5-7-5).",
            validation_fn=simple_validate(
                lambda x: (
                    len(x.split()) <= 20,  # rough proxy; replace with a real syllable counter
                    "Syllable count does not match the 5-7-5 pattern.",
                )
            ),
        ),
    ],
    strategy=RejectionSamplingStrategy(loop_budget=5),
    return_sampling_results=True,
)

if result.success:
    print(str(result.result))
else:
    # All attempts failed — decide what to do
    print("Could not generate a valid haiku after 5 attempts.")
    print("Best attempt:", str(result.sample_generations[0].value))
```

Common fallback patterns when `success` is `False`:

- **Use the best attempt anyway** — `result.sample_generations[0].value` gives the
  first (often the best) generation, even if requirements were not fully satisfied.
- **Lower the bar** — retry with reduced requirements or a higher `loop_budget`.
- **Return an error indicator** — tell the caller the operation could not be
  completed to spec, and let it decide.
- **Log and alert** — if this should rarely fail, log the attempts and notify.

### Inspecting failure reasons

`SamplingResult.sample_validations` gives per-attempt validation details. Use them
to understand which requirements are failing and why:

```python
if not result.success:
    for attempt_idx, validations in enumerate(result.sample_validations):
        print(f"Attempt {attempt_idx + 1}:")
        for requirement, val_result in validations:
            if not val_result:
                print(f"  FAIL: {requirement.description}")
                print(f"    Reason: {val_result.reason}")
```

A requirement that fails on every attempt usually indicates one of:

- The model cannot satisfy this constraint with the current prompt and model.
- The `validation_fn` has a bug (returns `False` unconditionally or has a logic error).
- The requirement is genuinely contradictory with the instruction.

### Precondition failures: `PreconditionException`

When `precondition_requirements` are attached to a `@generative` call, Mellea
validates the inputs before calling the model. If any precondition fails,
`PreconditionException` is raised immediately — no model call is made:

```python
from typing import Literal
from mellea import generative, start_session
from mellea.core import Requirement
from mellea.stdlib.components.genslot import PreconditionException
from mellea.stdlib.requirements import simple_validate


@generative
def classify_sentiment(text: str) -> Literal["positive", "negative", "neutral"]:
    """Classify the sentiment of the text."""


m = start_session()

try:
    result = classify_sentiment(
        m,
        text="I love this!",
        precondition_requirements=[
            Requirement(
                "Input must be fewer than 500 characters.",
                validation_fn=simple_validate(
                    lambda x: (
                        len(x) < 500,
                        f"Input is {len(x)} characters; must be under 500.",
                    )
                ),
            )
        ],
    )
    print(result)
except PreconditionException as e:
    print(f"Invalid input: {e}")
    for val_result in e.validation:
        print(f"  - {val_result.reason}")
    # Handle gracefully: sanitize input, reject the request, etc.
```

`PreconditionException.validation` is a list of `ValidationResult` objects for the
requirements that failed. Each `.reason` field explains what was wrong.

Use preconditions to:

- Validate untrusted inputs before they reach the model
- Enforce interface contracts between pipeline stages
- Fail fast on inputs that are guaranteed to produce bad output

## Unexpected errors

### Backend connection errors

If Ollama is not running, or a cloud API key is invalid, the backend raises an
exception on the first model call:

```python
import mellea

try:
    m = mellea.start_session()
    result = m.instruct("Hello.")
    print(str(result))
except Exception as e:
    # Backend errors are not Mellea-specific exceptions — they come from the
    # underlying HTTP client or the backend constructor.
    print(f"Backend error: {e}")
    # Handle: check connectivity, validate credentials, fall back to another backend
```

For production code, wrap session creation and the first call together:

```python
import mellea

def create_session_or_none():
    try:
        m = mellea.start_session()
        # Probe the connection with a cheap call
        m.chat("ping")
        return m
    except Exception as e:
        print(f"Could not connect to backend: {e}")
        return None
```

### Parse failures: `ComponentParseError`

When `@generative` or `instruct(format=...)` is used with a Pydantic model or
`Literal` return type, Mellea parses the raw model output into the declared type.
If parsing fails, a `ComponentParseError` is raised.

This typically means the model produced output that does not conform to the schema.
The IVR loop retries on parse failure automatically — `ComponentParseError` surfaces
only if all retries are exhausted.

```python
from typing import Literal
from mellea import generative, start_session
from mellea.core.base import ComponentParseError


@generative
def classify(text: str) -> Literal["a", "b", "c"]:
    """Classify the text into category a, b, or c."""


m = start_session()

try:
    result = classify(m, text="...")
except ComponentParseError as e:
    print(f"Model output could not be parsed: {e}")
    # Fall back to a raw string extraction or a default value
```

If `ComponentParseError` occurs in practice, check:

- Whether the model is large enough to follow the output format instructions.
- Whether the instruction and docstring are clear about the expected format.
- Whether the backend supports constrained decoding for the return type.

## Fallback and retry patterns

### Fallback to a simpler call

If a structured call fails, fall back to a plain `instruct()`:

```python
from pydantic import BaseModel
from mellea import generative, start_session
from mellea.core.base import ComponentParseError

class ExtractedData(BaseModel):
    name: str
    email: str

@generative
def extract(text: str) -> ExtractedData:
    """Extract name and email from the text."""

m = start_session()
try:
    data = extract(m, text="Contact Alice at alice@example.com.")
    print(data.name, data.email)
except ComponentParseError:
    # Fall back: get the raw text and parse manually
    raw = m.instruct("Extract the name and email from: {{text}}",
                     user_variables={"text": "Contact Alice at alice@example.com."})
    print("Raw fallback:", str(raw))
```

### Fallback to a different model

For calls that require higher capability, escalate to a stronger model on failure:

```python
from mellea import MelleaSession
from mellea.backends.ollama import OllamaModelBackend
from mellea.backends import model_ids
from mellea.stdlib.sampling import RejectionSamplingStrategy

def instruct_with_fallback(text: str) -> str:
    m_fast = MelleaSession(OllamaModelBackend(model_ids.IBM_GRANITE_4_MICRO_3B))
    result = m_fast.instruct(
        text,
        strategy=RejectionSamplingStrategy(loop_budget=3),
        return_sampling_results=True,
    )
    if result.success:
        return str(result.result)

    # Escalate to a larger model
    m_strong = MelleaSession(OllamaModelBackend(model_ids.IBM_GRANITE_3_3_8B))
    return str(m_strong.instruct(text))
```

This is the basis of the SOFAI (System 1 / System 2) pattern — fast model first,
strong model only when needed. Mellea provides `SOFAISamplingStrategy` as a
built-in implementation. See [Inference-Time Scaling](../advanced/inference-time-scaling.md).

## Logging failures

Use Python's standard `logging` module to record failures alongside generation
details:

```python
import logging
from mellea import start_session
from mellea.stdlib.sampling import RejectionSamplingStrategy

logger = logging.getLogger(__name__)

m = start_session()
result = m.instruct(
    "Classify: {{text}}",
    strategy=RejectionSamplingStrategy(loop_budget=3),
    user_variables={"text": "..."},
    return_sampling_results=True,
)

if not result.success:
    logger.warning(
        "instruct() failed after %d attempts",
        len(result.sample_generations),
        extra={
            "attempts": len(result.sample_generations),
            "first_output": str(result.sample_generations[0].value),
        },
    )
```

For structured telemetry across all calls, see
[Metrics and Telemetry](./metrics-and-telemetry.md).

---


**See also:** [The Requirements System](../concepts/requirements-system.md) |
[Write Custom Verifiers](../how-to/write-custom-verifiers.md)
