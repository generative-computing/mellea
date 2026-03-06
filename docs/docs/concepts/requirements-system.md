---
title: "The Requirements System"
description: "How Requirement, ValidationResult, and the IVR loop work together to enforce constraints on generative output."
# diataxis: explanation
---

# The Requirements System

Requirements are Mellea's mechanism for enforcing constraints on generative output.
They serve two roles simultaneously: they appear in the prompt so the model knows what
to aim for, and they are evaluated after generation so Mellea can detect and repair
failures automatically.

This page explains the requirements system in depth. For a quick introduction,
see [The Instruction Model](./instruct-validate-repair.md).

## What a requirement is

A `Requirement` is a `Component` that wraps a natural-language description and an
optional validation function. During the instruct–validate–repair (IVR) loop:

1. Mellea renders the requirement descriptions into the prompt alongside the instruction.
2. After the model generates output, each requirement is validated against that output.
3. If any requirement fails, Mellea sends the model a repair request, listing which
   requirements failed and why.
4. The loop retries up to `loop_budget` times (default: 2).

```python
from mellea.core import Requirement

# Simplest form: natural-language string.
# Mellea uses LLM-as-a-judge to check it.
r = Requirement("The email should have a salutation.")
```

Passing plain strings directly to `instruct()` is equivalent — they are
converted to `Requirement` objects internally:

```python
import mellea

m = mellea.start_session()
email = m.instruct(
    "Write an email inviting the team to a meeting.",
    requirements=["The email should have a salutation.", "Fewer than 150 words."],
)
```

## `req()` and `check()` shorthands

`req()` and `check()` are concise constructors from `mellea.stdlib.requirements`:

```python
from mellea.stdlib.requirements import check, req

# req() creates a standard Requirement (description included in the prompt)
r1 = req("The email should have a salutation.")

# check() creates a check-only Requirement (description NOT included in the prompt)
r2 = check("Do not mention purple elephants.")
```

The difference matters: when `check_only=True`, the requirement description is
evaluated after generation but **not** embedded in the prompt. This avoids the
[purple elephant effect](https://generative-computing.github.io/blog/) — where
mentioning something in a negative instruction (e.g., "do not mention purple
elephants") paradoxically increases the chance the model produces it.

Use `req()` for positive constraints you want the model to aim for. Use `check()` for
negative or hard-to-explain constraints that are better left out of the prompt.

## Custom validation functions

For deterministic checks, attach a `validation_fn`. Mellea skips LLM-as-a-judge and
runs your function directly:

```python
from mellea import start_session
from mellea.core import Requirement
from mellea.stdlib.requirements import simple_validate

word_limit = Requirement(
    "Fewer than 100 words.",
    validation_fn=simple_validate(lambda output: len(output.split()) < 100),
)

m = start_session()
email = m.instruct(
    "Write an email to {{name}}.",
    requirements=[word_limit],
    user_variables={"name": "Olivia"},
)
```

`simple_validate` is a convenience wrapper. It accepts a function that receives the
most recent model output as a string and returns either:

- `bool` — pass or fail; no reason is captured
- `tuple[bool, str]` — pass/fail plus a reason string that Mellea includes in the
  repair request

```python
from mellea.stdlib.requirements import simple_validate

# Boolean return
is_lowercase = simple_validate(lambda x: x.lower() == x)

# Tuple return — the reason is sent to the model on failure
within_limit = simple_validate(
    lambda x: (len(x.split()) < 100, f"Output is {len(x.split())} words; must be < 100.")
)
```

## `ValidationResult` in depth

`simple_validate` produces `ValidationResult` objects automatically. When you write
a full validation function directly, you construct `ValidationResult` yourself:

```python
from mellea.core import Context, ValidationResult


def validate_json(ctx: Context) -> ValidationResult:
    """Accept output only if it is valid JSON."""
    import json

    output = ctx.last_output()
    text = output.value if output is not None else ""
    try:
        json.loads(text)
        return ValidationResult(True)
    except json.JSONDecodeError as exc:
        return ValidationResult(False, reason=f"Invalid JSON: {exc}")
```

The `validation_fn` signature is `Callable[[Context], ValidationResult]`. The
`Context` object gives you access to the full session state if needed — not just the
last output.

`ValidationResult` fields:

| Field | Type | Description |
| ----- | ---- | ----------- |
| `result` | `bool` | Whether the requirement passed. |
| `reason` | `str \| None` | Human-readable explanation, included in repair requests. |
| `score` | `float \| None` | Optional numeric score from your validator. |
| `thunk` | `ModelOutputThunk \| None` | The model output used, if your validator ran a backend call. |
| `context` | `Context \| None` | The context snapshot at validation time. |

The `reason` field is the most useful in practice — a clear reason string helps the
model make a targeted repair rather than regenerating blindly.

## Preconditions in generative functions

The `@generative` decorator supports `precondition_requirements` alongside the
standard `requirements`. Preconditions are validated against the *inputs* to the
function before generation starts. If they fail, Mellea raises `PreconditionException`
immediately — no generation attempt is made and no IVR loop runs.

```python
from typing import Literal

from mellea import generative, start_session
from mellea.core import Requirement
from mellea.stdlib.components.genslot import PreconditionException
from mellea.stdlib.requirements import simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy


@generative
def classify_sentiment(text: str) -> Literal["positive", "negative", "neutral"]:
    """Classify the sentiment of the text."""


m = start_session()

# Precondition: validate inputs before the model is called
try:
    result = classify_sentiment(
        m,
        text="I love this!",
        precondition_requirements=[
            Requirement(
                "Input must be fewer than 200 characters.",
                validation_fn=simple_validate(lambda x: len(x) < 200),
            )
        ],
        requirements=["Avoid returning 'neutral' unless the sentiment is genuinely ambiguous."],
        strategy=RejectionSamplingStrategy(),
    )
    print(result)
except PreconditionException as e:
    print(f"Precondition failed: {e}")
    for val in e.validation:
        print(f"  - {val.reason}")
```

`PreconditionException.validation` is a list of `ValidationResult` objects for every
requirement that failed, giving you a complete picture of what went wrong.

> **Note:** `precondition_requirements` require a strategy to be specified (e.g.,
> `RejectionSamplingStrategy()`). Without a strategy the precondition check is skipped
> with a warning.

## Inspecting validation results

When you use `return_sampling_results=True`, `instruct()` returns a `SamplingResult`
instead of a `ModelOutputThunk`. This exposes per-attempt validation results:

```python
from mellea import start_session
from mellea.stdlib.requirements import req, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

m = start_session()
result = m.instruct(
    "Write a short note to {{name}}.",
    requirements=[
        req(
            "Use only lower-case letters.",
            validation_fn=simple_validate(
                lambda x: (x.lower() == x, "Output contains upper-case characters.")
            ),
        ),
    ],
    strategy=RejectionSamplingStrategy(loop_budget=3),
    user_variables={"name": "Olivia"},
    return_sampling_results=True,
)

if result.success:
    print(str(result.result))
else:
    # Inspect why each attempt failed
    for attempt_idx, attempt_validations in enumerate(result.sample_validations):
        print(f"Attempt {attempt_idx + 1}:")
        for requirement, val_result in attempt_validations:
            status = "PASS" if val_result else "FAIL"
            print(f"  [{status}] {requirement.description}: {val_result.reason}")
```

`SamplingResult.sample_validations` is a list of attempts, each containing a list
of `(Requirement, ValidationResult)` tuples. `SamplingResult.result_validations`
gives you the same for the final selected output only.

## LLM-as-a-judge vs custom validators

| Approach | When to use |
| -------- | ----------- |
| Plain string requirement | Subjective or hard-to-code constraints ("be polite", "stay on topic"). |
| `simple_validate(lambda ...)` | Simple deterministic checks (length, regex, JSON parse). |
| Full `validation_fn` | Multi-step logic, external API calls, or access to session context. |
| `ALoraRequirement` | Fine-tuned constraint LoRA — fastest at scale, requires adapter. |

LLM-as-a-judge requirements call the backend for each validation, which adds latency.
For high-throughput workloads, prefer `simple_validate` for deterministic checks and
reserve LLM-based requirements for subjective criteria that cannot be coded directly.

> **Advanced:** `ALoraRequirement` (from `mellea.stdlib.requirements`) uses a fine-tuned
> LoRA adapter for validation instead of LLM-as-a-judge. It falls back to LLM-as-a-judge
> if the adapter is unavailable. See [LoRA and aLoRA Adapters](../advanced/lora-and-alora-adapters.md).

## Composing requirements

Requirements are composable: mix strings, `req()`, `check()`, and `Requirement`
objects freely in the same list:

```python
from mellea.core import Requirement
from mellea.stdlib.requirements import check, req, simple_validate

requirements = [
    "The email should have a salutation.",          # plain string → LLM-as-a-judge
    req("Use only lower-case letters.",             # req() with custom validator
        validation_fn=simple_validate(lambda x: x.lower() == x)),
    check("Do not mention competitor products."),  # check-only → not in prompt
    Requirement(                                    # explicit Requirement object
        "Fewer than 100 words.",
        validation_fn=simple_validate(
            lambda x: (len(x.split()) < 100, f"Word count: {len(x.split())}")
        ),
    ),
]
```

All requirements are validated after each generation attempt. The repair request lists
every requirement that failed, not just the first one, so the model can address all
issues in a single repair pass.

---

**Previous:** [The Instruction Model](./instruct-validate-repair.md) |
**Next:** [Mellea vs Orchestration Frameworks](./architecture-vs-agents.md)
