---
title: "Write Custom Verifiers"
description: "Write validation functions that inspect LLM output and return pass/fail results with repair guidance."
# diataxis: how-to
---

# Write Custom Verifiers

**Prerequisites:** [The Requirements System](../concepts/requirements-system.md),
[Quick Start](../getting-started/quickstart.md) complete, `pip install mellea`.

Custom verifiers are Python functions that inspect LLM output and return a
`ValidationResult`. Mellea calls them as part of the IVR loop: when a verifier
returns `False`, Mellea sends the `reason` back to the model and retries.

## The `simple_validate` shortcut

For checks that only need the most recent output string, use `simple_validate`:

```python
from mellea.stdlib.requirements import simple_validate

# Boolean return: no repair guidance
is_lowercase = simple_validate(lambda x: x.lower() == x)

# Tuple return: failure reason helps the model repair
within_100_words = simple_validate(
    lambda x: (
        len(x.split()) <= 100,
        f"Output is {len(x.split())} words; must be 100 or fewer.",
    )
)
```

Use `simple_validate` when your logic only needs the output text and has no
side effects. For anything beyond that — JSON parsing with error details,
external API calls, access to conversation history — write a full validation
function.

## Writing a full validation function

A validation function receives the `Context` object and returns a
`ValidationResult`. The most common pattern is to inspect the last model output:

```python
import re
from mellea.core import Context, ValidationResult


def validate_email_format(ctx: Context) -> ValidationResult:
    """Check that the output is a valid email address."""
    output = ctx.last_output()
    text = output.value.strip() if output and output.value else ""

    email_pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    if re.match(email_pattern, text):
        return ValidationResult(True)
    return ValidationResult(
        False,
        reason=f"'{text}' is not a valid email address. Respond with only a single email address.",
    )
```

Attach it to a `Requirement`:

```python
from mellea import start_session
from mellea.core import Requirement

from .validators import validate_email_format

m = start_session()
result = m.instruct(
    "Extract the email address from: {{text}}",
    requirements=[Requirement("Must be a valid email address.", validation_fn=validate_email_format)],
    user_variables={"text": "Contact Alice at alice@example.com for details."},
)
print(str(result))
```

## Common validation patterns

### JSON validity

```python
import json
from mellea.core import Context, ValidationResult


def validate_json(ctx: Context) -> ValidationResult:
    output = ctx.last_output()
    text = output.value if output and output.value else ""
    try:
        json.loads(text)
        return ValidationResult(True)
    except json.JSONDecodeError as exc:
        return ValidationResult(
            False,
            reason=f"Output is not valid JSON. Error at position {exc.pos}: {exc.msg}. "
                   "Respond with only valid JSON, no surrounding text.",
        )
```

### Pydantic schema conformance

```python
from pydantic import BaseModel, ValidationError
from mellea.core import Context, ValidationResult


class PersonInfo(BaseModel):
    name: str
    age: int
    email: str


def validate_person_schema(ctx: Context) -> ValidationResult:
    output = ctx.last_output()
    text = output.value if output and output.value else ""
    try:
        PersonInfo.model_validate_json(text)
        return ValidationResult(True)
    except ValidationError as exc:
        errors = "; ".join(f"{e['loc']}: {e['msg']}" for e in exc.errors())
        return ValidationResult(
            False,
            reason=f"JSON does not match the required schema. Errors: {errors}. "
                   "Respond with JSON matching {name: str, age: int, email: str}.",
        )
```

### Regex patterns

```python
import re
from mellea.core import Context, ValidationResult


def validate_iso_date(ctx: Context) -> ValidationResult:
    output = ctx.last_output()
    text = output.value.strip() if output and output.value else ""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return ValidationResult(True)
    return ValidationResult(
        False,
        reason=f"'{text}' is not in ISO 8601 date format (YYYY-MM-DD). "
               "Respond with only the date in YYYY-MM-DD format.",
    )
```

### External API or database check

Validation functions are synchronous. For checks that call external systems,
make the call inline:

```python
import requests
from mellea.core import Context, ValidationResult


def validate_url_reachable(ctx: Context) -> ValidationResult:
    output = ctx.last_output()
    url = output.value.strip() if output and output.value else ""
    try:
        response = requests.head(url, timeout=5, allow_redirects=True)
        if response.status_code < 400:
            return ValidationResult(True)
        return ValidationResult(
            False,
            reason=f"URL '{url}' returned HTTP {response.status_code}. Provide a reachable URL.",
        )
    except requests.RequestException as exc:
        return ValidationResult(
            False,
            reason=f"Could not reach '{url}': {exc}. Provide a valid, reachable URL.",
        )
```

> **Note:** External calls in validators add latency to every validation attempt.
> Keep them fast and idempotent — the validator may be called multiple times
> per `instruct()` call if the IVR loop retries.

### Using `ValidationResult.score`

Some validators produce a numeric confidence score rather than a binary result.
Include it for observability and to support scoring-based sampling strategies:

```python
from mellea.core import Context, ValidationResult


def validate_length_score(ctx: Context) -> ValidationResult:
    """Pass if under 100 words; score reflects how far under the limit."""
    output = ctx.last_output()
    text = output.value if output and output.value else ""
    word_count = len(text.split())
    if word_count <= 100:
        score = 1.0 - (word_count / 100)  # 1.0 = empty, 0.0 = exactly at limit
        return ValidationResult(True, score=score)
    return ValidationResult(
        False,
        score=0.0,
        reason=f"Output is {word_count} words; must be 100 or fewer.",
    )
```

## Composing multiple verifiers

Mix `simple_validate` and full validation functions freely in a requirements list:

```python
from mellea import start_session
from mellea.core import Requirement
from mellea.stdlib.requirements import req, check, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

m = start_session()
result = m.instruct(
    "Extract the email address from: {{text}}",
    requirements=[
        req(
            "Must be a valid email address.",
            validation_fn=validate_email_format,        # full validator
        ),
        req(
            "Must not include any surrounding text or explanation.",
            validation_fn=simple_validate(              # simple_validate shortcut
                lambda x: "@" in x and " " not in x.strip()
            ),
        ),
        check("Do not include quotes around the email."),  # LLM-as-a-judge, check-only
    ],
    strategy=RejectionSamplingStrategy(loop_budget=3),
    user_variables={"text": "Reach out to support@example.com for help."},
)
print(str(result))
```

All requirements are evaluated after each generation attempt. Mellea collects every
failure and includes all failure `reason` strings in the repair request, so the model
can address multiple issues in a single pass.

## Debugging verifier failures

Use `return_sampling_results=True` to inspect which requirements failed and why:

```python
from mellea import start_session
from mellea.core import Requirement
from mellea.stdlib.sampling import RejectionSamplingStrategy

m = start_session()
result = m.instruct(
    "Extract the email address from: {{text}}",
    requirements=[
        Requirement("Must be a valid email address.", validation_fn=validate_email_format),
    ],
    strategy=RejectionSamplingStrategy(loop_budget=3),
    user_variables={"text": "Contact us at support@example.com."},
    return_sampling_results=True,
)

print(f"Success: {result.success}")
for attempt_idx, validations in enumerate(result.sample_validations):
    print(f"Attempt {attempt_idx + 1}:")
    for requirement, val_result in validations:
        status = "PASS" if val_result else "FAIL"
        print(f"  [{status}] {requirement.description}: {val_result.reason}")
```

This pattern is useful during development to confirm your verifier fires at the
right time and produces helpful repair guidance.

---

**Previous:** [Enforce Structured Output](./enforce-structured-output.md) |
**Next:** [Configure model options](./configure-model-options.md)

**See also:** [The Requirements System](../concepts/requirements-system.md) |
[Instruct, Validate, Repair](../concepts/instruct-validate-repair.md)
