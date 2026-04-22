---
title: "Evaluate with LLM-as-a-Judge"
description: "Use the LLM itself to evaluate output quality — inline as a requirement, or as a standalone validation pass."
# diataxis: how-to
---

**Prerequisites:** [The Requirements System](../concepts/requirements-system),
[Quick Start](../getting-started/quickstart) complete, `pip install mellea`.

LLM-as-a-judge (LLMaJ) uses a second model call to evaluate whether a generated
output meets a criterion expressed in natural language. In Mellea this is the
default validation strategy for [`req()`](../reference/glossary#requirement) — you describe what good output looks
like, and Mellea asks the model whether the output satisfies that description.

## How it works

When a [`Requirement`](../reference/glossary#requirement) has no `validation_fn`, Mellea runs a separate LLM call
after generation. The requirement's `description` and the model output are
formatted into a judge prompt, and the model returns a verdict. Mellea converts
the verdict to `True` / `False` by looking for `"yes"` (case-insensitive) in the
response.

```python
from mellea import start_session
from mellea.stdlib.requirements import req

m = start_session()

quality_check = req("The response must be under 30 words and include a concrete example.")

result = m.instruct(
    "Explain what a context manager is in Python.",
    requirements=[quality_check],
)

print(str(result))
# Output will vary — LLM responses depend on model and temperature.
```

If the output fails the requirement, Mellea retries (up to the `loop_budget`
limit) and feeds the failure reason back into the next attempt.

## Standalone validation with m.validate()

Run requirements against an existing output without triggering a new generation:

```python
from mellea import start_session
from mellea.stdlib.requirements import req

m = start_session()
result = m.instruct("Describe three benefits of TypeScript.")

completeness = req("The response must mention at least three distinct benefits.")
conciseness = req("The response must be under 100 words.")

validation_results = m.validate([completeness, conciseness])

for r, vr in zip([completeness, conciseness], validation_results):
    status = "PASS" if vr.result else "FAIL"
    print(f"{status}: {r.description}")
    if not vr.result:
        print(f"  Reason: {vr.reason}")
```

`m.validate()` returns a list of [`ValidationResult`](../reference/glossary#validationresult) objects, one per requirement.

## Capture judge reasoning with generate_logs

To inspect the full judge prompt and verdict, pass a [`GenerateLog`](../reference/glossary#generatelog) list:

```python
from mellea import start_session
from mellea.core import GenerateLog
from mellea.stdlib.requirements import req

logs: list[GenerateLog] = []

m = start_session()
result = m.instruct("Write a haiku about software bugs.")
print(str(result))
# Output will vary — LLM responses depend on model and temperature.

m.validate(
    [req("Must follow the 5-7-5 syllable structure.")],
    generate_logs=logs,
)

for log in logs:
    if isinstance(log, GenerateLog):
        print("Judge prompt:", log.prompt)
        print("Judge verdict:", log.result.value if log.result else None)
```

`GenerateLog` captures the prompt sent to the judge model and the raw verdict
string, which is useful for debugging requirements that are failing unexpectedly.

## Avoid the purple elephant effect with check()

Including a requirement description in the generation prompt can cause the model
to fixate on the thing you want to avoid — the [purple elephant effect](../reference/glossary#purple-elephant-effect). Use
[`check()`](../reference/glossary#requirement) to validate without including the description in the generation prompt:

```python
from mellea import start_session
from mellea.stdlib.requirements import req, check

m = start_session()
result = m.instruct(
    "Write a product description for noise-cancelling headphones.",
    requirements=[
        req("Mention battery life and comfort."),           # included in prompt
        check("Must not contain the phrase 'industry-leading'"),  # checked silently
    ],
)
print(str(result))
# Output will vary — LLM responses depend on model and temperature.
```

`req()` shapes what the model aims for. `check()` enforces a constraint the model
should satisfy naturally — without being told about it.

## Replace LLMaJ with a fast programmatic check

For deterministic criteria (length, format, regex), use `simple_validate` to
bypass the LLM judge entirely:

```python
from mellea import start_session
from mellea.stdlib.requirements import req, simple_validate

m = start_session()
word_count_check = req(
    "Response must be between 20 and 60 words.",
    validation_fn=simple_validate(lambda text: 20 <= len(text.split()) <= 60),
)

result = m.instruct(
    "Explain what a Python decorator does.",
    requirements=[word_count_check],
)
print(str(result))
# Output will vary — LLM responses depend on model and temperature.
```

`simple_validate` wraps a function that receives the model output as a string and
returns `bool` (or a `(bool, reason)` tuple). No LLM call is made for validation.

## Combine LLMaJ and programmatic checks

Use both in the same `requirements` list:

```python
import re
from mellea import start_session
from mellea.stdlib.requirements import req, simple_validate

m = start_session()
result = m.instruct(
    "Generate a UK postcode for central London.",
    requirements=[
        req("Must be a valid central London postcode."),
        req(
            "Must match UK postcode format.",
            validation_fn=simple_validate(
                lambda text: bool(re.fullmatch(r"[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}", text.strip())),
                reason="Output did not match postcode format",
            ),
        ),
    ],
)
print(str(result))
# Output will vary — LLM responses depend on model and temperature.
```

The first `req()` steers the model toward a valid postcode. The second uses
`simple_validate` to enforce the regex — cheaply, without a second LLM call.

## Return validation metadata with SamplingResult

To access the full validation outcome alongside the generated output, use
`return_sampling_results=True`:

```python
from mellea import start_session
from mellea.stdlib.requirements import req

m = start_session()
output = m.instruct(
    "Write a one-sentence definition of recursion.",
    requirements=[req("Must be accurate and under 20 words.")],
    return_sampling_results=True,
)

print(f"Output: {output.result}")
print(f"Passed: {output.success}")
print(f"Attempts: {len(output.sample_generations)}")
```

[`SamplingResult`](../reference/glossary#samplingresult)`.success` is `True` if at least one attempt satisfied all
requirements. `sample_generations` lists every attempt made.

**See also:** [The Requirements System](../concepts/requirements-system) |
[Write Custom Verifiers](../how-to/write-custom-verifiers) |
[Handling Exceptions and Failures](../how-to/handling-exceptions) |
[CLI Reference](../reference/cli)
