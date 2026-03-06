---
title: "Unit Test Generative Code"
description: "Write reliable tests for @generative functions using pytest markers and output validation."
# diataxis: how-to
---

**Prerequisites:** [Quick Start](../getting-started/quickstart) complete,
`pip install mellea`, Ollama running locally, `pytest` installed.

Testing generative code requires you to separate concerns: some assertions are
always deterministic (the output is the right type), while others depend on model
behaviour and are inherently qualitative. This page shows you how to structure
both categories, configure the right pytest markers, and make your CI pipeline
fast and reliable.

## Three levels of assertion

Every test for a `@generative` function falls into one of three levels:

| Level | What you assert | Deterministic? |
| ----- | --------------- | -------------- |
| **Type check** | `isinstance(result, bool)` | Yes — constrained decoding always returns the declared type |
| **Structural check** | `result in ["positive", "negative"]` or field names present | Yes — schema enforcement is deterministic |
| **Qualitative check** | `assert result is True` | No — depends on the model and prompt |

Type and structural checks run in CI. Qualitative checks carry
`@pytest.mark.qualitative` and are skipped in CI when `CICD=1` is set.

## Setting up a test session fixture

Use a `backend` fixture to handle CI versus local configuration, and a
function-scoped `session` fixture to give each test a clean slate:

```python
import pytest
from mellea import MelleaSession
from mellea.backends.litellm import LiteLLMBackend
from mellea.backends.model_ids import IBM_GRANITE_4_HYBRID_MICRO

_MODEL_ID = f"ollama_chat/{IBM_GRANITE_4_HYBRID_MICRO.ollama_name}"


@pytest.fixture(scope="module")
def backend(gh_run: int):
    """LiteLLM backend pointed at a local Ollama instance."""
    if gh_run == 1:
        # In CI the Ollama host may be set explicitly via OLLAMA_HOST.
        import os

        url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        url = url.replace("127.0.0.1", "http://localhost")
        return LiteLLMBackend(
            model_id=_MODEL_ID,
            base_url=url,
            model_options={"api_base": url},
        )
    return LiteLLMBackend(model_id=_MODEL_ID)


@pytest.fixture(scope="function")
def session(backend):
    """Fresh MelleaSession for each test."""
    m = MelleaSession(backend=backend)
    yield m
    m.reset()
```

The `gh_run` fixture comes from `test/conftest.py`. It returns `1` when the
environment variable `CICD=1` is set (GitHub Actions) and `0` otherwise.

> **Note:** Scoping `backend` to `module` and `session` to `function` strikes a
> balance between setup cost and test isolation. Each test gets a clean context,
> but the backend connection is created once per module.

## Module-level markers

Declare markers at the top of your test file with `pytestmark` so they apply to
every test in the module without repetition:

```python
import pytest

pytestmark = [pytest.mark.ollama, pytest.mark.llm]
```

Use `pytest.mark.litellm` as well if the module uses `LiteLLMBackend`:

```python
pytestmark = [pytest.mark.litellm, pytest.mark.ollama, pytest.mark.llm]
```

## Testing `@generative` functions

### Type assertions — always deterministic

The return type of a `@generative` function is enforced by constrained decoding
or output parsing. An `isinstance` check never depends on model behaviour:

```python
from typing import Literal

import pytest
from mellea import generative
from mellea.stdlib.requirements import Requirement, simple_validate


@generative
def classify_sentiment(text: str) -> Literal["positive", "negative"]:
    """Classify the sentiment of the provided text."""


def test_classify_sentiment_type(session):
    result = classify_sentiment(session, text="I love this product!")
    # Type check: always passes regardless of which value the model chose.
    assert isinstance(result, str)
```

### Structural assertions — always deterministic

For `Literal` return types, membership in the allowed values is enforced before
your test sees the result. The assertion is still deterministic:

```python
def test_classify_sentiment_structure(session):
    result = classify_sentiment(session, text="I love this product!")
    assert result in ["positive", "negative"]
```

For Pydantic model return types, assert that the required fields are present and
have the right types:

```python
from pydantic import BaseModel
from mellea import generative


class Review(BaseModel):
    summary: str
    score: int
    tags: list[str]


@generative
def extract_review(raw: str) -> Review:
    """Extract a structured review from raw text."""


def test_extract_review_structure(session):
    result = extract_review(
        session,
        raw="Excellent build quality. I rate it 9 out of 10. #durable #premium",
    )
    assert isinstance(result, Review)
    assert isinstance(result.summary, str)
    assert isinstance(result.score, int)
    assert isinstance(result.tags, list)
```

### Qualitative assertions — mark and skip in CI

When you want to assert on the *content* of a response, add
`@pytest.mark.qualitative`. These tests are skipped automatically in CI
(`CICD=1`) and are intended to run locally or in a dedicated quality gate:

```python
import pytest
from mellea import generative


@generative
def is_happy(text: str) -> bool:
    """Determine if the text has a happy mood."""


@pytest.mark.qualitative
def test_is_happy_positive(session):
    result = is_happy(session, text="I'm enjoying life.")
    assert isinstance(result, bool)
    # Qualitative: the correct answer is True, but this is model-dependent.
    assert result is True


@pytest.mark.qualitative
def test_classify_sentiment_positive(session):
    result = classify_sentiment(session, text="I love this product!")
    assert result == "positive"
```

> **Warning:** Do not assert on qualitative behaviour without `@pytest.mark.qualitative`.
> A deterministic-looking assertion like `assert score > 5` can flake across
> model versions, temperatures, and quantisation levels.

## Testing `instruct()` calls

`instruct()` calls are non-qualitative when you test structure, not content.
Assert that the call returns a value and that the value has the right type:

```python
from mellea.stdlib.sampling import RejectionSamplingStrategy


def test_instruct_returns_string(session):
    res = session.instruct(
        "Write an email to the interns.",
        requirements=["be funny"],
        strategy=RejectionSamplingStrategy(loop_budget=3),
    )
    assert res is not None
    assert isinstance(res.value, str)
```

### Inspecting logged model options

`_generate_log.model_options` lets you confirm that options you passed were
forwarded to the model. This is useful when testing custom model option handling:

```python
from mellea.backends import ModelOption


def test_model_options_forwarded(session):
    model_options = {
        ModelOption.TEMPERATURE: 0.5,
        ModelOption.MAX_NEW_TOKENS: 100,
        "custom_param": "should_pass_through",
    }
    res = session.instruct(
        "Write a one-sentence summary.",
        model_options=model_options,
    )
    assert "custom_param" in res._generate_log.model_options
```

> **Note:** `_generate_log` is an internal attribute. Its structure may change
> between Mellea versions. Use it for debugging and option-forwarding tests, not
> as a primary correctness check.

## Using `simple_validate` for deterministic checks

`simple_validate` wraps a plain function into a validation callable that
`Requirement` accepts. Use it to assert deterministic structural constraints
inside the IVR loop, or directly in tests to verify that your validator logic
behaves correctly:

```python
from mellea.stdlib.requirements import Requirement, simple_validate


def test_simple_validate_logic():
    """Unit-test a validator without making any LLM calls."""
    validator = simple_validate(lambda x: (len(x) > 0, "Output must not be empty."))

    # Confirm the validator passes for non-empty output.
    # simple_validate returns a Context -> ValidationResult callable.
    # You can test the underlying function directly:
    result_fn = lambda text: (len(text) > 0, "Output must not be empty.")
    ok, _ = result_fn("hello")
    assert ok is True

    empty_ok, reason = result_fn("")
    assert empty_ok is False
    assert "empty" in reason
```

When you attach `simple_validate` to a `Requirement`, it checks the last model
output as a string, regardless of how the output was parsed:

```python
from mellea.stdlib.requirements import Requirement, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy


def test_with_simple_validate_requirement(session):
    res = session.instruct(
        "Reply with a number between 1 and 10.",
        requirements=[
            Requirement(
                "Reply with a number between 1 and 10.",
                validation_fn=simple_validate(
                    lambda x: (x.strip().isdigit(), "Expected a digit.")
                ),
            )
        ],
        strategy=RejectionSamplingStrategy(loop_budget=5),
    )
    assert res is not None
    assert isinstance(res.value, str)
```

## The `unit_test_eval` component

`mellea.stdlib.components.unit_test_eval` provides `TestBasedEval`, a
`Component` that formats an LLM-as-a-judge evaluation task. You load test cases
from a JSON file and pass them to a judge session. This is useful for offline
evaluation pipelines, not for individual pytest assertions.

### JSON file format

Each entry in the JSON array defines one test:

```json
[
  {
    "source": "email-classifier",
    "name": "positive_case_001",
    "instructions": "Evaluate whether the prediction correctly identifies the category.",
    "id": "tc-001",
    "examples": [
      {
        "input_id": "ex-001",
        "input": [{"role": "user", "content": "Is this email spam?"}],
        "targets": [{"role": "assistant", "content": "no"}]
      }
    ]
  }
]
```

### Loading and running evaluations

```python
from mellea import MelleaSession, start_session
from mellea.stdlib.components.unit_test_eval import TestBasedEval

# Load one TestBasedEval per test definition in the file.
test_evals = TestBasedEval.from_json_file("tests/eval_data/email_classifier.json")

judge_session = start_session()

for eval_case in test_evals:
    for idx, input_text in enumerate(eval_case.inputs):
        # Generate the prediction from the system under test.
        prediction = "no"  # replace with your actual model call

        targets = eval_case.targets[idx] if eval_case.targets else []
        eval_case.set_judge_context(input_text, prediction, targets)

        verdict = judge_session.instruct(eval_case)
        print(f"{eval_case.name}: {verdict.value}")
```

> **Note:** `TestBasedEval` calls the judge model once per input. For large
> evaluation sets, consider batching or running evaluations asynchronously.

## CI strategy

Follow these rules when deciding which tests run in CI:

| Test category | Marker | Runs in CI (`CICD=1`)? |
| ------------- | ------ | ---------------------- |
| Type and structural checks | `@pytest.mark.llm` | Yes |
| Qualitative content checks | `@pytest.mark.qualitative` | No — skipped automatically |
| Tests needing Ollama | `@pytest.mark.ollama` | Yes, if Ollama is in the CI environment |
| Tests taking >5 minutes | `@pytest.mark.slow` | Excluded from standard CI runs |

The skip is automatic: `conftest.py` calls `pytest.skip()` for any test marked
`qualitative` when `CICD=1`. You do not need to add any skip logic yourself.

> **Tip:** Run the full suite including qualitative tests before merging a prompt
> change. Use `CICD=0 pytest -m qualitative` locally to target only those tests.
>
> **Advanced:** To add a dedicated quality gate that runs qualitative tests on a
> separate schedule, create a GitHub Actions workflow that omits `CICD=1` and
> uses `-m qualitative` as the pytest filter.

## Next steps

- [The Requirements System](../concepts/requirements-system) — understand how
  `Requirement`, `simple_validate`, and `check` interact with the IVR loop
- [Handling Exceptions](../evaluation-and-observability/handling-exceptions) —
  catch and diagnose errors that occur during generation
