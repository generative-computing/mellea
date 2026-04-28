---
title: "Unit Test Generative Code"
description: "Write reliable tests for @generative functions using pytest markers and output validation."
# diataxis: how-to
---

**Prerequisites:** [Quick Start](../getting-started/quickstart) complete,
`pip install mellea`, Ollama running locally, `pytest` installed.

> **Contributing to Mellea itself?** See the [Contributing Guide](../community/contributing-guide#testing)
> for Mellea's own test markers, fixtures, and CI setup.

Testing generative code requires you to separate concerns: some assertions are
always deterministic (the output is the right type), while others depend on model
behaviour and are inherently qualitative. This page shows you how to structure
both categories, configure the right pytest markers, and make your CI pipeline
fast and reliable.

## Three levels of assertion

Every test for a `@generative` function falls into one of four levels:

| Level | What you assert | Deterministic? |
| ----- | --------------- | -------------- |
| **Type check** | `isinstance(result, bool)` | Yes — constrained decoding always returns the declared type |
| **Structural check** | `result in ["positive", "negative"]` or field names present | Yes — schema enforcement is deterministic |
| **Qualitative check** | `assert result is True` | No — depends on the model and prompt |
| **Semantic evaluation** |	Judge model scores output against reference responses |	No — run separately, not a pytest assertion |

*For levels 1–3, use pytest with the patterns below. For semantic evaluation against reference examples — where you want a judge model to score your model's outputs in bulk — see [The `unit_test_eval` component for Generative Unit Tests](#the-unit_test_eval-component-for-generative-unit-tests) at the end of this page.*

Type and structural checks run in CI. Qualitative checks carry
`@pytest.mark.qualitative` and are skipped in CI when `CICD=1` is set.

## Setting up a test session fixture

Use a `backend` fixture to handle CI versus local configuration, and a
function-scoped `session` fixture to give each test a clean slate:

```python
# Requires: mellea, pytest
# Returns: None
import os
import pytest
from mellea import MelleaSession
from mellea.backends.ollama import OllamaModelBackend

_MODEL_ID = "granite4:micro"


@pytest.fixture(scope="module")
def backend():
    """Ollama backend — swap for any backend your app uses."""
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    return OllamaModelBackend(model_id=_MODEL_ID, host=host)


@pytest.fixture(scope="function")
def session(backend):
    """Fresh MelleaSession for each test."""
    m = MelleaSession(backend=backend)
    yield m
    m.reset()
```

> **Note:** Scoping `backend` to `module` and `session` to `function` strikes a
> balance between setup cost and test isolation. Each test gets a clean context,
> but the backend connection is created once per module.

## Module-level markers

Declare markers at the top of your test file with `pytestmark` so they apply to
every test in the module without repetition. Register your own markers in
`pyproject.toml` under `[tool.pytest.ini_options] markers` to avoid warnings:

```toml
[tool.pytest.ini_options]
markers = [
    "qualitative: tests that assert on LLM output content (skipped in CI)",
    "requires_ollama: tests that need Ollama running locally",
]
```

```python
import pytest

pytestmark = [pytest.mark.requires_ollama]
```

## Testing `@generative` functions

### Type assertions — always deterministic

The return type of a `@generative` function is enforced by constrained decoding
or output parsing. An `isinstance` check never depends on model behaviour:

```python
# Requires: mellea, pytest
# Returns: None
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
# Requires: mellea, pytest
# Returns: None
def test_classify_sentiment_structure(session):
    result = classify_sentiment(session, text="I love this product!")
    assert result in ["positive", "negative"]
```

For Pydantic model return types, assert that the required fields are present and
have the right types:

```python
# Requires: mellea, pydantic, pytest
# Returns: None
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
# Requires: mellea, pytest
# Returns: None
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
# Requires: mellea, pytest
# Returns: None
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
# Requires: mellea, pytest
# Returns: None
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
# Requires: mellea, pytest
# Returns: None
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
# Requires: mellea, pytest
# Returns: None
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

## The `unit_test_eval` component for Generative Unit Tests

`mellea.stdlib.components.unit_test_eval` provides `TestBasedEval`, a
`Component` that formats an LLM-as-a-judge evaluation task for generative unit testing. You load test cases
from a JSON file and pass them to a judge session. This is useful for offline
evaluation pipelines, 
not for individual pytest assertions. 

Given a task, you provide test cases that consist of evaluation instructions 
and a set of examples, along with associated metadata. Each example, in conversational format, consists of an input 
and (optional) target / reference response(s), which can be used to guide evaluation along with the evaluation instructions. 
They can either be instantiated inline or in JSON format, with a separate JSON file per task. 

There are no limitations on the number of test examples per task, and each input can have multiple reference responses. 
The evaluation instructions apply to all the test cases in your task.

### JSON file format

Each entry in the JSON array defines one test:

```json
[
  {
    "source": "professional-email-writing",
    "name": "case_001",
    "instructions": "The email should follow the instructions in the input.",
    "id": "tc-001",
    "examples": [
      {
        "input_id": "ex-001",
        "input": [{"role": "user", "content": "Write a brief professional follow-up email after a job interview"}],
        "targets": [
            {"role": "assistant", "content": "Thank you for taking the time to speak with me yesterday. I look forward to hearing about next steps at your convenience."},
            {"role": "assistant", "content": "I appreciate the opportunity to interview yesterday. Looking forward to hearing about next steps."}
        ]
      },
      {
        "input_id": "ex-002",
        "input": [{"role": "user", "content": "I just finished a client demo can you create a formal thank-you email"}],
        "targets": [{"role": "assistant", "content": "It was a pleasure sharing a product demo with you. Thank you for meeting with us."}]
      }
    ]
  }
]
```

### Loading and running evaluations

```python
# Requires: mellea
# Returns: None
from mellea import start_session
from mellea.stdlib.components import SimpleComponent
from mellea.stdlib.components.unit_test_eval import TestBasedEval

test_evals = TestBasedEval.from_json_file("tests/eval_data/email_writer.json")

judge_session = start_session(backend_name="ollama", model_id="granite4:micro")
generation_session = start_session(backend_name="ollama", model_id="granite4:micro")

for eval_case in test_evals:
    for idx, input_text in enumerate(eval_case.inputs):
        prediction = generation_session.act(
            SimpleComponent(instruction=input_text)
        ).value

        targets = eval_case.targets[idx] if eval_case.targets else []
        eval_case.set_judge_context(input_text, prediction, targets)

        verdict = judge_session.act(eval_case)
        # Note: verdict.value is the raw JSON string returned by the judge — {"score": 0|1, 
        # "justification": "..."}. Score 0 means the guidelines were violated; score 1 means the 
        # output is well aligned. Parse it to use the score programmatically:
        print(f"{eval_case.name}: {verdict.value}")
```

> **Note:** `TestBasedEval` calls the judge model once per input. For large
> evaluation sets, consider batching or running evaluations asynchronously.

> **CLI alternative:** The same evaluation can be run without writing Python:
`-m eval run tests/eval_data/email_writer.json --backend ollama --model granite4:micro`
> See `-m eval run --help` for full options.


## CI strategy

A simple `conftest.py` that skips qualitative tests in CI:

```python
# Requires: pytest
# Returns: None
# conftest.py
import os
import pytest

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "qualitative: assert on LLM output content — skip in CI"
    )

def pytest_collection_modifyitems(config, items):
    if os.environ.get("CI"):
        skip = pytest.mark.skip(reason="qualitative tests skipped in CI")
        for item in items:
            if "qualitative" in item.keywords:
                item.add_marker(skip)
```

Then in your GitHub Actions workflow:

```yaml
- name: Run tests
  run: pytest
  env:
    CI: "true"   # qualitative tests are automatically skipped
```

To run the full suite including qualitative tests locally:

```bash
pytest -m qualitative
```

| Test category | Marker | Runs in CI? |
| ------------- | ------ | ----------- |
| Type and structural checks | (none needed) | Yes |
| Qualitative content checks | `@pytest.mark.qualitative` | No — skipped when `CI=true` |
| Tests needing a running backend | `@pytest.mark.requires_ollama` | Only if Ollama is in CI |
| Long-running tests | `@pytest.mark.slow` | Optionally excluded |

## Next steps

- [The Requirements System](../concepts/requirements-system) — understand how
  `Requirement`, `simple_validate`, and `check` interact with the IVR loop
- [Handling Exceptions](../how-to/handling-exceptions) —
  catch and diagnose errors that occur during generation
- [Evaluate with LLM-as-a-Judge](../evaluation-and-observability/evaluate-with-llm-as-a-judge) —
  the `Requirement`-based approach for inline judge evaluation
