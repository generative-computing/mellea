---
canonical: "https://docs.mellea.ai/examples/data-extraction-pipeline"
title: "Data Extraction Pipeline"
description: "Use the @generative decorator with a typed return value to extract structured data from unstructured text in a single declarative function."
# diataxis: reference
---

This example shows the most direct path from raw text to typed, structured
output in Mellea: a `@generative` function whose return annotation tells the
runtime exactly what shape the result must have.

**Source file:** `docs/examples/information_extraction/101_with_gen_stubs.py`

## Concepts covered

- Declaring a generative function with `@generative`
- Using a `list[str]` return type as an extraction contract
- Passing a session (`m`) as the first argument to a generative function
- Keyword-only input via `doc=`

## Prerequisites

- [Quick Start](../getting-started/quickstart) complete
- Ollama running locally with `granite4.1:3b` pulled

## The full example

### Imports and session

```python
from mellea import generative, start_session
from mellea.backends import model_ids

m = start_session()
```

`start_session()` with no arguments creates a session backed by the default
local model. The `model_ids` import is available if you want to switch to a
specific model later (see [Backends and configuration](../how-to/backends-and-configuration)).

### Declaring the extraction function

```python
@generative
def extract_all_person_names(doc: str) -> list[str]:
    """Given a document, extract names of ALL mentioned persons. Return these names as list of strings."""
```

The `@generative` decorator converts a bare function stub into a generative
stub. Three things drive the extraction:

- **Parameter names** (`doc`) become the named inputs the model receives.
- **Return annotation** (`list[str]`) tells the runtime to parse and validate
  the response as a JSON array of strings. If the model returns something that
  cannot be coerced to that type, Mellea retries automatically.
- **Docstring** is the task description sent to the model. Write it as a
  precise instruction — the docstring is the prompt.

No function body is needed. The decorator supplies the implementation.

### Running the extraction

```python
# ref: https://www.nytimes.com/2012/05/20/world/world-leaders-at-us-meeting-urge-growth-not-austerity.html
NYTimes_text = "CAMP DAVID, Md. — Leaders of the world's richest countries banded together on Saturday to press Germany to back more pro-growth policies to halt the deepening debt crisis in Europe, as President Obama for the first time gained widespread support for his argument that Europe, and the United States by extension, cannot afford Chancellor Angela Merkel's one-size-fits-all approach emphasizing austerity."

person_names = extract_all_person_names(m, doc=NYTimes_text)

print(f"person_names = {person_names}")
# out: person_names = ['President Obama', 'Angela Merkel']
```

Calling the decorated function follows a consistent pattern across all
generative functions: pass the session as the first positional argument, then
pass the declared parameters as keyword arguments. The return value is the
extracted, type-validated data — not a raw string or a thunk.

### Full file

```python
# pytest: ollama, llm

"""Simple Example of information extraction with Mellea using generative stubs."""

from mellea import generative, start_session
from mellea.backends import model_ids

m = start_session()


@generative
def extract_all_person_names(doc: str) -> list[str]:
    """Given a document, extract names of ALL mentioned persons. Return these names as list of strings."""


# ref: https://www.nytimes.com/2012/05/20/world/world-leaders-at-us-meeting-urge-growth-not-austerity.html
NYTimes_text = "CAMP DAVID, Md. — Leaders of the world's richest countries banded together on Saturday to press Germany to back more pro-growth policies to halt the deepening debt crisis in Europe, as President Obama for the first time gained widespread support for his argument that Europe, and the United States by extension, cannot afford Chancellor Angela Merkel's one-size-fits-all approach emphasizing austerity."

person_names = extract_all_person_names(m, doc=NYTimes_text)

print(f"person_names = {person_names}")
# out: person_names = ['President Obama', 'Angela Merkel']
```

## Key observations

**The docstring is the prompt.** There is no separate template file or prompt
string. Writing a clear, imperative docstring is the primary tool for
controlling extraction quality.

**The return type is the schema.** `list[str]` is simple, but the same
mechanism works for `Literal["positive", "negative", "neutral"]`, Pydantic
models, or any other type that Mellea knows how to validate. See
[Enforce structured output](../how-to/enforce-structured-output) for richer
return types.

**Sessions are explicit.** Passing `m` as the first argument makes the
dependency on a live backend visible at the call site. You can pass different
sessions in tests (for example, a session backed by a mock) without changing
the function definition.

**What to try next:**

- Replace `list[str]` with a Pydantic model to extract multiple fields at
  once — see [Enforce structured output](../how-to/enforce-structured-output).
- Add `requirements` to the `@generative` call to enforce constraints on the
  extracted values — see the
  [requirements system concept](../concepts/requirements-system).
- Look at `docs/examples/information_extraction/advanced_with_m_instruct.py`
  for a version that uses `m.instruct()` directly with structured outputs.

---

**See also:** [Enforce Structured Output](../how-to/enforce-structured-output) | [The Requirements System](../concepts/requirements-system) | [Examples Index](./index)
