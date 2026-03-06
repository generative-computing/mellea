---
title: "Building Custom Components"
description: "Implement the Component Protocol to create reusable, testable generative building blocks."
# diataxis: how-to
---

> **Advanced:** This page is for developers who need to go beyond the standard
> `@generative`, `instruct()`, and `m.chat()` API. If you are getting started
> with Mellea, see the [Quick Start](../getting-started/quickstart) first.

The `Component` Protocol is the fundamental unit of composition in Mellea. Every
high-level API call — `m.instruct()`, `@generative`, `m.chat()` — is backed by a
`Component` that formats its input for the LLM and parses the output into a typed
result. This page shows you how to implement the protocol yourself.

## When to build a custom component

Use the standard API in most cases. Build a custom `Component` when:

- You need a domain-specific prompt structure that cannot be expressed as a
  `@generative` docstring or an `instruct()` template.
- You need deterministic, reusable parsing logic across many call sites —
  not ad-hoc post-processing.
- You want to unit-test prompt formatting and output parsing in isolation,
  without a real backend.
- You are building a reusable library component that other developers will import.
- You need to feed a `ModelOutputThunk` from one LLM call directly into the
  formatted input of another (lazy composition).

If none of these apply, `@generative` or `instruct()` covers your use case with
less boilerplate.

## The Component Protocol

[`Component`](../guide/glossary#component) is a `Protocol` generic over `S`, the return type produced when the
component parses LLM output:

```python
from mellea.core import CBlock, Component, ModelOutputThunk
```

The protocol has three required methods and one public method that wraps `_parse`:

| Method | Signature | Purpose |
| ------ | --------- | ------- |
| `parts()` | `-> list[Component \| CBlock]` | Returns child components and [`CBlock`](../guide/glossary#cblock) content blocks |
| `format_for_llm()` | `-> TemplateRepresentation \| str` | Formats the component for LLM consumption |
| `_parse()` | `(computed: ModelOutputThunk) -> S` | Parses LLM output into the return type `S` |
| `parse()` | `(computed: ModelOutputThunk) -> S` | Public wrapper — catches exceptions as [`ComponentParseError`](../guide/glossary#componentparseerror) |

You implement `parts()`, `format_for_llm()`, and `_parse()`. You do not override
`parse()` — the base implementation calls `_parse()` and wraps any exception in a
`ComponentParseError` so callers always get a consistent error type.

### Type parameter

`Component[S]` is parameterised by `S`: the Python type your `_parse` method
returns. For example, `Component[str]` returns a plain string, while
`Component[list[str]]` returns a list. The type parameter is enforced at static
analysis time by mypy.

## Minimal example: FeedbackForm

The following component formats a structured feedback request and parses the
model's response into a Python dictionary.

```python
import json

from mellea.core import CBlock, Component, ModelOutputThunk


class FeedbackForm(Component[dict[str, str]]):
    """Asks the model to rate content on several dimensions and return JSON."""

    def __init__(self, content: str, dimensions: list[str]) -> None:
        self._content = content
        self._dimensions = dimensions

    def parts(self) -> list[Component | CBlock]:
        return [CBlock(self._content)]

    def format_for_llm(self) -> str:
        dims = ", ".join(self._dimensions)
        return (
            f"Rate the following content on these dimensions: {dims}.\n"
            f"Respond with a JSON object mapping each dimension to a score "
            f'between 1 and 5 and a one-sentence reason. Use the format:\n'
            f'{{"dimension": {{"score": 3, "reason": "..."}}}}\n\n'
            f"Content:\n{self._content}"
        )

    def _parse(self, computed: ModelOutputThunk) -> dict[str, str]:
        raw = computed.value or ""
        # Strip markdown fences if the model wraps the JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
```

Pass the component to `m.act()` to get a result:

```python
import mellea.stdlib.functional as mfuncs
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.context import SimpleContext

backend = OllamaModelBackend("granite4:latest")
ctx = SimpleContext()

form = FeedbackForm(
    content="The onboarding flow was confusing and took too long.",
    dimensions=["clarity", "tone", "actionability"],
)

thunk, _ = mfuncs.act(action=form, context=ctx, backend=backend)
result = form.parse(thunk)
print(result)
# {"clarity": {"score": 2, "reason": "..."}, ...}
```

You can also use `MelleaSession.act()` — the session method is a thin wrapper
around the same functional API:

```python
from mellea import start_session

with start_session() as m:
    thunk = m.act(form)
    result = form.parse(thunk)
```

## Using TemplateRepresentation for Jinja2-based rendering

For components that need model-specific prompt formatting, return a
[`TemplateRepresentation`](../guide/glossary#templaterepresentation) from `format_for_llm()` instead of a plain string.
`TemplateRepresentation` is a dataclass with these fields:

| Field | Type | Purpose |
| ----- | ---- | ------- |
| `obj` | `Any` | The component instance (typically `self`) |
| `args` | `dict` | Variables passed to the Jinja2 template |
| `tools` | `dict \| None` | Tool definitions available in the template |
| `template` | `str \| None` | Inline Jinja2 template string |
| `template_order` | `list[str] \| None` | Template file names to look up; `"*"` means the class name |
| `images` | `list \| None` | Image blocks to include |

The formatter resolves template files from a `templates/prompts/` directory,
traversing subdirectories that match the model ID before falling back to
`default/`. See [Mellea Core Internals](../advanced/mellea-core-internals) for
the full lookup order.

```python
from mellea.core import CBlock, Component, ModelOutputThunk, TemplateRepresentation


class FeedbackFormTemplate(Component[dict]):
    """FeedbackForm variant using a Jinja2 template for rendering."""

    def __init__(self, content: str, dimensions: list[str]) -> None:
        self._content = content
        self._dimensions = dimensions

    def parts(self) -> list[Component | CBlock]:
        return [CBlock(self._content)]

    def format_for_llm(self) -> TemplateRepresentation:
        return TemplateRepresentation(
            obj=self,
            args={
                "content": self._content,
                "dimensions": self._dimensions,
            },
            template_order=["*"],  # looks up FeedbackFormTemplate.jinja2
        )

    def _parse(self, computed: ModelOutputThunk) -> dict:
        import json

        raw = computed.value or ""
        return json.loads(raw.strip())
```

Place the template file at
`mellea/templates/prompts/default/FeedbackFormTemplate.jinja2`:

```text
Rate the following content on these dimensions: {{ dimensions | join(", ") }}.
Respond with a JSON object mapping each dimension to a score between 1 and 5
and a one-sentence reason.

Content:
{{ content }}
```

Use inline `template=` for one-off components where a separate file is
unnecessary:

```python
from mellea.core import CBlock, Component, ModelOutputThunk, TemplateRepresentation

TEMPLATE = """\
Summarise in {{ max_words }} words or fewer:

{{ text }}
"""


class SummaryComponent(Component[str]):
    """Summarises text to a word limit."""

    def __init__(self, text: str, max_words: int = 50) -> None:
        self._text = text
        self._max_words = max_words

    def parts(self) -> list[Component | CBlock]:
        return [CBlock(self._text)]

    def format_for_llm(self) -> TemplateRepresentation:
        return TemplateRepresentation(
            obj=self,
            args={"text": self._text, "max_words": self._max_words},
            template=TEMPLATE,
        )

    def _parse(self, computed: ModelOutputThunk) -> str:
        return (computed.value or "").strip()
```

## Registering with act()

You do not need to register or annotate a custom component. Pass it directly to
`m.act()` or `mfuncs.act()`:

```python
import mellea.stdlib.functional as mfuncs
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.context import SimpleContext

backend = OllamaModelBackend("granite4:latest")
ctx = SimpleContext()

component = SummaryComponent("Long article text here...", max_words=30)
thunk, _ = mfuncs.act(action=component, context=ctx, backend=backend)
result = component.parse(thunk)
print(result)
```

For async workflows, use `mfuncs.aact()`:

```python
import asyncio
import mellea.stdlib.functional as mfuncs
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.context import SimpleContext


async def main() -> None:
    backend = OllamaModelBackend("granite4:latest")
    ctx = SimpleContext()
    component = SummaryComponent("Long article text here...", max_words=30)
    thunk, _ = await mfuncs.aact(action=component, context=ctx, backend=backend)
    print(component.parse(thunk))


asyncio.run(main())
```

## Testing custom components

Because `Component` is a Protocol, you can test formatting and parsing without a
real backend. Create a `ModelOutputThunk` with a known value to exercise `_parse`
directly.

```python
import json
import pytest
from mellea.core import CBlock, ModelOutputThunk


def make_thunk(value: str) -> ModelOutputThunk:
    """Return a pre-computed thunk containing value."""
    thunk = ModelOutputThunk(value=value)
    return thunk


class TestFeedbackForm:
    def test_format_for_llm_contains_dimensions(self):
        form = FeedbackForm(
            content="Great product.",
            dimensions=["clarity", "tone"],
        )
        rendered = form.format_for_llm()
        assert "clarity" in rendered
        assert "tone" in rendered

    def test_parts_returns_cblock(self):
        form = FeedbackForm(content="Great product.", dimensions=["clarity"])
        parts = form.parts()
        assert len(parts) == 1
        assert isinstance(parts[0], CBlock)
        assert parts[0].value == "Great product."

    def test_parse_valid_json(self):
        form = FeedbackForm(content="x", dimensions=["clarity"])
        payload = json.dumps({"clarity": {"score": 4, "reason": "Clear."}})
        thunk = make_thunk(payload)
        result = form._parse(thunk)
        assert result["clarity"]["score"] == 4

    def test_parse_raises_component_parse_error_on_bad_json(self):
        from mellea.core import ComponentParseError

        form = FeedbackForm(content="x", dimensions=["clarity"])
        thunk = make_thunk("this is not json")
        with pytest.raises(ComponentParseError):
            form.parse(thunk)
```

> **Note:** `ModelOutputThunk` accepts a `value` keyword argument in tests. Check
> the current constructor signature in `mellea/core/base.py` if the import path
> changes in a future release.
>
> **Tip:** Keep `_parse` pure — no I/O, no side effects. This makes it trivial to
> unit test and means failures are always the model's fault, not your parsing code.

---

## Next steps

- [Mellea Core Internals](../advanced/mellea-core-internals) — understand
  `CBlock`, `ModelOutputThunk`, and the full abstraction stack that custom
  components plug into.
- [Write Custom Verifiers](../how-to/write-custom-verifiers) — combine custom
  components with requirement validation to build structured output pipelines
  with automatic retry.
