---
title: "Building Extensions"
description: "Create custom components, backends, sampling strategies, and requirements to extend Mellea."
# diataxis: how-to
---

**Prerequisites:** Mellea installed (`uv sync --all-extras --all-groups`), familiarity with the [core concepts](../concepts/requirements-system).

Mellea is designed to be extended at every layer. You can add new Requirements,
Components, Sampling Strategies, and Backends without modifying the core library.

## Three contribution pathways

Choose the pathway that fits the scope of your work:

| Pathway | When to use |
| ------- | ----------- |
| **Core repository** | General-purpose additions that benefit all users — open an issue first to discuss placement |
| **Your own repo** (`mellea-` prefix) | Application-specific or domain-specific libraries |
| **[mellea-contribs](https://github.com/generative-computing/mellea-contribs)** | Experimental or specialized components not yet ready for the standard library |

> **Note:** For general-purpose Components, Requirements, or Sampling Strategies,
> open an issue before submitting a PR. This avoids duplication and ensures
> the addition lands in the right place (standard library vs. mellea-contribs).

## Custom requirements

A [`Requirement`](../reference/glossary#requirement) validates a generation against a
criterion. You can provide a Python function for deterministic checks, or rely on
LLM-as-a-Judge for semantic validation.

### Deterministic requirement

Pass a `validation_fn` that receives a `Context` and returns a `ValidationResult`:

```python
from mellea.core.requirement import Requirement, ValidationResult
from mellea.core.base import Context


def contains_json(ctx: Context) -> ValidationResult:
    """Check that the last output contains a JSON object."""
    last = ctx.last_output()
    text = last.value or ""
    passed = "{" in text and "}" in text
    return ValidationResult(
        passed,
        reason="Output contains JSON" if passed else "No JSON object found",
    )


json_requirement = Requirement(
    description="The output must contain a JSON object.",
    validation_fn=contains_json,
)
```

### LLM-as-a-Judge requirement

Omit `validation_fn` to use LLM-as-a-Judge. Mellea sends the requirement
`description` to the model and interprets a "yes"/"no" answer:

```python
from mellea.core.requirement import Requirement

formal_tone = Requirement(
    description="The response uses formal, professional language throughout.",
)
```

### Custom output-to-bool mapping

Supply `output_to_bool` to change how the model's response is interpreted:

```python
from mellea.core.requirement import Requirement
from mellea.core.base import CBlock


def strict_yes(output: CBlock | str) -> bool:
    """Accept only an exact 'YES' response."""
    return str(output).strip().upper() == "YES"


strict_requirement = Requirement(
    description="The answer is factually accurate.",
    output_to_bool=strict_yes,
)
```

For deeper validation patterns, see [Write Custom Verifiers](../how-to/write-custom-verifiers).

## Custom components

A [`Component`](../reference/glossary#component) is a composite data structure that an LLM
can read and write. Implement the `Component` protocol by providing `parts`,
`format_for_llm`, and `_parse`:

```python
from mellea.core.base import (
    CBlock,
    Component,
    ModelOutputThunk,
    TemplateRepresentation,
)


class TaggedOutput(Component[str]):
    """A component that wraps output in XML-style tags."""

    def __init__(self, tag: str, prompt: str) -> None:
        """Initialize a tagged output component.

        Args:
            tag: The XML tag name to wrap the output.
            prompt: The instruction prompt for the LLM.
        """
        self.tag = tag
        self.prompt = prompt

    def parts(self) -> list[Component | CBlock]:
        """Return the constituent parts of this component."""
        return [CBlock(self.prompt)]

    def format_for_llm(self) -> TemplateRepresentation | str:
        """Format the component for the LLM."""
        return f"{self.prompt}\nRespond inside <{self.tag}></{self.tag}> tags."

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Extract the content between the tags."""
        text = computed.value or ""
        start = text.find(f"<{self.tag}>")
        end = text.find(f"</{self.tag}>")
        if start == -1 or end == -1:
            return text
        return text[start + len(self.tag) + 2 : end]
```

For a full walkthrough of the Component protocol and templating system, see
[Custom Components](../advanced/custom-components).

## Custom sampling strategies

A [`SamplingStrategy`](../reference/glossary#sampling-strategy) controls how Mellea
generates and validates outputs — for example, rejection sampling, best-of-n, or
beam search. Subclass `SamplingStrategy` and implement `sample`:

```python
import asyncio
from mellea.core.backend import Backend
from mellea.core.base import Component, Context, ModelOutputThunk, S
from mellea.core.requirement import Requirement
from mellea.core.sampling import SamplingResult, SamplingStrategy


class BestOfNStrategy(SamplingStrategy):
    """Sample N candidates and return the one that passes the most requirements."""

    def __init__(self, n: int = 3) -> None:
        """Initialize best-of-n sampling.

        Args:
            n: Number of candidates to generate before selecting the best.
        """
        self.n = n

    async def sample(
        self,
        action: Component[S],
        context: Context,
        backend: Backend,
        requirements: list[Requirement] | None,
        *,
        validation_ctx: Context | None = None,
        format: type | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> SamplingResult[S]:
        """Generate N candidates and return the best one.

        Args:
            action: The component to generate a response for.
            context: The current session context.
            backend: The backend used for generation.
            requirements: Requirements to validate each candidate against.
            validation_ctx: Optional context override for validation.
            format: Structured output format, if any.
            model_options: Model options to pass to the backend.
            tool_calls: Whether to enable tool calls during generation.

        Returns:
            SamplingResult containing the selected candidate and validation details.
        """
        generations: list[ModelOutputThunk[S]] = []
        contexts: list[Context] = []
        actions: list[Component[S]] = []
        validations: list[list[tuple[Requirement, object]]] = []

        for _ in range(self.n):
            thunk, new_ctx = await backend.generate_from_context(
                action,
                context,
                format=format,
                model_options=model_options,
                tool_calls=tool_calls,
            )
            await thunk.avalue()
            generations.append(thunk)
            contexts.append(new_ctx)
            actions.append(action)
            validations.append([])

        # Return the first generation for this minimal example.
        return SamplingResult(
            result_index=0,
            success=True,
            sample_generations=generations,
            sample_validations=validations,
            sample_actions=actions,
            sample_contexts=contexts,
        )
```

For built-in strategies and advanced patterns, see
[Inference-Time Scaling](../advanced/inference-time-scaling).

## Custom backends

A [`Backend`](../reference/glossary#backend) connects Mellea to an inference provider.
Subclass the abstract `Backend` class from `mellea.core.backend` and implement
the two abstract methods:

```python
import asyncio
from collections.abc import Sequence

from mellea.core.backend import Backend
from mellea.core.base import C, CBlock, Component, Context, ModelOutputThunk


class EchoBackend(Backend):
    """A minimal backend that echoes the action text back as output.

    Useful for testing pipelines without a real inference provider.
    """

    async def generate_from_context(
        self,
        action: Component[C] | CBlock,
        ctx: Context,
        *,
        format: type | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[ModelOutputThunk[C], Context]:
        """Generate a response by echoing the action text.

        Args:
            action: The action component or block to respond to.
            ctx: The current session context.
            format: Ignored by this backend.
            model_options: Ignored by this backend.
            tool_calls: Ignored by this backend.

        Returns:
            A tuple of (ModelOutputThunk, updated Context).
        """
        text = str(action)
        thunk: ModelOutputThunk[C] = ModelOutputThunk(value=f"ECHO: {text}")
        new_ctx = ctx.add(thunk)
        return thunk, new_ctx

    async def generate_from_raw(
        self,
        actions: Sequence[Component[C] | CBlock],
        ctx: Context,
        *,
        format: type | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> list[ModelOutputThunk]:
        """Generate responses for a list of actions without using context.

        Args:
            actions: List of actions to generate responses for.
            ctx: Context (not used by this backend).
            format: Ignored by this backend.
            model_options: Ignored by this backend.
            tool_calls: Ignored by this backend.

        Returns:
            List of ModelOutputThunks, one per action.
        """
        return [ModelOutputThunk(value=f"ECHO: {str(a)}") for a in actions]
```

The full `Backend` abstract interface is documented in the
[API reference](/api/mellea/core/backend).

> **Note:** Production backends handle async streaming, tokenization, and error
> recovery. Study an existing backend in `mellea/backends/` before implementing
> a provider integration.

## Community contributions via mellea-contribs

[mellea-contribs](https://github.com/generative-computing/mellea-contribs) is the
home for experimental and specialized extensions that are not yet part of the
standard library. It is the right place for:

- Domain-specific Components (legal, medical, code review, etc.)
- Experimental Sampling Strategies under active research
- Backend integrations for niche or self-hosted providers

**To contribute:**

1. Open an issue on mellea-contribs describing your extension.
2. Fork the repository and create a branch.
3. Follow the coding standards from the [contributing guide](../community/contributing-guide).
4. Open a pull request referencing the issue.

If a contribution in mellea-contribs matures and proves broadly useful, it can
graduate to the standard library via an issue in the core repository.

---

**See also:**
[Custom Components](../advanced/custom-components),
[Write Custom Verifiers](../how-to/write-custom-verifiers),
[Inference-Time Scaling](../advanced/inference-time-scaling)
