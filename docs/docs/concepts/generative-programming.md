---
title: "Generative Programming"
description: "The ideas behind Mellea — what generative programs are, why they're hard, and how Mellea addresses those challenges."
# diataxis: explanation
---

# Generative Programming

A [_generative program_](../guide/glossary#generative-program) is any program that contains calls to an LLM. This covers
everything from a simple prompt wrapper to a complex multi-step reasoning system.
The term is deliberately broad: what matters is not how many LLM calls a program
makes, but the structural challenges that arise when you combine stochastic LLM
operations with deterministic code.

Mellea is a library for writing generative programs well.

## The fundamental challenge

Classical programs are deterministic. Given the same input, they produce the same
output. You can reason about them, test them, and trust that the test results
generalise.

LLM calls are not deterministic. The same prompt, sent to the same model, with
the same temperature, may produce different outputs. These outputs may each be
valid responses to the prompt in a natural-language sense, but one may satisfy
the downstream requirements of your program and another may not.

Generative programs interleave these two modes. A Python function that calls an
LLM and then applies regular deterministic logic to the result is partly
predictable and partly not. The challenge of generative programming is managing
that boundary — ensuring that the stochastic parts are sufficiently constrained,
that failures are handled gracefully, and that uncertainty does not accumulate
unchecked through the system.

## Requirements as the core tool

The primary mechanism Mellea provides for managing stochasticity is [_requirements_](../guide/glossary#requirement).
A requirement is a validation function that checks whether an LLM output meets a
specified criterion:

```python
from mellea import start_session
from mellea.stdlib.requirements import req

m = start_session()
result = m.instruct(
    "Summarise this document in one sentence.",
    requirements=[
        req("Must be a single sentence"),
        req("Must be under 30 words"),
    ],
)
```

When the model's output fails a requirement, Mellea can retry the generation with
feedback — the [_Instruct–Validate–Repair_ (IVR)](../guide/glossary#ivr-instruct-validate-repair) loop. This transforms a
probabilistically unreliable call into one with measurable, controllable reliability:
set a `loop_budget` and the probability of the output satisfying your requirements
approaches 1 as budget increases.

Requirements can be simple string constraints, Python validation functions, or
powerful model-based validators like IBM Granite Guardian. The same machinery
handles all of them.

## Failure handling and sampling strategies

Not all requirements can be checked cheaply. A constraint like "this JSON is
syntactically valid" can be verified in microseconds; a constraint like "this
answer is grounded in the provided context" may require a second model call.

Mellea's [sampling strategies](../guide/glossary#sampling-strategy) control how retries work:

- **`RejectionSamplingStrategy`** — retry until a requirement passes or the budget
  is exhausted. The simplest strategy; good for cheap validators.
- **`SOFAISamplingStrategy`** — escalate from a fast S1 model to a slower S2 model
  only when S1 fails. Keeps cost low on easy inputs while handling hard ones.
- **`BudgetForcingSamplingStrategy`** — force extended thinking on hard problems
  by retrying with explicit budget pressure.

The feedback from a failed requirement (`ValidationResult.reason`) is passed back
to the model on the next attempt. This means the model can repair its output in
light of exactly what was wrong, rather than generating blindly.

## Uncertainty and long computation paths

In programs with multiple sequential LLM calls, uncertainty compounds. If each
call has a 90% chance of passing its requirements on the first attempt, a chain of
five calls has only about a 59% chance of all passing without a retry. Requirements
at every step are not defensive overhead — they are the mechanism that keeps
uncertainty from becoming multiplicative.

Intermediate validation also gives you early-exit points. A program that validates
each intermediate result can abandon a failing path quickly rather than running to
completion and then discovering the final output is wrong.

## Context and the accumulation of history

Generative programs also face a second structural challenge: context growth. Each
model call can take some prior context (conversation history, retrieved documents,
examples) as input, and over the course of a long program, that context can grow
large enough to exceed model limits or degrade output quality.

Mellea addresses this through explicit context management:

- **[`SimpleContext`](../guide/glossary#context)** (default) resets history on each call. The model sees only
  the current instruction. This is usually the right choice for independent calls.
- **[`ChatContext`](../guide/glossary#context)** preserves history for multi-turn conversations.
- **[Components](../guide/glossary#component)** ([`@mify`](../guide/glossary#mify--mify), [`@generative`](../guide/glossary#generative)) encapsulate the context needed for a
  single call, keeping context management compositional rather than global.

## Mellea's position in the ecosystem

Mellea is not an orchestration framework. It does not provide agents that plan and
dispatch subtasks, or graph-based workflow engines.

Mellea is the _reliable execution layer_ that those frameworks call. It is the part
of the system that ensures a single LLM call — or a tightly coupled group of calls —
meets its requirements before returning a result. Orchestrators like LangChain or
smolagents can use Mellea-instrumented functions as tools, and the reliability
guarantees those functions provide hold regardless of the orchestrator's structure.

This distinction matters for how you design systems. Mellea handles the vertical
reliability of each call. You handle the horizontal structure of the program —
how calls are composed, what order they run in, what data flows between them.

## Design principles

These principles recur throughout Mellea:

- **Circumscribe every LLM call with requirement verifiers.** Stochastic operations
  without verification are a source of silent failures.
- **Keep prompts small and composable.** Mellea decomposes programs into Components.
  Each Component encapsulates one prompt and its context. Complex programs are
  compositions of simple components, not one giant prompt.
- **Co-design models and inference programs.** Where possible, the prompting style
  used at inference time should match the style used during training. Mellea's
  support for Granite models reflects this: the library's prompting conventions and
  the models were built together.
- **Manage context explicitly.** Context is not a passive accumulation of everything
  that has happened. It is a resource that you manage deliberately, allocating what
  the model needs and discarding what it does not.

---


**See also:**
[Instruct, Validate, Repair](./instruct-validate-repair) |
[Inference-Time Scaling](../advanced/inference-time-scaling) |
[Working with Data](../guide/working-with-data)
