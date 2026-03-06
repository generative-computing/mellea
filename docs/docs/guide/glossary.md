---
title: "Glossary"
description: "Definitions of Mellea-specific terms and concepts."
# diataxis: reference
---

# Glossary

Mellea-specific terms used throughout this guide. Terms are listed alphabetically.
Cross-links from guide pages point here on **first use only**.

---

## act() / aact()

`act()` is the generic session method that runs any `Component` and returns a
result. Every higher-level method (`instruct()`, `chat()`, `query()`,
`transform()`) builds a Component and delegates to `act()`. Use `act()` directly
when working with custom components or building your own inference loops.

`aact()` is the async counterpart — same signature, same return types.

See: [act() and aact()](./act-and-aact.md)

---

## aLoRA (Activated LoRA)

An **Activated LoRA** (aLoRA) is a LoRA adapter dynamically loaded by
`LocalHFBackend` at inference time to serve as a lightweight requirement verifier.
Instead of running a full LLM call to check a requirement, the adapter is activated
on the same model weights already in memory.

See: [LoRA and aLoRA Adapters](../advanced/lora-and-alora-adapters.md)

---

## Backend

A backend is an inference engine that Mellea uses to run LLM calls. Examples:
`OllamaModelBackend`, `OpenAIBackend`, `LocalHFBackend`, `LocalVLLMBackend`,
`WatsonxAIBackend`. Backends are configured via `MelleaSession` or
`start_session()`.

See: [Backends and Configuration](./backends-and-configuration.md)

---

## CBlock

A `CBlock` (content block) is the low-level unit of content in Mellea. A `CBlock`
holds text (or image data) and is assembled by a `Component` into the prompt sent
to the backend. Multiple CBlocks compose into a single LLM request.

See: [Mellea Core Internals](../advanced/mellea-core-internals.md)

---

## Component

A `Component` is a reusable, composable unit in Mellea that encapsulates a prompt
structure, its requirements, and its parsing logic. `Instruction`, `Message`,
`MObject`, and `Document` are all Component subclasses. Components are the building
blocks of generative programs.

---

## Context

A `Context` holds the conversation history threaded through a `MelleaSession`.
Mellea provides `SimpleContext` (single-turn) and `ChatContext` (multi-turn). Push
and pop operations let you branch and restore context state across calls.

See: [Context and Sessions](../concepts/context-and-sessions.md)

---

## Generative function

A Python function decorated with `@generative`. Mellea uses the function's type
annotation as the output schema and its docstring as the prompt. Generative
functions are called with a `MelleaSession` as the first argument and return the
annotated type.

See: [Generative Functions](./generative-functions.md)

---

## Generative program

Any computer program that contains calls to an LLM. Mellea is a library for writing
robust, composable generative programs.

See: [Generative Programming](../concepts/generative-programming.md)

---

## GuardianCheck

A safety requirement in Mellea that validates LLM outputs against defined safety
rules before they are returned to the caller. Uses the Granite Guardian model as a
verifier.

See: [Security and Taint Tracking](../advanced/security-and-taint-tracking.md)

---

## Intrinsic

An `Intrinsic` is a backend-level primitive in Mellea — a structured generation
operation with special handling (e.g., constrained decoding, RAG retrieval). The
`LocalHFBackend` exposes Intrinsics directly; server backends route them through
adapter endpoints.

See: [Intrinsics](../advanced/intrinsics.md)

---

## IVR (Instruct-Validate-Repair)

A core generative programming pattern in Mellea:

1. **Instruct** — call the LLM with a prompt.
2. **Validate** — check the output against a `Requirement`.
3. **Repair** — if validation fails, retry or fix the output.

See: [Instruct, Validate, Repair](../concepts/instruct-validate-repair.md)

---

## MelleaSession

The primary entry point for Mellea. A `MelleaSession` wraps a backend and provides
`instruct()`, `chat()`, `act()`, `aact()`, `query()`, and `transform()` as
session-level methods. Use `mellea.start_session()` to create one with defaults.

```python
import mellea
m = mellea.start_session()  # returns a MelleaSession
```

---

## mify / @mify

The `@mify` decorator turns any Python class into an **MObject** — an
LLM-queryable, tool-accessible wrapper around your data. You specify which fields
and methods are visible to the LLM; everything else remains hidden.

See: [MObjects and mify](../concepts/mobjects-and-mify.md)

---

## MObject

An **MObject** is a Python class decorated with `@mify`. It wraps existing data
objects so they can be queried and transformed by the LLM via `m.query()` and
`m.transform()`. Unlike `@generative`, `@mify` does not change the class's Python
interface — it adds a layer that the LLM can see and call.

See: [MObjects and mify](../concepts/mobjects-and-mify.md)

---

## ModelOption

An enum (`mellea.backends.ModelOption`) of backend-agnostic inference options:
`TEMPERATURE`, `SEED`, `MAX_NEW_TOKENS`, `SYSTEM_PROMPT`, etc. Using `ModelOption`
keys ensures the same options work across all backends.

```python
from mellea.backends import ModelOption
```

See: [Configure Model Options](../how-to/configure-model-options.md)

---

## ModelOutputThunk

The return type of `m.instruct()`, `m.act()`, and most session-level generative
calls. Access the result via `.value` (returns the typed output) or `str(thunk)`.
The value is evaluated lazily — not computed until first accessed.

---

## Requirement

A `Requirement` is a validation constraint applied to a generative function's
output. Requirements can be programmatic (lambda, regex, type check) or generative
(another LLM call). Used in the IVR pattern.

See: [Requirements System](../concepts/requirements-system.md)

---

## Sampling strategy

A `SamplingStrategy` controls how the IVR loop behaves when a requirement fails.
Mellea's built-in strategies:

| Strategy | Behaviour |
| --- | --- |
| `RejectionSamplingStrategy` | Retry up to `loop_budget` times; return first passing result |
| `MajorityVotingStrategy` | Generate N candidates; return the one supported by most |
| `SOFAISamplingStrategy` | Fast System-1 generation verified by a slower System-2 model |
| `BudgetForcingSamplingStrategy` | Inject thinking tokens to expand reasoning budget |

See: [Inference-Time Scaling](../advanced/inference-time-scaling.md)

---

## SamplingResult

The return type of session calls made with `return_sampling_results=True`, and of
the `serve()` function used with `m serve`. Holds `.result` (the selected output),
`.success` (whether a requirement was met), and `.sample_generations` (all
candidates generated).

---

## SOFAI

**SOFAI** (System-1 / System-2 AI) is a sampling strategy in Mellea that mirrors
dual-process cognition: a fast "System 1" model generates candidates and a slower
"System 2" model verifies them. Uses `SOFAISamplingStrategy`.

See: [Inference-Time Scaling](../advanced/inference-time-scaling.md)

---

## Tool

A Python function decorated with `@tool` (or registered via `MelleaSession`) that
Mellea exposes to an LLM for function calling. Tools have typed inputs and outputs
so the LLM can call them reliably without free-form parsing.

See: [Tools and Agents](./tools-and-agents.md)

---

## Thunk

See [ModelOutputThunk](#modeloutputthunk).

---

**Previous:** [Mellea Core Internals](../advanced/mellea-core-internals.md) |
**Next:** [Common Errors](../troubleshooting/common-errors.md)
