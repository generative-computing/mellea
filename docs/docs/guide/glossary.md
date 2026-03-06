---
title: "Glossary"
description: "Definitions of Mellea-specific terms and concepts."
# diataxis: reference
---

# Glossary

Mellea-specific terms used throughout this guide. Terms are listed alphabetically.
Cross-links from guide pages point here on **first use only**.

---

## ACT / AACT

**ACT** (Asynchronous Computation Tree) and **AACT** (Async ACT) are Mellea's execution models for running generative programs. ACT describes a tree of computations where nodes can be LLM calls, tool calls, or classical functions. AACT is the asynchronous variant.

See: [ACT and AACT](./act-and-aact.md)

---

## Backend

A backend is an inference engine that Mellea uses to run LLM calls. Examples: Ollama, OpenAI-compatible APIs (vLLM, WatsonX), HuggingFace. Backends are configured via `MelleaSession` or `start_session()`.

See: [Backends and Configuration](./backends-and-configuration.md)

---

## CBlock

A `CBlock` (computation block) is the low-level unit of computation in Mellea's execution model. CBlocks represent individual LLM calls or tool invocations and are composed into Components.

See: [Mellea Core Internals](../advanced/mellea-core-internals.md)

---

## Component

A `Component` is a reusable, composable unit in Mellea that encapsulates a prompt, its requirements, and its context. Components are the building blocks of generative programs.

---

## Generative function

A Python function decorated with `@generative` (or the equivalent `@mify` decorator). Generative functions call an LLM and return a `ModelOutputThunk`.

See: [Generative Functions](./generative-functions.md)

---

## Generative program

Any computer program that contains calls to an LLM. Mellea is a library for writing robust, composable generative programs.

See: [Generative Programming](../concepts/generative-programming.md)

---

## GuardianCheck

A safety mechanism in Mellea that validates LLM outputs against defined safety rules before they are returned to the caller.

See: [Security and Taint Tracking](../advanced/security-and-taint-tracking.md)

---

## Intrinsic

An `Intrinsic` is a backend-level primitive in Mellea — a low-level operation with special handling for structured generation (e.g., constrained decoding). Intrinsics give fine-grained control over how generation happens.

See: [Intrinsics](../advanced/intrinsics.md)

---

## IVR (Instruct-Validate-Repair)

A core generative programming pattern in Mellea:

1. **Instruct** — call the LLM with a prompt.
2. **Validate** — check the output against a `Requirement`.
3. **Repair** — if validation fails, retry or fix the output.

---

## MelleaSession

The primary entry point for Mellea. A `MelleaSession` wraps a backend and provides `instruct()`, `generate()`, and other session-level methods.

```python
import mellea
m = mellea.start_session()  # returns a MelleaSession
```

---

## ModelOption

An enum (`mellea.backends.types.ModelOption`) of backend-agnostic inference options: `TEMPERATURE`, `SEED`, `MAX_NEW_TOKENS`, `SYSTEM_PROMPT`, etc. Using `ModelOption` keys ensures portability across backends.

See: [Backends and Configuration](./backends-and-configuration.md)

---

## ModelOutputThunk

The return type of `m.instruct()` and most session-level generative calls. Access the result via `.value` (returns a string) or `str(thunk)`.

---

## Requirement

A `Requirement` is a validation constraint applied to a generative function's output. Requirements can be programmatic (regex, type checks) or generative (another LLM call). Used in the IVR pattern.

---

## Sampling strategy

The algorithm used to select outputs during LLM inference. Mellea provides standard strategies (greedy, top-k, top-p) and advanced ones including `RejectionSamplingStrategy` and `SOFAISamplingStrategy`.

See: [Inference-Time Scaling](../advanced/inference-time-scaling.md)

---

## SOFAI

**SOFAI** (System-1 / System-2 AI) is an advanced sampling strategy in Mellea that uses a fast "System 1" model for initial generation and a slower "System 2" model to verify and potentially repair outputs — mirroring dual-process cognition theory.

See: [Inference-Time Scaling](../advanced/inference-time-scaling.md)

---

## Tool

A Python function decorated with `@tool` that Mellea exposes to an LLM for function calling. Tools have typed inputs and outputs so the LLM can call them reliably.

See: [Tools and Agents](./tools-and-agents.md)

---

## Thunk

See [ModelOutputThunk](#modeloutputthunk).

---

**Previous:** [Mellea Core Internals](../advanced/mellea-core-internals.md) |
**Next:** [Common Errors](../troubleshooting/common-errors.md)
