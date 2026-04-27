---
title: "Glossary"
description: "Definitions of Mellea-specific terms and concepts."
# diataxis: reference
---

Mellea-specific terms used throughout this guide. Terms are listed alphabetically.
Cross-links from guide pages point here on **first use only**.

---

## act() / aact()

`act()` is the generic session method that runs any `Component` and returns a
result. Every higher-level method (`instruct()`, `chat()`, `query()`,
`transform()`) builds a Component and delegates to `act()`. Use `act()` directly
when working with custom components or building your own inference loops.

`aact()` is the async counterpart — same signature, same return types.

See: [act() and aact()](./act-and-aact)

---

## aLoRA (Activated LoRA)

An **Activated LoRA** (aLoRA) is a LoRA adapter dynamically loaded by
`LocalHFBackend` at inference time to serve as a lightweight requirement verifier.
Instead of running a full LLM call to check a requirement, the adapter is activated
on the same model weights already in memory.

See: [LoRA and aLoRA Adapters](../advanced/lora-and-alora-adapters)

---

## @generative

A decorator that converts a typed Python function into an AI-powered function.
`@generative` uses the function's name, docstring, parameters, and return type
annotation to instruct the LLM. The output is constrained to match the return type.
Write the function in idiomatic Python — the more natural the signature and
docstring, the better the model understands and imitates it.

```python
from mellea import generative, start_session

@generative
def classify_language(code: str) -> str:
    """Return the programming language of the code snippet."""
    ...

m = start_session()
lang = classify_language(m, code="print('hello')")
```

See: [Generative Functions](./generative-functions)

---

## Backend

A backend is an inference engine that Mellea uses to run LLM calls. Examples:
`OllamaModelBackend`, `OpenAIBackend`, `LocalHFBackend`, `WatsonxAIBackend`. Backends are configured via `MelleaSession` or
`start_session()`.

See: [Backends and Configuration](./backends-and-configuration)

---

## ChatContext

The standard multi-turn context implementation. `ChatContext` accumulates the full
conversation history and passes it to the backend on each call. Create one at the
start of a session and pass it through all calls to maintain state:

```python
from mellea.stdlib.context import ChatContext
ctx = ChatContext()
```

Use `window_size` to cap how many turns are sent to the backend:

```python
ctx = ChatContext(window_size=10)
```

Use `SimpleContext` instead for stateless, single-turn calls.

See: [Context and Sessions](../concepts/context-and-sessions)

---

## CBlock

A `CBlock` (content block) is the low-level unit of content in Mellea. A `CBlock`
holds text (or image data) and is assembled by a `Component` into the prompt sent
to the backend. Multiple CBlocks compose into a single LLM request.

See: [Mellea Core Internals](../advanced/mellea-core-internals)

---

## Component

A `Component` is a reusable, composable unit in Mellea that encapsulates a prompt
structure, its requirements, and its parsing logic. `Instruction`, `Message`,
`MObject`, and `Document` are all Component subclasses. Components are the building
blocks of generative programs.

See: [Building Custom Components](../advanced/custom-components)

---

## ComponentParseError

The exception raised by `Component.parse()` when the model's output cannot be
parsed into the component's declared return type `S`. `parse()` catches any
exception from `_parse()` and re-raises it as `ComponentParseError` so all callers
get a consistent error type regardless of the underlying parse implementation.

```python
from mellea.core import ComponentParseError

try:
    result = form.parse(thunk)
except ComponentParseError as e:
    print(f"Parsing failed: {e}")
```

See: [Building Custom Components](../advanced/custom-components)

---

## ContextTurn

A single turn of model input and model output stored inside a `Context`. Each call
to `m.instruct()`, `m.chat()`, or `m.act()` appends a `ContextTurn` to the active
context. Turns are consumed by the backend formatter to build the conversation
history sent to the model.

---

## Context

A `Context` holds the conversation history threaded through a `MelleaSession`.
Mellea provides `SimpleContext` (single-turn) and `ChatContext` (multi-turn). Push
and pop operations let you branch and restore context state across calls.

See: [Context and Sessions](../concepts/context-and-sessions)

---

## Document

A `Component` that wraps a plain-text reference document for inclusion in a prompt.
Pass one or more `Document` objects in the `_docs` field of a `Message` or directly
as grounding context in an `Instruction`. Unlike `RichDocument`, `Document` holds
pre-extracted text rather than a parsed file.

```python
from mellea.stdlib.components.docs.document import Document
doc = Document(text="...", title="My doc", doc_id="ref-1")
```

---

## Generative function

A Python function decorated with `@generative`. Mellea uses the function's type
annotation as the output schema and its docstring as the prompt. Generative
functions are called with a `MelleaSession` as the first argument and return the
annotated type.

See: [Generative Functions](./generative-functions)

---

## Generative program

Any computer program that contains calls to an LLM. Mellea is a library for writing
robust, composable generative programs.

See: [Generative Programming](../concepts/generative-programming)

---

## GenerateLog

A dataclass that captures a single model call in detail. Pass a `list[GenerateLog]`
to `m.validate()` via the `generate_logs=` parameter to record the judge prompt and
raw verdict for each requirement validation:

```python
from mellea import start_session
from mellea.core import GenerateLog
from mellea.stdlib.requirements import req

logs: list[GenerateLog] = []
m = start_session()
result = m.instruct("Summarise this text.")
m.validate([req("Must be under 30 words.")], generate_logs=logs)

for log in logs:
    print(log.prompt)   # full judge prompt sent to the model
    print(log.result.value if log.result else None)  # raw verdict string
```

Key fields: `prompt`, `result` (`ModelOutputThunk | None`), `backend`,
`model_options`, `is_final_result`.

See: [Evaluate with LLM-as-a-Judge](../how-to/evaluate-with-llm-as-a-judge)

---

## Hook / HookType

A **hook** is an async function that intercepts a specific event in Mellea's
execution pipeline. The `@hook` decorator marks a function as a hook handler,
binding it to a `HookType` — one of 17 named events spanning session lifecycle,
component execution, generation, validation, sampling, and tool invocation.

Hooks receive a frozen payload and read-only context, and return `None` (pass
through), a `modify()` result (alter the payload), or a `block()` result (reject
the operation).

```python
from mellea.plugins import HookType, hook

@hook(HookType.GENERATION_PRE_CALL)
async def my_hook(payload, ctx):
    ...
```

See: [Plugins & Hooks](../concepts/plugins)

---

## grounding_context

The `grounding_context` parameter of `m.instruct()` accepts a dictionary of
named text entries that Mellea injects into the prompt as grounding evidence.
Each entry is tracked as a separate context component, so it can be traced
and rendered independently from the instruction template.

Use `grounding_context` to anchor the model's output to retrieved documents,
knowledge-base passages, or any reference material — without mixing that content
into `user_variables`:

```python
answer = m.instruct(
    "Answer the question: {{question}}",
    user_variables={"question": query},
    grounding_context={"doc0": doc_text_0, "doc1": doc_text_1},
)
```

Without `grounding_context`, `m.instruct()` generates from the model's parametric
knowledge only. It is the primary integration point for RAG pipelines.

See: [Build a RAG Pipeline](../how-to/build-a-rag-pipeline)

---

## CRITERIA_BANK

A dictionary mapping short string keys to full criteria descriptions used by
[`guardian_check()`](#guardian_check). Pass a key directly as the `criteria`
argument — the function looks up the full description automatically.

Available keys: `"harm"`, `"social_bias"`, `"jailbreak"`, `"profanity"`,
`"unethical_behavior"`, `"violence"`, `"groundedness"`, `"answer_relevance"`,
`"context_relevance"`, `"function_call"`.

```python
from mellea.stdlib.components.intrinsic.guardian import CRITERIA_BANK

print(list(CRITERIA_BANK.keys()))
```

See: [Safety Guardrails](../how-to/safety-guardrails)

---

## factuality_correction()

A Guardian Intrinsic function that generates a corrected version of the assistant's
last response grounded in the documents provided in context. Returns the corrected
text as a `str`, or `"none"` if the original response was already factually correct.

```python
from mellea.stdlib.components.intrinsic.guardian import factuality_correction
```

See: [Safety Guardrails](../how-to/safety-guardrails#factuality-correction)

---

## factuality_detection()

A Guardian Intrinsic function that evaluates whether the assistant's last response
is factually consistent with the documents in context. Returns `"yes"` if the
response contains factual errors, or `"no"` if it is consistent.

```python
from mellea.stdlib.components.intrinsic.guardian import factuality_detection
```

See: [Safety Guardrails](../how-to/safety-guardrails#factuality-detection)

---

## guardian_check()

A Guardian Intrinsic function that evaluates the last message from a given role in
a `ChatContext` against a safety or quality criterion. Returns a `float` score from
`0.0` (no risk) to `1.0` (risk detected); values at or above `0.5` indicate risk.

Accepts any key from [`CRITERIA_BANK`](#criteria_bank) or a custom free-text
criteria string.

```python
from mellea.stdlib.components.intrinsic import guardian

score = guardian.guardian_check(context, backend, criteria="harm")
```

See: [Safety Guardrails](../how-to/safety-guardrails)

---

## GuardianCheck

> **Deprecated as of v0.4.** Use [`guardian_check()`](#guardian_check),
> [`policy_guardrails()`](#policy_guardrails), or
> [`factuality_detection()`](#factuality_detection) from the Guardian Intrinsics
> instead. See [Safety Guardrails](../how-to/safety-guardrails).

A `Requirement` subclass that validated LLM outputs using a separately loaded
Granite Guardian model. Required an independent Ollama or HuggingFace backend
for the Guardian model.

See: [Security and Taint Tracking (deprecated)](../advanced/security-and-taint-tracking)

---

## GuardianRisk

> **Deprecated as of v0.4.** Use [`CRITERIA_BANK`](#criteria_bank) string keys
> with [`guardian_check()`](#guardian_check) instead.

An enum specifying which safety risk category the deprecated `GuardianCheck`
class should detect. Replaced by the string keys in `CRITERIA_BANK`.

See: [Security and Taint Tracking (deprecated)](../advanced/security-and-taint-tracking)

---

## policy_guardrails()

A Guardian Intrinsic function that checks whether a scenario complies with a
natural-language policy. Returns `"Yes"` (compliant), `"No"` (non-compliant), or
`"Ambiguous"` (insufficient information to decide).

```python
from mellea.stdlib.components.intrinsic.guardian import policy_guardrails
```

See: [Safety Guardrails](../how-to/safety-guardrails#policy-compliance)

---

## KV smashing

The technique of concatenating key-value attention caches from separately prefilled
prompt chunks along the time axis, producing a single merged `DynamicCache` that
covers the full context. Used by `LocalHFBackend` to avoid re-running forward
passes on content that has already been cached.

When a prompt contains a mix of cached and uncached `CBlock` objects, Mellea
prefills each block independently, then smashes the resulting caches together
before generation — giving results identical to a single full-context forward pass
at a fraction of the prefill cost.

See: [Prefix Caching and KV Blocks](../advanced/prefix-caching-and-kv-blocks)

---

## LiteLLM / LiteLLMBackend

`LiteLLMBackend` wraps [LiteLLM](https://docs.litellm.ai/) — a unified interface
over 100+ model providers. Use it to reach providers not covered by Mellea's
native backends: Bedrock via IAM, Vertex AI, Together AI, Cohere, and others.

```bash
pip install 'mellea[litellm]'
```

```python
m = mellea.start_session(
    backend_name="litellm",
    model_id="bedrock/converse/us.amazon.nova-pro-v1:0",
)
```

See: [Backends and Configuration](./backends-and-configuration)

---

## LLM-as-a-judge

The default validation strategy for `req()` in Mellea. After the model generates
an output, a second LLM call is made using the requirement's `description` as the
evaluation criterion. Mellea converts the judge's response to `True` / `False` by
looking for `"yes"` (case-insensitive) in the reply.

Use `simple_validate` instead when the criterion is deterministic (word count,
regex, type check) — no second LLM call is needed.

See: [Evaluate with LLM-as-a-Judge](../how-to/evaluate-with-llm-as-a-judge)

---

## ImageBlock

A Mellea type that represents an image in a backend-agnostic, encoded form. Use
`ImageBlock.from_pil_image(pil_image)` to convert a [Pillow](https://python-pillow.org/)
`Image` object into an `ImageBlock`. Both raw PIL images and `ImageBlock` objects are
accepted in the `images=[...]` parameter of `instruct()` and `chat()`.

Use `ImageBlock` when you need an already-encoded representation, or when the PIL image
is not directly available (e.g., passing between functions or caching).

See: [Use Images and Vision Models](../how-to/use-images-and-vision)

---

## Intrinsic

An `Intrinsic` is a backend-level primitive in Mellea — a structured generation
operation with special handling (e.g., constrained decoding, RAG retrieval). The
`LocalHFBackend` exposes Intrinsics directly; server backends route them through
adapter endpoints.

See: [Intrinsics](../advanced/intrinsics)

---

## Instruction

The core `Component` in the IVR loop. An `Instruction` wraps a prompt description,
optional requirements, in-context examples, and grounding context into a single
object that `m.act()` can execute. `m.instruct()` is a convenience wrapper that
builds an `Instruction` for you.

```python
from mellea.stdlib.components.instruction import Instruction
instr = Instruction(
    description="Summarise the following text: {{text}}",
    requirements=[req("Must be under 50 words.")],
    user_variables={"text": "..."},
)
result = m.act(instr)
```

---

## IVR (Instruct-Validate-Repair)

A core generative programming pattern in Mellea:

1. **Instruct** — call the LLM with a prompt.
2. **Validate** — check the output against a `Requirement`.
3. **Repair** — if validation fails, retry or fix the output.

See: [Instruct, Validate, Repair](../concepts/instruct-validate-repair)

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

See: [MObjects and mify](../concepts/mobjects-and-mify)

---

## MObject

An **MObject** is a Python class decorated with `@mify`. It wraps existing data
objects so they can be queried and transformed by the LLM via `m.query()` and
`m.transform()`. Unlike `@generative`, `@mify` does not change the class's Python
interface — it adds a layer that the LLM can see and call.

See: [MObjects and mify](../concepts/mobjects-and-mify)

---

## ModelOption

An enum (`mellea.backends.ModelOption`) of backend-agnostic inference options:
`TEMPERATURE`, `SEED`, `MAX_NEW_TOKENS`, `SYSTEM_PROMPT`, etc. Using `ModelOption`
keys ensures the same options work across all backends.

```python
from mellea.backends import ModelOption
```

See: [Configure Model Options](../how-to/configure-model-options)

---

## ModelOutputThunk

The return type of `m.instruct()`, `m.act()`, and most session-level generative
calls. It wraps the model's raw output and an optional parsed representation typed
to your output schema (accessible via `.result`).

The value is computed lazily — the underlying inference call may not have completed
when the thunk is returned. Accessing `.value` blocks until the result is ready.
For async code, use `await thunk.avalue()` to await completion, or
`await thunk.astream()` to consume output chunk by chunk as it arrives.

You can also call `str(thunk)` to get the raw string output directly.

Use `thunk.is_computed()` to check whether the value has already been filled
without triggering evaluation.

---

## Plugin

A **Plugin** is a class-based extension point in Mellea that groups multiple
hooks sharing instance state. Inherit from `Plugin` and set `name` and `priority`
as class keyword arguments. Decorate methods with `@hook(HookType.XXX)` to
subscribe to pipeline events.

Use standalone `@hook` functions for single-concern hooks. Use `Plugin` subclasses
when multiple hooks need shared state (e.g., a redaction counter or rate limiter).

```python
from mellea.plugins import Plugin, hook, HookType

class MyPlugin(Plugin, name="my-plugin", priority=10):
    @hook(HookType.GENERATION_PRE_CALL)
    async def before_llm(self, payload, ctx):
        ...
```

See: [Plugins & Hooks](../concepts/plugins)

---

## PluginSet

A **PluginSet** groups related hooks and `Plugin` instances into a reusable,
named bundle. Use it to organize plugins by concern (security, observability,
compliance) and register or scope them as a unit. A `PluginSet` accepts any mix
of standalone `@hook` functions, `Plugin` instances, or nested `PluginSet`s.

```python
from mellea.plugins import PluginSet

security = PluginSet("security", [hook_a, hook_b, plugin_instance])
```

See: [Plugins & Hooks](../concepts/plugins)

---

## PreconditionException

Raised when a requirement attached to a `@generative` function's input arguments
fails — i.e., before the LLM call is made. Catch it to handle pre-call validation
failures gracefully.

```python
from mellea.stdlib.components.genstub import PreconditionException

try:
    result = my_generative_fn(m, ...)
except PreconditionException as e:
    print(e.validation)  # list of ValidationResult
```

See: [Handling Exceptions and Failures](../how-to/handling-exceptions)

---

## Purple elephant effect

The tendency for a model to produce the very thing you instructed it to avoid,
because the instruction draws attention to it. Named after the cognitive phenomenon:
"Don't think about a purple elephant" — and now you are.

In Mellea, avoid it by using `check()` instead of `req()` for negative constraints.
`check()` validates the output without including the constraint description in the
generation prompt:

```python
from mellea.stdlib.requirements import req, check

requirements=[
    req("Mention key features."),                        # model is told this
    check("Must not use the phrase 'industry-leading'"), # model is not told this
]
```

See: [Evaluate with LLM-as-a-Judge](../how-to/evaluate-with-llm-as-a-judge)

---

## ReAct

**Reason + Act** — a goal-driven agentic loop where the LLM alternates between
reasoning about the next step and calling a tool, repeating until the goal is
achieved. Mellea provides `mellea.stdlib.frameworks.react.react()` as a built-in
async implementation:

```python
from mellea.stdlib.frameworks.react import react
result, _ = await react(goal="...", context=ChatContext(), backend=m.backend, tools=[...])
```

See: [Tools and Agents](./tools-and-agents)

---

## Requirement

A `Requirement` is a validation constraint applied to a generative function's
output. Requirements can be programmatic (lambda, regex, type check) or generative
(another LLM call). Used in the IVR pattern.

`req()` and `check()` are the common shorthand constructors from `mellea.stdlib.requirements`:

- **`req(description)`** — creates a `Requirement` whose description is included in the prompt,
  so the model knows to aim for it.
- **`check(description)`** — creates a check-only `Requirement` whose description is
  *not* included in the prompt (avoids the "purple elephant effect" — mentioning a
  forbidden thing often makes the model produce it).
- **`simple_validate(fn)`** — wraps a lambda or function into a `validation_fn`,
  bypassing LLM-as-a-judge for fast deterministic checks.
- **`PythonExecutionReq`** — verifies that Python code in the LLM's output runs
  without raising an exception. Import from `mellea.stdlib.requirements.python_reqs`.
  Accepts `timeout`, `allowed_imports`, and `use_sandbox` (Docker-based isolation).

See: [Requirements System](../concepts/requirements-system)

---

## RichDocument

A `RichDocument` wraps a [Docling](https://docling-project.github.io/docling/) parsed document
to make PDFs, tables, and structured files queryable by the LLM. Extract tables as
`Table` objects and pass them directly to `m.transform()` or `m.query()`.

```bash
pip install 'mellea[docling]'
```

See: [Working with Data](./working-with-data)

---

## SimpleLRUCache

An LRU (least-recently-used) cache for storing `DynamicCache` KV blocks in
`LocalHFBackend`. Pass one at construction time to enable prefix caching:

```python
from mellea.backends.cache import SimpleLRUCache

backend = LocalHFBackend(
    model_id="ibm-granite/granite-3.3-2b-instruct",
    cache=SimpleLRUCache(capacity=5),
)
```

When the cache reaches `capacity`, the least recently used block is evicted and
its GPU memory freed. Choose capacity based on available VRAM and block size —
1–3 for large documents, up to 10 for small reused fragments.

See: [Prefix Caching and KV Blocks](../advanced/prefix-caching-and-kv-blocks)

---

## SimpleContext

A stateless context where each call is independent — no conversation history is
accumulated or sent to the backend. Use it for single-shot tasks where prior turns
are irrelevant.

```python
from mellea.stdlib.context import SimpleContext
ctx = SimpleContext()
```

For multi-turn conversations, use `ChatContext` instead.

See: [Context and Sessions](../concepts/context-and-sessions)

---

## Sampling strategy

A `SamplingStrategy` controls how the IVR loop behaves when a requirement fails.
Mellea's built-in strategies:

| Strategy | Behaviour |
| --- | --- |
| `RejectionSamplingStrategy` | Retry up to `loop_budget` times; return first passing result |
| `RepairTemplateStrategy` | Like rejection sampling but appends failure reasons to the original instruction |
| `MultiTurnStrategy` | Add validation failures as a new chat turn; model revises its previous attempt |
| `MajorityVotingStrategyForMath` | Generate N candidates; return the one supported by most (math expressions) |
| `MBRDRougeLStrategy` | Minimum Bayes Risk decoding using ROUGE-L; best for text generation tasks |
| `SOFAISamplingStrategy` | Fast System-1 generation verified by a slower System-2 model |
| `BudgetForcingSamplingStrategy` | Inject thinking tokens to expand reasoning budget |
| `BaseSamplingStrategy` | Abstract base; extend to implement custom repair and selection logic |

See: [Inference-Time Scaling](../advanced/inference-time-scaling)

---

## SamplingResult

The return type of session calls made with `return_sampling_results=True`, and of
the `serve()` function used with `m serve`. Holds `.result` (the selected output),
`.success` (whether a requirement was met), and `.sample_generations` (all
candidates generated).

---

## Table

An `MObject` wrapping a single table extracted from a `RichDocument`. Supports
`m.query()` and `m.transform()` directly, plus `.to_markdown()` and `.transpose()`.

```python
tables = rich_doc.get_tables()
summary = m.query(tables[0], "What is the total in the last row?")
```

See: [Working with Data](./working-with-data)

---

## TestBasedEval

A `Component` in `mellea.stdlib.components.unit_test_eval` that formats an
LLM-as-a-judge evaluation task for structured test cases loaded from JSON. Use it
in offline evaluation pipelines to verify model behaviour against a set of
input/target pairs.

```python
from mellea.stdlib.components.unit_test_eval import TestBasedEval

test_evals = TestBasedEval.from_json_file("tests/eval_data/cases.json")
for eval_case in test_evals:
    verdict = judge_session.instruct(eval_case)
```

See: [Unit Test Generative Code](../how-to/unit-test-generative-code)

---

## TemplateFormatter

A `ChatFormatter` subclass that renders prompts using Jinja2 templates instead of
the default chat-message format. Use it when you need precise control over how
components are serialised into the final prompt string. Configured per-backend.

See: [Template Formatting](../advanced/template-formatting)

---

## TemplateRepresentation

The data class a `Component` returns from `format_for_llm()` to describe itself to
the `TemplateFormatter`. It carries the component's template string, named
arguments, tool definitions, and field list — everything the formatter needs to
render the component into a prompt fragment.

See: [Mellea Core Internals](../advanced/mellea-core-internals)

---

## SOFAI

**SOFAI** (System-1 / System-2 AI) is a sampling strategy in Mellea that mirrors
dual-process cognition: a fast "System 1" model generates candidates and a slower
"System 2" model verifies them. Uses `SOFAISamplingStrategy`.

See: [Inference-Time Scaling](../advanced/inference-time-scaling)

---

## Tool

A Python function decorated with `@tool` (or registered via `MelleaSession`) that
Mellea exposes to an LLM for function calling. Tools have typed inputs and outputs
so the LLM can call them reliably without free-form parsing.

See: [Tools and Agents](./tools-and-agents)

---

## ValidationResult

The return type of a custom verifier function. Holds a boolean `result` (pass/fail)
and optional metadata — `reason` (string explanation), `score` (float), and
`thunk` (the raw `ModelOutputThunk` if the verifier used an LLM call internally).

```python
from mellea.core.requirement import ValidationResult

def my_verifier(output: str) -> ValidationResult:
    passed = len(output.split()) < 50
    return ValidationResult(passed, reason="Too long" if not passed else None)
```

See: [Write Custom Verifiers](../how-to/write-custom-verifiers)

---

## Thunk

See [ModelOutputThunk](#modeloutputthunk).

---

## wait_for_all_mots

A helper from `mellea.helpers.async_helpers` that concurrently resolves a list
of [`ModelOutputThunk`](#modeloutputthunk) objects. All thunks in the list are
awaited in parallel; the call returns when every thunk has been computed.

```python
from mellea.helpers.async_helpers import wait_for_all_mots

thunks = [await m.ainstruct(...) for _ in items]
await wait_for_all_mots(thunks)
# All thunks are now resolved — access .value on each.
```

Total wall-clock time is roughly the latency of the slowest single call rather
than the sum of all calls. Use `SimpleContext` (the default) when calling
`wait_for_all_mots`; concurrent writes to `ChatContext` can corrupt state.

See: [Tutorial 02: Streaming and Async](../tutorials/02-streaming-and-async)
