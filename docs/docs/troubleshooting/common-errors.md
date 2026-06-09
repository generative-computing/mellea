---
title: "Common Errors"
description: "Common errors, diagnostic steps, and fixes for Mellea programs."
# diataxis: reference
---

## Installation

### `granite4.1:3b` not found

```text
Error: model "granite4.1:3b" not found
```

Pull the model before running:

```bash
ollama pull granite4.1:3b
```

### Python 3.13: `outlines` install failure

```text
error: could not compile `outlines-core`
```

`outlines` requires a Rust compiler. Either [install Rust](https://www.rust-lang.org/tools/install)
or pin Python to 3.12:

```bash
uv python pin 3.12
uv add mellea
```

### Intel Mac: `torch` errors

Create a Conda environment, install `torchvision`, then install Mellea inside it:

```bash
conda create -n mellea python=3.12
conda activate mellea
conda install 'torchvision>=0.22.0'
uv pip install mellea
```

### Missing optional dependency

```text
ImportError: The 'hf' backend requires extra dependencies.
Please install them with: pip install 'mellea[hf]'
```

Each backend has an optional extras group. Install what you need:

```bash
pip install "mellea[hf]"         # HuggingFace / local inference
pip install "mellea[litellm]"    # LiteLLM multi-provider
pip install "mellea[watsonx]"    # IBM WatsonX
pip install "mellea[tools]"      # Tool / agent dependencies
pip install "mellea[telemetry]"  # OpenTelemetry tracing + metrics
```

---

## Ollama connectivity

### Connection refused

```text
ConnectionError: Could not connect to Ollama at http://localhost:11434
```

Ollama is not running. Start it:

```bash
ollama serve
```

Then verify it is reachable:

```bash
curl http://localhost:11434/api/version
```

### Wrong Ollama URL

If Ollama is running on a non-default host or port, pass the URL explicitly:

```python
from mellea.backends.ollama import OllamaModelBackend

m = MelleaSession(OllamaModelBackend(base_url="http://my-ollama-host:11434"))
```

---

## Requirements and sampling

### Requirements always failing — output looks fine

If the model keeps retrying but the output looks correct, the validation function
may be too strict. Inspect what is being rejected:

```python
from mellea import start_session
from mellea.stdlib.requirements import req

m = start_session()
result = m.instruct(
    "Write a haiku.",
    requirements=[req("Must be exactly 17 syllables")],
    return_sampling_results=True,
)

print(f"Success: {result.success}")
for i, (generation, validations) in enumerate(
    zip(result.sample_generations, result.sample_validations)
):
    print(f"\nAttempt {i + 1}:")
    print(f"  Output: {generation.value}")
    for requirement, validation in validations:
        print(f"  {requirement.description}: {validation._result} — {validation._reason}")
```

`return_sampling_results=True` makes `instruct()` return a `SamplingResult` instead
of a `ModelOutputThunk`. Use `result.success` to check whether the budget was
exhausted without a passing output.

### Budget exhausted — `result.success` is `False`

The model failed all `loop_budget` attempts. Options:

- Increase `loop_budget`:

  ```python
  from mellea.stdlib.sampling import RejectionSamplingStrategy

  strategy = RejectionSamplingStrategy(loop_budget=5)
  result = m.instruct("...", requirements=[...], strategy=strategy)
  ```

- Simplify or relax the requirement.
- Provide a more specific validation function that gives the model useful feedback via
  `ValidationResult.reason` — the reason string is passed back to the model on retry.
- Switch to `SOFAISamplingStrategy` to escalate to a stronger model when the primary
  model fails.

### `PreconditionException` from `@generative`

```text
mellea.stdlib.components.genstub.PreconditionException
```

A precondition check in a `@generative` function failed before generation. This is
intentional — the function declared that its inputs do not meet a precondition.
Check the function's `@precondition` decorators and validate your inputs before calling.

---

## Agents and tools

### `react()` raises `RuntimeError`

```text
RuntimeError: could not complete react loop in N iterations
```

The ReACT loop exhausted its `loop_budget` without finding a final answer. Either
increase the budget or check that the tool functions are returning the information
the model needs to reach a conclusion.

### Tool not called / wrong tool called

If the model is not calling tools as expected:

- Verify `ModelOption.TOOLS` is set in the session's model options.
- Check the tool's docstring — the model uses it to decide when to call the tool.
  A vague or absent docstring leads to poor tool selection.
- Use `guardian_check(context, backend, criteria="function_call")` from the
  [Guardian Intrinsics](../how-to/safety-guardrails) to detect function call
  hallucinations.

---

## Async

### `RuntimeError: no running event loop`

```text
RuntimeError: no running event loop
```

You are calling a synchronous Mellea method from inside an async function.
Switch to the async method (`ainstruct`, `achat`, `aact`) or wrap in `asyncio.run()`
if you are at the top level.

### `asyncio.run()` inside a Jupyter notebook

Jupyter notebooks already run an event loop. Use `await` directly or install
`nest_asyncio`:

```bash
pip install nest_asyncio
```

```python
import nest_asyncio
nest_asyncio.apply()
```

---

## Guardian / safety validation

Guardian Intrinsics (`guardian_check()`, `policy_guardrails()`,
`factuality_detection()`, `factuality_correction()`) require `LocalHFBackend`
with an IBM Granite model.
See [Safety Guardrails](../how-to/safety-guardrails) for full usage.

### `guardian_check()` returns unexpected scores

- Double-check the `criteria` argument — use a key from `CRITERIA_BANK` (e.g.
  `"harm"`, `"groundedness"`) or a free-text criteria string.
- For groundedness checks, attach source documents via `documents=[Document(...)]`
  on the `Message("assistant", ...)` in the evaluation context — not as a separate
  user message.
- Scores below `0.5` are safe; at or above `0.5` indicates risk detected.

### Deprecated `GuardianCheck` warnings

```text
DeprecationWarning: GuardianCheck is deprecated as of version 0.4.
Use the Guardian Intrinsics instead
```

Replace `GuardianCheck` / `GuardianRisk` imports with the Guardian Intrinsics API.
See [Safety Guardrails](../how-to/safety-guardrails) for migration guidance.

---

## Getting more help

- **GitHub Issues:** [github.com/generative-computing/mellea/issues](https://github.com/generative-computing/mellea/issues)
- **Examples:** [`docs/examples/`](https://github.com/generative-computing/mellea/tree/main/docs/examples)
- Enable telemetry to inspect what is happening at each step — see
  [Telemetry](../observability/telemetry).

---

**See also:**
[Quick Start](../getting-started/quickstart) |
[Inference-Time Scaling](../advanced/inference-time-scaling) |
[Safety Guardrails](../how-to/safety-guardrails)
