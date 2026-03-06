---
title: "Inference-Time Scaling"
description: "Control how Mellea generates and validates outputs: rejection sampling, SOFAI, budget forcing, and majority voting."
# diataxis: how-to
---

**Prerequisites:** [Instruct, Validate, Repair](../concepts/instruct-validate-repair)
complete, `pip install mellea`, Ollama running locally.

A sampling strategy controls what happens after the first generation: whether to
retry on failure, how to repair output, and whether to escalate to a more powerful
model. You pass a strategy to `instruct()` via the `strategy` parameter.

## Rejection sampling

`RejectionSamplingStrategy` is the default. It generates once, validates all
requirements, and retries from scratch up to `loop_budget` times on failure:

```python
from mellea import start_session
from mellea.stdlib.requirements import req, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

m = start_session()
result = m.instruct(
    "Write a haiku about autumn.",
    requirements=[
        req(
            "The response must be exactly three lines.",
            validation_fn=simple_validate(lambda x: len(x.strip().splitlines()) == 3),
        ),
    ],
    strategy=RejectionSamplingStrategy(loop_budget=5),
    return_sampling_results=True,
)

if result.success:
    print(str(result.result))
else:
    print("All attempts failed. Best effort:")
    print(str(result.sample_generations[0].value))
# Output will vary — LLM responses depend on model and temperature.
```

With `return_sampling_results=True`, `instruct()` returns a `SamplingResult` with:

- `result.success` — whether any attempt passed all requirements
- `result.result` — the passing output (if any)
- `result.sample_generations` — all intermediate generations

Without `return_sampling_results=True`, `instruct()` returns a `ModelOutputThunk`
directly (the last generation, regardless of whether validation passed).

The default strategy when you don't pass `strategy` explicitly is
`RejectionSamplingStrategy(loop_budget=2)`.

## Validation feedback

The repair loop works best when failing requirements provide a reason. The
`ValidationResult.reason` string is included in the repair prompt sent to the model:

```python
import json
from mellea import start_session
from mellea.stdlib.requirements import ValidationResult, req
from mellea.stdlib.sampling import RejectionSamplingStrategy

def check_valid_json(ctx) -> ValidationResult:
    output = ctx.last_output()
    try:
        json.loads(str(output.value))
        return ValidationResult(True, reason="Valid JSON.")
    except json.JSONDecodeError as e:
        return ValidationResult(False, reason=f"Invalid JSON: {e}")

m = start_session()
result = m.instruct(
    "Return a JSON object with keys 'name' and 'score'.",
    requirements=[req("Output must be valid JSON.", validation_fn=check_valid_json)],
    strategy=RejectionSamplingStrategy(loop_budget=3),
    return_sampling_results=True,
)

if result.success:
    data = json.loads(str(result.result))
    print(data)
# Output will vary — LLM responses depend on model and temperature.
```

## SOFAI — dual-model escalation

> **Advanced:** SOFAI (Slow and Fast AI) uses two backends: S1 (fast, small) handles
> most cases; S2 (slower, larger) escalates when S1 exhausts its budget.

`SOFAISamplingStrategy` is useful when a fast local model handles easy inputs but
you need a more capable model for hard cases:

```python
import mellea
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements import ValidationResult, req
from mellea.stdlib.sampling import SOFAISamplingStrategy

def check_coloring(ctx) -> ValidationResult:
    """Validate a graph coloring solution."""
    output = ctx.last_output()
    # ... your validation logic ...
    if errors:
        return ValidationResult(False, reason=" | ".join(errors))
    return ValidationResult(True, reason="Valid coloring.")

requirements = [req("The coloring must be valid.", validation_fn=check_coloring)]

s1_backend = OllamaModelBackend(model_id="phi4-mini:latest")
s2_backend = OllamaModelBackend(model_id="llama3.1:8b")

sofai = SOFAISamplingStrategy(
    s1_solver_backend=s1_backend,
    s2_solver_backend=s2_backend,
    s2_solver_mode="fresh_start",
    loop_budget=3,
)

m = mellea.MelleaSession(backend=s1_backend, ctx=ChatContext())
result = m.instruct(
    "Color the graph nodes so no two adjacent nodes share a color: A-B, B-C, A-C.",
    requirements=requirements,
    strategy=sofai,
    return_sampling_results=True,
)

print(f"Success: {result.success}")
print(f"Attempts: {len(result.sample_generations)}")
# Output will vary — LLM responses depend on model and temperature.
```

`s2_solver_mode` controls how S2 starts when escalated:

| Mode | Behavior |
| ---- | -------- |
| `"fresh_start"` | S2 receives a clean context with no S1 history |
| `"continue_chat"` | S2 continues from S1's conversation history |
| `"best_attempt"` | S2 starts from S1's best attempt so far |

The `ValidationResult.reason` string is passed to both S1 and S2 as repair guidance —
write specific, actionable failure reasons for best results.

> **Full example:** [`docs/examples/sofai/sofai_graph_coloring.py`](../../examples/sofai/sofai_graph_coloring.py)

## Budget forcing

> **Advanced:** `BudgetForcingSamplingStrategy` controls thinking-token budgets for
> models that support extended reasoning (e.g., models with `<think>` tokens).

```python
from mellea import start_session
from mellea.stdlib.sampling.budget_forcing import BudgetForcingSamplingStrategy

strategy = BudgetForcingSamplingStrategy(
    loop_budget=3,
    think_max_tokens=1024,
    answer_max_tokens=256,
)

m = start_session()
result = m.instruct(
    "Solve: if a train travels 60 mph for 2.5 hours, how far does it travel?",
    strategy=strategy,
)
print(str(result))
# Output will vary — LLM responses depend on model and temperature.
```

> **Note:** `BudgetForcingSamplingStrategy` is not exported from
> `mellea.stdlib.sampling` directly — import from
> `mellea.stdlib.sampling.budget_forcing`. Token defaults are `think_max_tokens=4096`
> and `answer_max_tokens=None`. The strategy wraps `RejectionSamplingStrategy` so
> you can combine it with requirements and `loop_budget`.

## Majority voting

> **Advanced:** `MajorityVotingStrategyForMath` generates multiple independent
> answers and selects the most common one — useful for math and reasoning tasks where
> the correct answer should appear frequently across independent samples.

```python
from mellea import start_session
from mellea.stdlib.sampling.majority_voting import MajorityVotingStrategyForMath

strategy = MajorityVotingStrategyForMath(number_of_samples=5)

m = start_session()
result = m.instruct(
    "What is 17 × 23?",
    strategy=strategy,
    return_sampling_results=True,
)
print(str(result.result))
# Output will vary — LLM responses depend on model and temperature.
# Expected: 391
```

> **Note:** `MajorityVotingStrategyForMath` is designed for numeric math expressions
> (it normalises and compares parsed values). `MBRDRougeLStrategy` uses ROUGE-L
> scoring for text tasks — pass `number_of_samples` to control how many independent
> generations are compared. Neither is exported from `mellea.stdlib.sampling`
> directly — import from `mellea.stdlib.sampling.majority_voting`.

## Other built-in strategies

Two additional strategies are exported from `mellea.stdlib.sampling`:

**`RepairTemplateStrategy`** — like `RejectionSamplingStrategy` but appends
validation failure reasons to a copy of the original instruction rather than
retrying from a clean state. Use this when you want the repair prompt to include
the full original instruction plus a "what went wrong" addendum:

```python
from mellea import start_session
from mellea.stdlib.requirements import req, simple_validate
from mellea.stdlib.sampling import RepairTemplateStrategy

m = start_session()
result = m.instruct(
    "List three fruits, one per line.",
    requirements=[
        req(
            "Must contain exactly three lines.",
            validation_fn=simple_validate(
                lambda x: (len(x.strip().splitlines()) == 3, "Not exactly three lines.")
            ),
        )
    ],
    strategy=RepairTemplateStrategy(loop_budget=3),
)
print(str(result))
# Output will vary — LLM responses depend on model and temperature.
```

**`MultiTurnStrategy`** — multi-turn repair that adds validation failures as a
new chat turn rather than rewriting the original instruction. The model sees
its previous attempt in the context and is asked to revise it. Use with
`ChatContext` for agentic repair loops:

```python
from mellea import start_session
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements import req, simple_validate
from mellea.stdlib.sampling import MultiTurnStrategy

m = start_session(ctx=ChatContext())
result = m.instruct(
    "List three fruits, one per line.",
    requirements=[
        req(
            "Must contain exactly three lines.",
            validation_fn=simple_validate(
                lambda x: (len(x.strip().splitlines()) == 3, "Not exactly three lines.")
            ),
        )
    ],
    strategy=MultiTurnStrategy(loop_budget=3),
)
print(str(result))
# Output will vary — LLM responses depend on model and temperature.
```

## Building a custom strategy

Extend `BaseSamplingStrategy` to implement your own repair logic. You must
implement two static methods:

- `repair(old_ctx, new_ctx, past_actions, past_results, past_val)` — returns a
  `(Component, Context)` tuple for the next generation attempt.
- `select_from_failure(sampled_actions, sampled_results, sampled_val)` — returns
  the index of the best result when the budget is exhausted with no success.

```python
from mellea.stdlib.sampling import BaseSamplingStrategy
from mellea.core import Component, Context, ModelOutputThunk, ValidationResult
from mellea.stdlib.requirements import Requirement


class MyStrategy(BaseSamplingStrategy):
    @staticmethod
    def repair(old_ctx, new_ctx, past_actions, past_results, past_val):
        # Return the original action and context unchanged — equivalent to
        # plain rejection sampling.
        return past_actions[-1], old_ctx

    @staticmethod
    def select_from_failure(sampled_actions, sampled_results, sampled_val):
        # Return the last attempt as the fallback.
        return len(sampled_results) - 1
```

Pass your custom strategy to `instruct()` just like the built-in ones:

```python
from mellea import start_session

m = start_session()
result = m.instruct(
    "Describe a tree in one sentence.",
    strategy=MyStrategy(loop_budget=2),
)
print(str(result))
# Output will vary — LLM responses depend on model and temperature.
```
