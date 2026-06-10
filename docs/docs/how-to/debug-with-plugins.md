---
title: "Debug with Plugins"
description: "Use built-in debug plugins to trace generation, validation, and sampling behavior in detail."
# diataxis: how-to
---

**Prerequisites:** [The Requirements System](../concepts/requirements-system),
[Sampling Strategies](../concepts/sampling-strategies), `pip install mellea`.

Mellea's plugin system provides debug hooks that trace the full lifecycle of
generation, validation, and sampling. Use these plugins to understand:

- What prompts are sent to the LLM
- Model latency and token usage
- Which requirements pass/fail and why
- When repair strategies trigger and what feedback they provide
- End-to-end flow through the sampling loop

## Built-in debug plugins

Mellea ships with three categories of debug plugins in `mellea.plugins.builtin_debug`:

### Generation pipeline plugins

Trace all LLM backend calls with request/response inspection, latency, and tokens.

```python
from mellea.plugins.builtin_debug.generation import (
    log_generation_pre_call,
    log_generation_post_call,
)
from mellea.plugins import register

register([
    log_generation_pre_call,
    log_generation_post_call,
])
```

**Output:**

```text
[📤 GEN-PRE-CALL gen_id=abc123...] model=granite4.1:3b | prompt=Write a thank you note
[📥 GEN-POST-CALL gen_id=abc123...] model=granite4.1:3b | latency=397ms | tokens=(47+19=66) | response=hello there thank you...
```

**Logs:**

- Generation ID for correlation
- Model being called
- Request: prompt preview (first 100 chars)
- Response: preview, latency, token counts
- **Repair feedback** when present (shows guidance the model receives during repair)

### Validation pipeline plugins

Trace requirement validation with pre-check setup and per-requirement results.

```python
from mellea.plugins.builtin_debug.validation import (
    log_validation_pre_check,
    log_validation_post_check,
)
from mellea.plugins import register

register([
    log_validation_pre_check,
    log_validation_post_check,
])
```

**Output:**

```text
[🔍 VALIDATION-PRE-CHECK] requirements=3 | target=ModelOutputThunk

[❌ VALIDATION-POST-CHECK] MIXED RESULTS: 2/3 passed, 1/3 failed
    ✓ Use only lowercase letters
    ✓ Include the phrase 'thank you'
    ❌ Start with a greeting
       └─ validated as "no"
```

**Logs:**

- Pre-check: how many requirements, what's being validated
- Post-check: pass/fail count per requirement
- Per-requirement status with reasons for failures

### Sampling pipeline plugins

Trace the sampling strategy lifecycle including iterations, validation results,
and repair events.

```python
from mellea.plugins.builtin_debug.sampling import (
    log_sampling_loop_start,
    log_sampling_iteration,
    log_sampling_repair,
    log_sampling_loop_end,
)
from mellea.plugins import register

register([
    log_sampling_loop_start,
    log_sampling_iteration,
    log_sampling_repair,
    log_sampling_loop_end,
])
```

**Output:**

```text
[🎯 SAMPLING-START] strategy=RepairTemplateStrategy | loop_budget=3 | requirements=3

[❌ SAMPLING-ITER 1] FAILED: 2/3 validations passed
    ❌ Start with a greeting

[🔧 REPAIR-TRIGGERED] at iteration 1
   repair_type=template
   failed_validations:
     • Start with a greeting

[❌ SAMPLING-ITER 2] FAILED: 2/3 validations passed
    ❌ Start with a greeting

[🎉 SAMPLING-END] SUCCESS in 2 iteration(s) using RepairTemplateStrategy
   total_attempts=2
   best_validation_score=3/3
```

**Logs:**

- Loop start: strategy, budget, requirement count
- Each iteration: pass/fail count, failed requirement names
- Repair events: when triggered, repair type, failed requirements
- Loop end: success/failure, iterations used, final statistics

## Enabling multiple plugins together

Combine plugins for complete end-to-end visibility:

```python
from mellea.plugins.builtin_debug.generation import (
    log_generation_pre_call,
    log_generation_post_call,
)
from mellea.plugins.builtin_debug.validation import (
    log_validation_pre_check,
    log_validation_post_check,
)
from mellea.plugins.builtin_debug.sampling import (
    log_sampling_loop_start,
    log_sampling_iteration,
    log_sampling_repair,
    log_sampling_loop_end,
)
from mellea.plugins import register

register([
    # Generation hooks
    log_generation_pre_call,
    log_generation_post_call,
    # Validation hooks
    log_validation_pre_check,
    log_validation_post_check,
    # Sampling hooks
    log_sampling_loop_start,
    log_sampling_iteration,
    log_sampling_repair,
    log_sampling_loop_end,
])
```

This reveals the complete flow:

```text
[🎯 SAMPLING-START] strategy=... | loop_budget=... | requirements=...

[📤 GEN-PRE-CALL] prompt=...
[📥 GEN-POST-CALL] response=... | latency=... | tokens=...

[🔍 VALIDATION-PRE-CHECK] requirements=... | target=...
[📤 GEN-PRE-CALL] prompt=Start with a greeting (validation check)
[📥 GEN-POST-CALL] response=no
[❌ VALIDATION-POST-CHECK] MIXED RESULTS: 2/3 passed, 1/3 failed

[❌ SAMPLING-ITER 1] FAILED: 2/3 validations passed

[🔧 REPAIR-TRIGGERED] at iteration 1
   failed_validations: Start with a greeting

[📤 GEN-PRE-CALL] prompt=Write a thank you note
   [⭐ REPAIR ATTEMPT] Repair feedback provided: ...
[📥 GEN-POST-CALL] response=... | latency=... | tokens=...

[🔍 VALIDATION-PRE-CHECK] requirements=... | target=...
[📤 GEN-PRE-CALL] prompt=Start with a greeting
[📥 GEN-POST-CALL] response=yes
[✅ VALIDATION-POST-CHECK] ALL PASSED: 3/3 requirements

[✅ SAMPLING-ITER 2] SUCCESS: 3/3 validations passed

[🎉 SAMPLING-END] SUCCESS in 2 iteration(s)
```

## Example scripts

Ready-to-run examples are available in `docs/examples/plugins/`:

| Script                            | Plugins                    | Purpose                             |
| --------------------------------- | -------------------------- | ----------------------------------- |
| `builtin_generation_tracing.py`   | Generation                 | Basic model call tracing            |
| `builtin_validation_tracing.py`   | Validation                 | Requirement validation              |
| `builtin_validation_failures.py`  | Validation                 | Show validation failures            |
| `builtin_sampling_diagnostics.py` | Sampling                   | Strategy iterations                 |
| `builtin_full_pipeline_tracing.py`| Generation + Sampling      | End-to-end with model visibility    |
| `builtin_complete_diagnostics.py` | All 3                      | Complete pipeline with validation   |

Run any example:

```bash
uv run python docs/examples/plugins/builtin_generation_tracing.py
uv run python docs/examples/plugins/builtin_validation_failures.py
uv run python docs/examples/plugins/builtin_complete_diagnostics.py
```

## Common debugging scenarios

### "Why is the model generating a different response than I expected?"

Enable **generation tracing** to see:

- Exactly what prompt was sent
- Model's latency and token usage
- Response preview
- When repair feedback is provided (if using RepairTemplateStrategy)

This shows whether the issue is in the prompt, model behavior, or repair strategy.

### "Why are my requirements failing?"

Enable **validation tracing** to see:

- Each requirement being checked
- Pass/fail status per requirement
- Failure reason (e.g., "validated as 'no'")
- Pass/fail counts

This pinpoints which requirements are problematic and why.

### "Why isn't the repair strategy helping?"

Enable **all three plugin categories** to see:

- Initial attempt (generation + validation)
- What failed (validation results)
- Repair feedback provided (in generation pre-call logs)
- Second attempt with feedback (generation + validation)
- Whether the repair improved the results

This reveals whether the repair strategy is receiving the right feedback and the model is responding appropriately.

### "Why is sampling taking so long?"

Enable **sampling tracing** to see:

- How many iterations ran
- Validation results per iteration
- When repairs were triggered
- Total attempts before success/failure

This identifies whether the issue is budget exhaustion, frequent failures, or ineffective repair.

## Controlling log output

By default, debug plugins log at INFO level for important events and DEBUG level
for details. Control verbosity:

```python
import logging

# Show only failures and key events
logging.basicConfig(level=logging.INFO)

# Show all details including passed requirements
logging.basicConfig(level=logging.DEBUG)

# Silence a specific logger
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("ollama").setLevel(logging.ERROR)
```

## Performance notes

Debug plugins have minimal overhead:

- Pre-hooks check whether plugins are registered before building payloads
- Logging is formatted efficiently
- No plugins fire in the hot path when not registered

For production use, you can safely leave plugins registered — they only log when
enabled. For maximum performance, simply don't register them.

## Next steps

- [Observability: Tracing](../observability/tracing.md) — export traces to Jaeger or Grafana
- [Handling Exceptions and Failures](./handling-exceptions.md) — work with sampling failures
- [The Requirements System](../concepts/requirements-system) — understand validation in depth
