# Hook System — Known Bugs and Analysis Findings

This document records bugs and architectural concerns identified during a code review of the
Mellea plugin hook system (`mellea/plugins/`). All findings reference the hook system design
spec (`docs/dev/hook_system.md`) and implementation plan
(`docs/dev/hook_system_implementation_plan.md`).

Each finding includes: description, affected files, severity, and a proposed remediation.

---

## A1 — `GenerationPreCallPayload.formatted_prompt` is always `""` (HIGH)

### Description

`GenerationPreCallPayload.formatted_prompt` is always an empty string when the
`generation_pre_call` hook fires. The hook fires inside
`Backend.generate_from_context_with_hooks()` **before** calling
`generate_from_context()`, but prompt linearisation (converting the context and action
into the actual string/message list sent to the LLM API) happens **inside**
`generate_from_context()` within each backend implementation.

The design spec lists `formatted_prompt` as both observable (plugins inspect it) and
writable (plugins override it). In the current implementation it is neither — no value
is ever observed, and no backend reads the modified value back from the post-hook payload.

### Affected files

- `mellea/core/backend.py` — line ~127: `formatted_prompt=""` hardcoded

### Severity

**HIGH** — Any plugin registered on `generation_pre_call` that tries to inspect
(`payload.formatted_prompt`) or modify the prompt before it is sent receives and acts on
an empty string. This silently breaks prompt-inspection use cases (token estimation,
content filtering) without any error.

### Remediation options

**Option A — Override-only field (minimal change)**

Accept that `formatted_prompt` starts as `""` and is **write-only** at pre-call time.
Update the field docstring in `GenerationPreCallPayload` to state:
> "At generation_pre_call time this field is always an empty string; prompt linearisation
> has not yet occurred. Plugins may write a replacement prompt here, but the backend must
> be updated to read it back."

Also add the missing `estimated_tokens: int | None = None` field to the payload (present
in the design spec, absent from implementation).

**Option B — Expose linearisation in the Backend ABC (recommended long-term)**

Add an abstract `format_prompt(action, ctx, **kwargs) -> str | list[dict]` method to the
`Backend` ABC. Each backend already performs linearisation internally; this just surfaces
it. `generate_from_context_with_hooks()` would call `format_prompt()` before firing the
hook, populating `formatted_prompt` with the real value.

This change also enables Option B to close **A2** partially (see below).

---

## A2 — `GenerationPostCallPayload` fields are mostly unpopulated (MEDIUM)

### Description

The `GenerationPostCallPayload` class defines six descriptive fields per the design spec:
`prompt`, `raw_response`, `processed_output`, `token_usage`, `finish_reason`, and
`model_output`. The `generation_post_call` hook invocation in
`Backend.generate_from_context_with_hooks()` only sets two of them:

```python
# current (backend.py ~line 150)
post_payload = GenerationPostCallPayload(
    model_output=out_result,
    latency_ms=int((time.monotonic() - t0) * 1000),
)
```

All other fields default to `""`, `{}`, `0`, or `None`. Plugins that subscribe to
`generation_post_call` for output filtering, PII detection, or audit logging receive no
useful data.

### Affected files

- `mellea/core/backend.py` — lines ~150–152

### Severity

**MEDIUM** — The hook fires, but is largely useless for the use-cases it was designed
for. No error is raised.

### Remediation

**Partial fix available now**: Every backend sets `out_result._generate_log.prompt` to
the actual formatted prompt before returning from `generate_from_context()`. The
post-call invocation can be updated to:

```python
glog = getattr(out_result, "_generate_log", None)
post_payload = GenerationPostCallPayload(
    prompt=glog.prompt if glog else "",
    model_output=out_result,
    latency_ms=int((time.monotonic() - t0) * 1000),
)
```

This restores the `prompt` field. The remaining fields (`raw_response`, `token_usage`,
`finish_reason`) require the Backend ABC to return response metadata alongside the
`ModelOutputThunk`. A `BackendResponseMeta` dataclass (or extension of `GenerateLog`)
would be the clean vehicle for this, but is a larger API change.

---

## A3 — `SAMPLING_LOOP_END` failure path: context argument inconsistency (FIXED)

### Description

The `sampling_loop_end` hook is fired from two places in
`BaseSamplingStrategy.sample()`:

| Path | `invoke_hook` context arg | payload `final_context` field |
|------|--------------------------|-------------------------------|
| Success (~line 270) | `context=result_ctx` (context after final generation) | `final_context=result_ctx` |
| Failure (~line 363) | `context=context` (the **original** input context) | `final_context=None` (not set) |

A plugin receiving `ctx.global_context.state["context"]` sees a different object
depending on whether sampling succeeded or failed. This inconsistency makes it impossible
to write a single symmetric handler for both outcomes.

### Affected files

- `mellea/stdlib/sampling/base.py` — lines ~347–364

### Severity

**MEDIUM** — Functional bug. The hook fires correctly but with misleading context. Plugins
that inspect the current context state will observe the original (pre-loop) context on
failure instead of the last-iteration context.

### Remediation

Use the context from the `best_failed_index` iteration on the failure path. The
`sample_contexts` list is populated during the loop:

```python
# sampling/base.py — failure path
_final_ctx = sample_contexts[best_failed_index] if sample_contexts else context
end_payload = SamplingLoopEndPayload(
    success=False,
    iterations_used=loop_count,
    final_result=sampled_results[best_failed_index],
    final_action=sampled_actions[best_failed_index],
    final_context=_final_ctx,               # was: None
    failure_reason=f"Budget exhausted after {loop_count} iterations",
    all_results=sampled_results,
    all_validations=sampled_scores,
)
await invoke_hook(
    HookType.SAMPLING_LOOP_END,
    end_payload,
    backend=backend,
    context=_final_ctx,                     # was: context (original)
)
```

---

## A4 — Seven hook types are declared but unimplemented (MEDIUM)

### Description

The `HookType` enum in `mellea/plugins/types.py` declares seven hook types that have no
corresponding implementation anywhere in the codebase:

| Hook type | Expected file | Status |
|-----------|--------------|--------|
| `ADAPTER_PRE_LOAD` | `mellea/plugins/hooks/adapter.py` | Missing |
| `ADAPTER_POST_LOAD` | `mellea/plugins/hooks/adapter.py` | Missing |
| `ADAPTER_PRE_UNLOAD` | `mellea/plugins/hooks/adapter.py` | Missing |
| `ADAPTER_POST_UNLOAD` | `mellea/plugins/hooks/adapter.py` | Missing |
| `CONTEXT_UPDATE` | `mellea/plugins/hooks/context_ops.py` | Missing |
| `CONTEXT_PRUNE` | `mellea/plugins/hooks/context_ops.py` | Missing |
| `ERROR_OCCURRED` | `mellea/plugins/hooks/error.py` | Missing |

Consequences:
1. `_build_hook_registry()` does not include these types → they are never registered with
   ContextForge's `HookRegistry`.
2. No `invoke_hook(...)` call sites exist for them anywhere in the codebase.
3. Plugins registered on these hook types will never fire.

There is no call-time error; the hooks are silently absent.

### Affected files

- `mellea/plugins/types.py` — `_build_hook_registry()` omits these 7 types
- `mellea/plugins/hooks/` — three files missing
- `mellea/backends/openai.py`, `mellea/backends/huggingface.py` — no adapter hooks
- `mellea/stdlib/context.py` — no context hooks
- `mellea/stdlib/functional.py` — no error hook

### Severity

**MEDIUM** — These are fully unimplemented features. A plugin author consulting the
`HookType` enum would believe these hooks exist and function normally.

### Remediation

1. Create `mellea/plugins/hooks/adapter.py`, `context_ops.py`, `error.py` with payload
   classes as specified in `hook_system.md` sections G, H, I.
2. Add all seven types to `_build_hook_registry()` in `types.py`.
3. Add `invoke_hook(...)` call sites in:
   - `mellea/backends/openai.py` and `mellea/backends/huggingface.py` for adapter hooks
   - `mellea/stdlib/context.py` for context hooks
   - `mellea/stdlib/functional.py` (utility function `fire_error_hook`) for error hook

---

## A5 — Memory retention risk from live object references in payloads (ARCHITECTURAL)

### Description

All Mellea hook payloads are frozen Pydantic models with `arbitrary_types_allowed=True`,
allowing them to store live Python objects (not copies). This is intentional — in-process
plugins need direct access to `Backend`, `Context`, `Component`, and `ModelOutputThunk`
instances to be useful.

The risk arises when a **buggy plugin retains a reference to the payload** beyond the
hook call (e.g., appends it to a module-level list for deferred processing or logging).
The following payloads carry particularly heavy object graphs:

| Payload | Field | Object held | Retention risk |
|---------|-------|-------------|----------------|
| `SamplingLoopEndPayload` | `all_results` | `list[ModelOutputThunk]` — ALL intermediate generations | **HIGH** — up to `loop_budget` ModelOutputThunks, each holding generation state |
| `SamplingLoopEndPayload` | `all_validations` | `list[list[tuple[Requirement, ValidationResult]]]` | MEDIUM — all validation results across all iterations |
| `ComponentPostSuccessPayload` | `context_before`, `context_after` | Two full `Context` snapshots | HIGH — contexts hold the entire conversation history |
| `SamplingRepairPayload` | `failed_result`, `repair_action`, `repair_context` | `ModelOutputThunk`, `Component`, `Context` | MEDIUM — duplicates already in the sampling loop |
| `ToolPreInvokePayload` | `tool_callable` | The actual Python `Callable` | LOW — function objects are small, but keeps the module alive |

None of these are bugs in normal usage. The concern is specifically that
`fire_and_forget` hooks are dispatched as background tasks and may outlive the current
generation cycle. If such a hook stores the payload, it prevents garbage collection of
the objects for an indefinite period.

### Affected files

- `mellea/plugins/hooks/sampling.py` — `SamplingLoopEndPayload`, `SamplingRepairPayload`
- `mellea/plugins/hooks/component.py` — `ComponentPostSuccessPayload`
- `mellea/plugins/hooks/tool.py` — `ToolPreInvokePayload`

### Severity

**ARCHITECTURAL** — Not a bug under correct plugin usage. Becomes a real problem only
when plugins misbehave by storing payloads.

### Remediation options

**Option A — Document and trust plugin authors**: Add prominent docstrings to the
heaviest payloads (`SamplingLoopEndPayload`, `ComponentPostSuccessPayload`) warning that
retaining the payload beyond the hook call will prevent GC of the referenced objects.

**Option B — Provide `slim()` method**: Add a `slim()` method to heavy payload classes
that returns a lightweight copy with live references replaced by identifiers (e.g.,
`context_id: str`, `result_id: str` or similar). Fire-and-forget plugins would call
`payload.slim()` before storing.

**Option C — Separate observability payloads**: For hooks with both modify and observe
use cases, provide two payload types: a full payload for in-process enforce/permissive
hooks, and a serializable summary payload for fire-and-forget hooks. This is a larger
design change aligned with the spec's "Serializable" principle (Section 1, point 5).

**Immediate recommendation**: Implement Option A (documentation). Options B and C can
be pursued if memory pressure becomes observable in production.
