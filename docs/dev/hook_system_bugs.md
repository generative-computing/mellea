# Hook System Known Issues & Bug Tracker

## Status legend
- **Open** — confirmed in code, not yet addressed
- **Fixed** — resolved; commit or PR noted where applicable
- **Closed (not a bug)** — investigation showed no defect

---

## A1 — `GenerationPreCallPayload.formatted_prompt` always `""`

**Status: Closed (not a bug)**

The original report was incorrect. `GenerationPreCallPayload` has no `formatted_prompt`
field — the prompt has not been formatted yet at the pre-call stage, so the field was
intentionally omitted. The post-call payload (`GenerationPostCallPayload.prompt`) is
populated correctly from `_generate_log.prompt`.

---

## A2 — `GenerationPostCallPayload` mostly empty

**Status: Fixed**

All three fields (`prompt`, `model_output`, `latency_ms`) are populated at the call site
in `backend.py`. `prompt` is extracted from `mot._generate_log.prompt`; `model_output` is
the fully-computed `ModelOutputThunk`; `latency_ms` is the elapsed wall-clock time.

---

## A3 — `SAMPLING_LOOP_END` failure path passed original context

**Status: Fixed**

The failure path now uses `sample_contexts[best_failed_index]` instead of the original
context.

---

## A4 — HookType stubs with no payload or call site

**Status: Fixed**

The 7 unimplemented hook types (`ADAPTER_PRE_LOAD`, `ADAPTER_POST_LOAD`,
`ADAPTER_PRE_UNLOAD`, `ADAPTER_POST_UNLOAD`, `CONTEXT_UPDATE`, `CONTEXT_PRUNE`,
`ERROR_OCCURRED`) were removed from the `HookType` enum. They can be re-added when
the corresponding call sites and payload classes are implemented.

---

## A5 — Payloads hold live object references

**Status: Fixed**

Payload fields that hold live framework-owned objects (`Context`, `Component`,
`MelleaSession`, `SamplingStrategy`, `CBlock`) are now typed as `WeakProxy`
(defined in `mellea/plugins/base.py`). `WeakProxy` is a Pydantic `Annotated` type
that wraps the value in `weakref.proxy()` at construction time.

**Behavior**: Cached payloads no longer prevent GC of framework objects. Accessing
a `WeakProxy` field after the referent is garbage-collected raises `ReferenceError`.
This is intentional — plugins must not cache payloads for use beyond the hook
invocation.

**Fields that remain strong references** (plugins may legitimately retain these):
`model_output`, `result`, `final_result`, `generate_log`, `sampling_results`,
`all_results`, `error`, `tool_output`, `tool_message`, `model_tool_call`,
`validation_results`, `results`.

---

## Issue 1 — `"tools"` typo in `generation_pre_call` policy

**Status: Fixed**

`policies.py` had `"tools"` in the `writable_fields` frozenset for `generation_pre_call`,
but the actual payload field is named `tool_calls`. The typo meant the policy never
accepted plugin modifications to `tool_calls`. Fixed by changing `"tools"` to `"tool_calls"`
in `MELLEA_HOOK_PAYLOAD_POLICIES`.

---

## Issue 2 — Observe-only hooks accept all plugin modifications

**Status: Fixed**

`ensure_plugin_manager()` and `initialize_plugins()` in `manager.py` now pass
`default_hook_policy=DEFAULT_HOOK_POLICY` (where `DEFAULT_HOOK_POLICY = "deny"`)
directly to the `PluginManager` constructor. The cpex framework uses this argument
to reject all plugin-proposed modifications on hooks that are not present in
`MELLEA_HOOK_PAYLOAD_POLICIES`.

---

## Issue 3 — `_ClassPluginAdapter` ignored per-method execution modes

**Status: Fixed**

`_ClassPluginAdapter` was replaced by `_MethodHookAdapter`. Each `@hook`-decorated
method on a `@plugin` class is now registered as its own `_MethodHookAdapter` instance,
carrying the mode (`SEQUENTIAL`, `FIRE_AND_FORGET`, etc.) declared on that specific
`@hook` decorator. Adapter names follow the pattern `"<plugin_name>.<hook_type>"`.

**Note**: `initialize()` and `shutdown()` delegate to the underlying class instance and
may be called once per registered hook method. Make them idempotent when a class defines
multiple `@hook` methods.

---

## Issue 4 — `PluginSet.flatten()` didn't propagate priority through nested sets

**Status: Fixed**

`PluginSet.flatten()` now applies `self.priority` to all items yielded by nested
`PluginSet.flatten()` calls when `self.priority is not None`.

---

## Issue 5 — `_FunctionHookAdapter` name collision across modules

**Status: Fixed**

`_FunctionHookAdapter` now uses `f"{fn.__module__}.{fn.__qualname__}"` as the
plugin name (previously only `fn.__qualname__`). `_unregister_single` was updated
to match. This eliminates collisions between same-named functions in different modules.

---

## Issue 6 — `FIRE_AND_FORGET` hooks block in sync entry points

**Status: Documented**

`start_session()`, `reset()`, and `cleanup()` in `mellea/stdlib/session.py` invoke hooks
via `_run_async_in_thread()`. This means `FIRE_AND_FORGET` hooks at these points do not
block their *own execution* (cpex dispatches them as background tasks), but the
thread-spawn overhead still adds a small amount of latency to the synchronous call.

This is expected behavior for sync entry points and does not require a code change.
See the hook system specification (`docs/dev/hook_system.md`) for details.
