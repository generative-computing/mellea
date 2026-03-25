# Mellea Plugin Hook System — Internal Design Notes

> **User-facing documentation:** [Plugins & Hooks](../../docs/docs/concepts/plugins.mdx) covers usage, registration, execution modes, hook types reference, and patterns. This file retains only internal design rationale and decisions for contributors.

---

## Design principles

1. **Consistent interface**: All hooks follow the same async pattern with payload and context parameters
2. **Composable**: Multiple plugins can register for the same hook, executing in priority order
3. **Fail-safe**: Hook failures can be handled gracefully without breaking core execution
4. **Minimal intrusion**: Plugins are opt-in; default Mellea behavior remains unchanged without plugins. Plugins work identically whether invoked through a session (`m.instruct(...)`) or via the functional API (`instruct(backend, context, ...)`)
5. **Architecturally aligned**: Hook categories reflect Mellea's true abstraction boundaries — Session lifecycle, Component lifecycle, and the (Backend, Context) generation pipeline
6. **Code-first**: Plugins are defined and composed in Python. The `@hook` decorator and `Plugin` base class are the primary registration mechanisms; YAML configuration is a secondary option for deployment-time overrides
7. **Functions-first**: The simplest plugin is a plain async function decorated with `@hook`. Class-based plugins (via the `Plugin` base class) exist for stateful, multi-hook scenarios but are not required

---

## Concurrency model

Hooks use Python's `async`/`await` cooperative multitasking. Because Python's event loop only switches execution at `await` points, hook code won't be interrupted mid-logic. This means:

- **Sequential when awaited**: Calling `await hook(...)` keeps control flow deterministic — the hook completes before the caller continues.
- **Race conditions only at `await` points**: Shared state is safe to read and write between `await` calls within a single hook. Races only arise if multiple hooks modify the same shared state and are dispatched concurrently.
- **No preemptive interruption**: Unlike threads, a hook handler runs uninterrupted until it yields control via `await`.

---

## Hook invocation responsibilities

Hooks are called from Mellea's base classes (`Component.aact()`, `Backend.generate()`, `SamplingStrategy.run()`, etc.). This means hook invocation is a framework-level concern, and authors of new backends, sampling strategies, or components do not need to manually insert hook calls.

The caller (the base class method) is responsible for both invoking the hook and processing the result. Processing means checking the result for one of three possible outcomes:

1. **Continue with original payload** — `PluginResult(continue_processing=True)` with no `modified_payload`. The caller proceeds unchanged.
2. **Continue with modified payload** — `PluginResult(continue_processing=True, modified_payload=...)`. The plugin manager applies the hook's payload policy, accepting only changes to writable fields and discarding unauthorized modifications. The caller uses the policy-filtered payload in place of the original.
3. **Block execution** — `PluginResult(continue_processing=False, violation=...)`. The caller raises or returns early with structured error information.

Hooks cannot redirect control flow, jump to arbitrary code, or alter the calling method's logic beyond these outcomes. This is enforced by the `PluginResult` type.

---

## Payload design principles

1. **Strongly typed** — Each hook has a dedicated payload dataclass (not a generic dict). This enables IDE autocompletion, static analysis, and clear documentation of what each hook receives.
2. **Sufficient (maximize-at-boundary)** — Each payload includes everything available at that point in time. Post-hooks include the pre-hook fields plus results. This avoids forcing plugins to maintain their own state across pre/post pairs.
3. **Frozen (immutable)** — Payloads are frozen Pydantic models (`model_config = ConfigDict(frozen=True)`). Plugins cannot mutate payload attributes in place. To propose changes, plugins must call `payload.model_copy(update={...})` and return the copy via `PluginResult.modified_payload`. This ensures every modification is explicit and flows through the policy system.
4. **Policy-controlled** — Each hook type declares a `HookPayloadPolicy` specifying which fields are writable. The plugin manager applies the policy after each plugin returns, accepting only changes to writable fields and silently discarding unauthorized modifications. This separates "what the plugin can observe" from "what the plugin can change" — and enforces it at the framework level.
5. **Serializable** — Payloads should be serializable for external (MCP-based) plugins that run out-of-process. All payload fields use types that can round-trip through JSON or similar formats.
6. **Versioned** — Payload schemas carry a `payload_version` so plugins can detect incompatible changes at registration time rather than at runtime.
7. **Isolation** — Each plugin receives a copy-on-write (CoW) snapshot of the payload. Mutable containers (dicts, lists) are wrapped so mutations in one plugin do not affect others. Plugins should not cache payloads beyond the hook invocation — payload fields reference live framework objects (`Context`, `Component`, `MelleaSession`) whose lifecycle is managed by the framework.

---

## GlobalContext design (ambient metadata)

The `GlobalContext` passed to hooks carries lightweight, cross-cutting ambient metadata that is useful to every hook regardless of type. Hook-specific data (context, session, action, etc.) belongs on the **typed payload**, not on the global context.

### What goes in GlobalContext

```python
# GlobalContext.state — same for all hook types
backend_name: str                  # Derived from backend.model_id (when backend is passed)
```

The `backend_name` is a lightweight string extracted from `backend.model_id`. The full `backend` and `session` objects are **not** stored in GlobalContext — this avoids giving plugins unchecked mutable access to core framework objects.

### Design rationale

Previously, `context`, `session`, and `backend` were passed both on payloads and in `GlobalContext.state`, creating duplication. The same mutable object accessible via two paths was a footgun — plugins could be confused about which to read/modify. The refactored design:

1. **Payloads** are the primary API surface — typed, documented, policy-controlled
2. **GlobalContext** holds only truly ambient metadata (`backend_name`) that doesn't belong on any specific payload
3. No mutable framework objects (`Backend`, `MelleaSession`, `Context`) are stored in GlobalContext

---

## Design decision: separate success/error hooks

`component_post_success` and `component_post_error` are separate hooks rather than a single `component_post` with a sum type over success/failure. The reasons are:

1. **Registration granularity** — Plugins subscribe to only what they need. An audit logger may only care about errors; a metrics collector may only care about successes.
2. **Distinct payload shapes** — Success payloads carry `result`, `generate_log`, and `sampling_results`; error payloads carry `exception`, `error_type`, and `stack_trace`. A sum type would force nullable fields or tagged unions, adding complexity for every consumer.
3. **Different execution modes** — Error hooks may be fire-and-forget (for alerting); success hooks may be blocking (for output transformation). Separate hooks allow per-hook execution timing configuration.

---

## Design decision: component_pre_create / component_post_create deferral

`component_pre_create` and `component_post_create` are not implemented. `Component` is currently a `Protocol`, not an abstract base class. This means Mellea has no ownership over component initialization: there are no guarantees about when or how subclass `__init__` methods run, and there is no single interception point that covers all `Component` implementations.

Placing hook calls inside `Instruction.__init__` and `Message.__init__` works for those specific classes, but it is fragile (any user-defined `Component` subclass is invisible to the hooks) and architecturally wrong (the hook system should not need to be threaded manually into every `__init__`).

If `Component` were refactored to an abstract base class, Mellea could wrap `__init__` at the ABC level and fire these hooks generically for all subclasses. Until then, use `component_pre_execute` for pre-execution policy enforcement.

---

## Unimplemented hooks

The following hooks are designed but not yet implemented. They are included in the design for completeness and may be implemented as demand arises.

| Hook Point | Category | Notes |
| --- | --- | --- |
| `component_pre_create` | Component Lifecycle | Blocked on Component-as-ABC refactoring (see above) |
| `component_post_create` | Component Lifecycle | Blocked on Component-as-ABC refactoring (see above) |
| `generation_stream_chunk` | Generation Pipeline | Per-chunk interception during streaming |
| `adapter_pre_load` | Backend Adapter Ops | Before `backend.load_adapter()` |
| `adapter_post_load` | Backend Adapter Ops | After adapter loaded |
| `adapter_pre_unload` | Backend Adapter Ops | Before `backend.unload_adapter()` |
| `adapter_post_unload` | Backend Adapter Ops | After adapter unloaded |
| `context_update` | Context Operations | When context changes (append/reset) |
| `context_prune` | Context Operations | When context is trimmed for token budget |
| `error_occurred` | Error Handling | Cross-cutting hook for unrecoverable errors |

---

## Scoping implementation

A single `PluginManager` instance manages all plugins. Plugins are tagged with an optional `session_id`. At dispatch time, the manager filters: global plugins (no session tag) always run; session-tagged plugins run only when the dispatch context matches their session ID.

With-block scopes use the same `session_id` tagging mechanism. Each `with` block gets a unique UUID scope ID; the plugin manager filters plugins by scope ID at dispatch time and deregisters them by scope ID on exit.

---

## YAML configuration (secondary)

For deployment-time configuration, plugins can be loaded from YAML. This is useful for enabling/disabling plugins or changing priorities without code changes. The `disabled` mode (`PluginMode.DISABLED`) is available in YAML configuration for deployment-time control but is not exposed in Mellea's public `PluginMode` enum.

---

## Custom hook types

The plugin framework supports custom hook types for domain-specific extension points beyond the built-in lifecycle hooks. This is particularly relevant for agentic patterns (ReAct, tool-use loops, etc.) where the execution flow is application-defined. Custom hooks use the same `@hook` decorator and follow the same calling convention, payload chaining, and result semantics. As agentic patterns stabilize in Mellea, frequently-used custom hooks may be promoted to built-in hooks.

---

## Functional API support

The functional API (`instruct(backend, context, ...)`) does not require a session. Hooks still fire at the same execution points. If global plugins are registered, they execute. If no plugins are registered, hooks are no-ops with zero overhead. Session-scoped plugins do not apply because there is no session.
