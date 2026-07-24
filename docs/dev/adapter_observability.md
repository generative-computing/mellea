# Adapter function lifecycle, options, and observability

Epic #929 Phase 2, issue #1140. Covers three things landed together in the
same PR: the narrowed `AdapterMixin` verb contract, the shared
`resolve_model_options` helper, and the `IntrinsicMetricsPlugin` skeleton.

## AdapterMixin verb contract

`AdapterMixin` (`mellea/backends/adapters/adapter.py`) exposes **seven**
verbs, not the four stated in #1140's acceptance criteria. That's a direct
conflict with the issue text as written: Phase 1 (PR #1269) already added
`resolve_adapter()`, which depends on `base_model_name` and `add_adapter`
staying on the mixin, so trimming to four verbs isn't possible without
breaking Phase 1. The count below reflects what actually ships.

### Universal (every backend implements these)

- `base_model_name` — the underlying model's identifier. Read directly by
  `resolve_adapter()` to construct new adapters lazily.
- `add_adapter(adapter)` — registers an adapter with the backend.
  `resolve_adapter()` calls this internally the first time an adapter name
  is resolved.
- `list_adapters()` — returns every adapter the backend *knows about*,
  whether or not it's currently active. Both `LocalHFBackend` and
  `OpenAIBackend` now share this "registered/known" contract:
  `LocalHFBackend.list_adapters()` reads `self._added_adapters` (previously
  it read `self._loaded_adapters`, which only included adapters that had
  been explicitly loaded — that mismatch with `OpenAIBackend`'s semantics is
  fixed as part of this issue).

### Reality-specific (each backend overrides only its own)

Each of the following raises `NotImplementedError` on the mixin by default;
a backend overrides only the verb matching its own adapter reality.

- `load_peft_adapter(name)` / `unload_peft_adapter(name)` — LocalFile/PEFT
  reality (`LocalHFBackend`). Loads or unloads LoRA/aLoRA weights from disk.
  Renamed from the previous `load_adapter`/`unload_adapter`.
- `render_controls(name, active: bool)` — Embedded/Granite Switch reality
  (`OpenAIBackend`). Weights are already baked into the served model, so
  there's nothing to load or unload; this verb exists for future
  control-token rendering. `active=True`/`False` map to the intended
  `activate()`/`deactivate()` calls once #1142 wires EmbeddedBinding.
- `set_request_adapter(name)` — ServerMediated reality. No backend
  implements this yet; the verb name is defined for when that reality is
  built.

`resolve_adapter()` and `adapter_scope()` are unchanged Phase 1 scaffolding
and out of scope for this issue — their real wiring into
`WeightsBinding.activate()`/`deactivate()` belongs to #1141/#1142.

## resolve_model_options

`mellea/backends/_options.py` centralizes the model-options merge logic that
`LocalHFBackend._simplify_and_merge` and `OpenAIBackend._simplify_and_merge`
each used to duplicate. Precedence, lowest to highest:

```text
backend_defaults < helper_defaults < call_options
```

`remap` translates backend/caller-specific option names to `ModelOption`
keys before merging; `helper_defaults` is assumed to already be in
`ModelOption` key form. `call_intrinsic` (`mellea/stdlib/components/intrinsic/_util.py`)
also routes through this helper for its `TEMPERATURE: 0.0` default, so
caller-supplied `model_options` can't be silently clobbered by a hardcoded
default — the same class of bug PR #972 fixed elsewhere.

## IntrinsicMetricsPlugin (skeleton)

`mellea/telemetry/metrics_plugins.py` adds `IntrinsicMetricsPlugin`, hooking
`intrinsic_invocation_complete` and `intrinsic_phase_complete`
(`mellea/plugins/hooks/intrinsic.py`). Three metrics:

- `mellea.adapter_function.invocations` (counter) — labels: `name`, `revision`,
  `binding_type`, `adapter_type`, `outcome` (`success` | `schema_error` |
  `error`).
- `mellea.adapter_function.phase_duration` (histogram, unit `s`) —
  labels: `name`, `phase` (`prepare` | `activate` | `generate` | `parse` |
  `deactivate`).
- `mellea.adapter_function.parse_failures` (counter) — labels: `name`, `revision`.
  Incremented automatically whenever an invocation's `outcome` is
  `schema_error` (i.e. an `AdapterSchemaMismatchError`), acting as a
  schema-drift detector.

No production code fires these hooks yet — this is a skeleton, unit-tested
against synthetic payloads only (`test/telemetry/test_intrinsic_metrics_plugin.py`).
Real wiring from `prepare`/`activate`/`generate`/`parse`/`deactivate` is
expected to go in with #1141 (LocalFileBinding) and #1142 (EmbeddedBinding).

## Span tree (structure)

Span *emission* ships with the Bindings (#1141/#1142) — no span code lands in
this PR. What this issue fixes is the *shape*, so the traces align with the
metrics and follow Mellea's existing tracing conventions rather than a bespoke
scheme. Spans are opened through the `start_*_span` helper family in
`mellea/telemetry/tracing.py` (mirroring `start_backend_span` /
`start_action_span`): the span is named by its operation with `gen_ai.*` set
where the semantic conventions apply, and Mellea-specific fields are attached
under the `mellea.*` prefix — the same convention as `mellea.action_type`,
`mellea.num_actions`, etc.

An invocation opens one parent span with a child span per lifecycle phase:

- **Parent** (the invocation) — carries `mellea.adapter_function.name`,
  `mellea.adapter_function.revision`, `mellea.adapter_function.binding_type`,
  `mellea.adapter_function.adapter_type`, and
  `mellea.adapter_function.outcome`, mirroring the
  `mellea.adapter_function.invocations` counter.
- **Children** (one per phase: `prepare`, `activate`, `generate`, `parse`,
  `deactivate`) — each carries `mellea.adapter_function.phase` and
  corresponds one-to-one with a `mellea.adapter_function.phase_duration`
  histogram sample of the same phase.

Note the deliberate split, consistent with the rest of Mellea: **metric labels
are bare** (`name`, `phase`, `revision`, …) while **span attributes are
`mellea.*`-prefixed** — same values, different surface, each following its
signal type's existing convention.

## Content capture (`MELLEA_TRACES_CONTENT`)

Span *metadata* — names, revisions, phase durations, outcomes — is always safe
to record. Adapter *input and output content* — prompts, retrieved documents,
generated text — is gated behind the **existing** `MELLEA_TRACES_CONTENT`
environment variable: the same content-capture gate Mellea's other spans
already use (it also honours `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`),
**off by default**, so traces never capture PII or proprietary content unless
explicitly opted in. When unset or falsey, the phase spans carry metadata only;
when set truthy, they additionally attach the adapter's input/output content.
The adapter-function spans **reuse this gate rather than introducing a new one**;
content attributes are attached when the Bindings (#1141/#1142) emit spans.

(#1140's acceptance criteria named this `MELLEA_TRACE_CONTENT`; the real,
already-implemented variable is `MELLEA_TRACES_CONTENT` — see
`mellea/telemetry/tracing.py`.)
