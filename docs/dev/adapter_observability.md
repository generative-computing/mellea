# Adapter function lifecycle, options, and observability

Epic #929 Phase 2, issue #1140. Covers three things landed together in the
same PR: the narrowed `AdapterMixin` verb contract, the shared
`resolve_model_options` helper, and the `AdapterFunctionMetricsPlugin` skeleton.

Issue #1141 built on top of this: `LocalFileBinding` (PEFT/aLoRA reality) now
has a real `prepare`/`activate`/`deactivate`/`release` lifecycle and a real
`from_catalog()` constructor, `adapter_scope()` really activates/deactivates
weights instead of being a no-op, and the span/metric plumbing described below
actually fires. The sections that were written in future tense against #1141
are updated in place rather than left as a historical record — see each
section for what #1141 changed. `EmbeddedBinding` (#1142, Granite Switch
reality) is still unimplemented. The existing `IntrinsicAdapter` /
`resolve_adapter()` / `_generate_from_intrinsic` production hot path is
**not** rewired onto this machinery yet — it still uses its own inline
`set_adapter()` calls and doesn't open these spans or fire these hooks. That
cutover is issue 4.1's job.

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

`resolve_adapter()` is unchanged Phase 1 scaffolding — it still only knows
about `IntrinsicAdapter`/`LocalHFAdapter` and is not used to look up
`LocalFileBinding`/`Adapter` instances. `adapter_scope()` is no longer
scaffolding: as of #1141 it really calls `adapter.weights.activate()` before
the `with` body and `adapter.weights.deactivate()` after (in a `finally`, so
deactivation runs even if the body raises), wrapping both in the span/metric
plumbing described below. Wiring `EmbeddedBinding.activate()`/`deactivate()`
for the Granite Switch reality is #1142's job; `adapter_scope()` itself
doesn't change again for that.

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

## AdapterFunctionMetricsPlugin (skeleton)

`mellea/telemetry/metrics_plugins.py` adds `AdapterFunctionMetricsPlugin`, hooking
`adapter_function_invocation_complete` and `adapter_function_phase_complete`
(`mellea/plugins/hooks/adapter_function.py`). Three metrics:

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

As of #1141, `LocalFileBinding.prepare()` and `AdapterMixin.adapter_scope()`'s
`activate`/`deactivate` phases fire `ADAPTER_FUNCTION_PHASE_COMPLETE`, and
`adapter_scope()` fires `ADAPTER_FUNCTION_INVOCATION_COMPLETE` when the parent
scope closes — both through the standard `has_plugins()`-then-`invoke_hook()`
idiom, so the metrics plugin now receives real payloads whenever
`LocalFileBinding` is prepared and activated/deactivated through
`adapter_scope()`. `release()` opens and closes its own
`adapter_function.release` span but does **not** fire a phase-complete metric:
`AdapterFunctionPhaseCompletePayload.phase`'s `Literal` (`prepare` | `activate`
| `generate` | `parse` | `deactivate`) has no `"release"` value, so there's
nothing for `LocalFileBinding.release()` to report against — this is the
existing #1140 contract, not a #1141 oversight. `generate` and `parse` never
fire in this issue either: nothing in production calls `io_contract.parse()`
yet, since the `IntrinsicAdapter` hot path isn't wired onto this machinery
(see the note at the top of this doc). Closing that gap, and wiring the
equivalent hooks for `EmbeddedBinding`, is issue 4.1's and #1142's job
respectively. `test/telemetry/test_metrics_plugins.py` still exercises the
plugin against synthetic payloads directly; it isn't yet exercised through a
real invocation end-to-end.

## Span tree (structure)

Span *emission* for `LocalFileBinding` ships with #1141, via
`start_adapter_function_span`/`finish_adapter_function_span_success`/
`finish_adapter_function_span_error` and
`start_adapter_function_phase_span`/`finish_adapter_function_phase_span` in
`mellea/telemetry/tracing.py`. `EmbeddedBinding` (#1142) still has no span
emission. What #1140 fixed was the *shape*, so the traces align with the
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
  `deactivate`, plus `release` which only ever gets a span, never a metric —
  see the metrics section above) — each carries `mellea.adapter_function.phase`
  and, except for `release`, corresponds one-to-one with a
  `mellea.adapter_function.phase_duration` histogram sample of the same phase.
  As of #1141, only `prepare`, `activate`, `deactivate`, and `release`
  actually open a span for `LocalFileBinding`; `generate`/`parse` are
  structurally supported but nothing calls them yet (see above).

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
The adapter-function spans **reuse this gate rather than introducing a new one**
by design, but as of #1141 no content attributes are attached yet — the
`start_adapter_function_span`/`start_adapter_function_phase_span` helpers only
set the metadata attributes listed above. There's no adapter input/output
content to attach until `generate`/`parse` actually fire, which doesn't happen
in this issue (see above). Wiring `MELLEA_TRACES_CONTENT`-gated content
attributes is deferred to whichever issue first makes `generate`/`parse` fire
in production — expected to be issue 4.1.

(#1140's acceptance criteria named this `MELLEA_TRACE_CONTENT`; the real,
already-implemented variable is `MELLEA_TRACES_CONTENT` — see
`mellea/telemetry/tracing.py`.)
