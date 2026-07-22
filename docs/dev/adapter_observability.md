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

- `mellea.intrinsic.invocations` (counter) — labels: `name`, `revision`,
  `binding_type`, `adapter_type`, `outcome` (`success` | `schema_error` |
  `error`).
- `mellea.intrinsic.phase_duration` (histogram, unit `s`) — labels: `name`, `phase`
  (`prepare` | `activate` | `generate` | `parse` | `deactivate`).
- `mellea.intrinsic.parse_failures` (counter) — labels: `name`, `revision`.
  Incremented automatically whenever an invocation's `outcome` is
  `schema_error` (i.e. an `AdapterSchemaMismatchError`), acting as a
  schema-drift detector.

No production code fires these hooks yet — this is a skeleton, unit-tested
against synthetic payloads only (`test/telemetry/test_intrinsic_metrics_plugin.py`).
Real wiring from `prepare`/`activate`/`generate`/`parse`/`deactivate` is
expected to go in with #1141 (LocalFileBinding) and #1142 (EmbeddedBinding).
