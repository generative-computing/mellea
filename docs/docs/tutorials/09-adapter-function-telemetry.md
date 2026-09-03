---
title: "Tutorial: Reading Adapter Function Telemetry"
description: "Build a dashboard from Mellea's adapter function metrics: invocation outcomes, phase durations, and parse failures."
# diataxis: tutorial
---

Adapter functions emit three OpenTelemetry metrics that cover invocation
outcomes, per-phase latency, and schema-mismatch failures. This tutorial
walks through enabling them and reading each one.

By the end you will have covered:

- Enabling metrics and picking an exporter
- The three adapter-function instruments and their attributes
- Building a dashboard query for each: success rate, phase latency, and
  parse-failure rate
- What's *not* covered yet — span-level tracing for adapter functions

**Prerequisites:** [Metrics](../observability/metrics.md) covers the
general metrics setup this tutorial builds on. `pip install "mellea[hf,telemetry]"`.

---

## Step 1: Enable metrics

Adapter function metrics follow the same on/off switch and exporters as
every other Mellea metric — no adapter-specific configuration:

```bash
export MELLEA_METRICS_ENABLED=true
export MELLEA_METRICS_CONSOLE=true  # or OTLP/Prometheus — see the Metrics guide
```

They are recorded by `AdapterFunctionMetricsPlugin`
(`mellea/telemetry/metrics_plugins.py`), which subscribes to the
`adapter_function_invocation_complete` and `adapter_function_phase_complete`
hooks — the same hooks `AdapterMixin.adapter_scope()` and
`EmbeddedBinding.apply_activation()` fire for every adapter call, composed
`Adapter` or deprecated shim alike.

## Step 2: The three instruments

| Metric | Type | Attributes | What it tells you |
| --- | --- | --- | --- |
| `mellea.adapter_function.invocations` | Counter | `name`, `revision`, `binding_type`, `adapter_type`, `outcome` | How many calls, split by outcome (`success`, `schema_error`, `error`) |
| `mellea.adapter_function.phase_duration` | Histogram | `name`, `phase` | How long each lifecycle phase took, in seconds |
| `mellea.adapter_function.parse_failures` | Counter | `name`, `revision` | How many calls failed *specifically* on `AdapterSchemaMismatchError` |

`revision` is normalised to the string `"unpinned"` when the binding's
revision is `None` — filter on that value to find adapters running without a
pinned revision (see
[Adapter schema migrations](./08-adapter-schema-migrations.md) for why that
matters). `phase` is one of `"prepare"`, `"activate"`, `"generate"`,
`"parse"`, or `"deactivate"` — though as of this writing only three are
actually emitted: `"prepare"` once per `LocalFileBinding` (fired from
`add_adapter`/`binding.prepare()` — the download-and-load cost), then
`"activate"`/`"deactivate"` per call from `adapter_scope()` for the
LocalFile/PEFT reality, or `"activate"` alone per call for the Embedded
reality (`EmbeddedBinding.apply_activation()`, which has no lifecycle to
deactivate). `"generate"`/`"parse"` phase timing is tracked as future work
in issue #1466.

## Step 3: Build the three dashboard panels

**Success rate** — `mellea.adapter_function.invocations`, grouped by `name`
and `outcome`. A PromQL sketch:

```promql
sum by (name, outcome) (rate(mellea_adapter_function_invocations_total[5m]))
```

Watch `outcome="error"` and `outcome="schema_error"` separately — an error
means the call itself failed (network, generation); a schema error means the
call succeeded but its output didn't parse against the declared contract
(see [Adapter schema migrations](./08-adapter-schema-migrations.md)).

**Phase latency** — `mellea.adapter_function.phase_duration`, grouped by
`name` and `phase`. A p95 activate-phase latency that jumps for one adapter
but not others usually means its weights just got evicted from whatever
cache your deployment relies on (HF Hub's local cache, or your own).

**Parse-failure rate** — `mellea.adapter_function.parse_failures` divided by
`mellea.adapter_function.invocations` for the same `name`. This is the
leading indicator for the schema-drift scenario in
[Adapter schema migrations](./08-adapter-schema-migrations.md): a
sudden rise, correlated with a `revision` change in your own deploy history,
is exactly the signal to re-pin or roll back.

## What's not covered yet

Mellea does not currently open OpenTelemetry **spans** for adapter function
calls — only these three metrics, driven by hooks, exist today. The
`ADAPTER_FUNCTION_*` hook family has no start hook for a tracing plugin to
open a span on yet (see the note in
`AdapterMixin.adapter_scope()`'s docstring); span-level tracing is tracked
separately in issue #1466. If your dashboard needs request-level tracing
today, correlate the metrics above with your own application-level spans
around the `mfuncs.act()`/`m.instruct()` call site instead.

**See also:**
[Metrics](../observability/metrics.md) |
[Telemetry](../observability/telemetry.md) |
[Adapter functions](../advanced/intrinsics.md) |
[Adapter schema migrations](./08-adapter-schema-migrations.md)
