# Intrinsic Adapter Lifecycle — Design Proposal

> Status: proposal for shape agreement.
> Addresses Epic #929.
> Structure: **Part I** is for agreeing the problem, goals, terminology, and end state. **Part II** contains the supporting detail — read after Part I lands, not before.

---

# Part I — Summary for agreement

## 1. The problem we are solving

Mellea intrinsics — `check_answerability`, `requirement_check`, `find_citations`, the Guardian helpers — let users add specialised capabilities to a base model. Under the hood each one is an **adapter**: a small artefact that specialises the base model for that one task.

Three sources of friction have accumulated:

1. **Three different kinds of adapter share one class hierarchy.** Local PEFT adapters (weights on disk), Granite Switch "embedded" adapters (weights baked into the base model), and the yet-to-return OpenAI-compatible adapters (weights served behind an API) all try to live under one base class. The code branches on backend identity (`if backend._uses_embedded_adapters:`) to route between them.
2. **Adapter lifecycle is not modelled.** `call_intrinsic` constructs an `IntrinsicAdapter` as a side effect of invoking one, which triggers an unconditional weight download even when no download is needed. The user sees a misleading download error; the real error is masked. There is no concept of "prepare," "activate," "deactivate" as distinct steps.
3. **Small, visible follow-on issues cluster around these two roots** — a five-place model-options hierarchy with a silent-overwrite bug; JSON output keys hardcoded in helpers (`result_json["answerability"]`) that break when an adapter ships a new output schema; the `"requirement-check"` string duplicated across four files; a `CustomIntrinsicAdapter` whose constructor monkey-patches the global catalog with a self-confessed "temporary hack."

Every thread in #929 is a symptom of not having separated the kinds of adapter and their lifecycles cleanly.

## 2. What we are trying to achieve

- **One coherent mental model** of what an adapter is, so users and contributors can reason about intrinsic behaviour without reading the implementation.
- **One code path** through `call_intrinsic` that works regardless of whether the adapter's weights are local, baked into the base model, or hosted on a server.
- **Correct, documented model-option precedence** that does not silently overwrite caller intent.
- **Schema-version safety** so adapters can evolve their output format without breaking callers, and so an adapter whose schema drifts is visible rather than silent.
- **First-class custom adapters** — users can ship their own without monkey-patching a global registry.
- **Observable intrinsic calls** so failures during download, activation, or parsing are diagnosable on first run, not after ad-hoc `print` debugging.
- **Parity, not breakage** — high-level helpers (`check_answerability` etc.) keep their shape; manual adapter construction becomes simpler, not harder.

## 3. Terminology

Names matter in this design because they appear in user-facing error messages, docs, and telemetry attributes. Glossary for this proposal:

| Term | Meaning |
| --- | --- |
| **Base model** | The general-purpose LLM (e.g. `ibm-granite/granite-4.1-3b`) that everything runs on top of. |
| **Intrinsic** | A specialised capability — answerability, citations, requirement-check, and so on — invoked via a high-level helper or the `Intrinsic` AST component. |
| **Adapter** | The artefact that implements an intrinsic on top of a base model. In the redesign, `Adapter` is one class composed of three parts (identity, I/O contract, weights binding). |
| **Identity** | The part of an adapter that says *what it is*: name (e.g. `answerability`), adapter type (`lora` / `alora`), schema version, and optional role. |
| **I/O contract** | The parsed `io.yaml` — prompt template, output parser, model-option defaults. Always present, same shape regardless of reality. |
| **Weights binding** | The part of an adapter that says *how its weights are made available*. Three subclasses, one per reality (see below). Exposes `prepare`, `activate`, `deactivate`, `release`. |
| **Reality A / B / C** | Shorthand for the three "where the weights live" stories: A = local PEFT file, B = baked into the base model (Granite Switch), C = server-mediated (future OpenAI/vLLM). |
| **LoRA / aLoRA** | Two PEFT technologies. LoRA weights always participate; aLoRA only participates after an activation token is seen. Both are adapter types that a given intrinsic may ship as. |
| **Role** | A *semantic* label on an adapter distinct from its name — e.g. `requirement_check`, `context_attribution`. Used by callers (the `Requirement` rerouting path) to find "the adapter that plays this role" without hardcoding a name string. |
| **Qualified name** | Today's disambiguator: `<name>_<adapter_type>`. In the redesign, derived on demand from `identity` rather than stored as a field. |
| **Catalog** | The registry of known intrinsics at `mellea/backends/adapters/catalog.py`. Becomes optional and advisory rather than mandatory and monkey-patched. |
| **`io.yaml`** | The YAML file that declares an adapter's input template, output schema, and generation parameters. Lives in the adapter's HuggingFace repo. |

## 4. Rough end result

An **Adapter** is a small object composed of three parts:

```
Adapter
├── identity      — name, adapter type (lora/alora), schema version, optional role
├── io_contract   — parsed io.yaml: prompt building, output parsing, model options
└── weights       — one of three pluggable bindings (LocalFile, Embedded, ServerMediated)
```

The **weights binding** is where the three realities live. It exposes a single verb set — `prepare`, `activate`, `deactivate`, `release` — that every backend calls uniformly. Each concrete binding implements those verbs for its reality:

| Binding | `prepare` | `activate` | `deactivate` |
| --- | --- | --- | --- |
| `LocalFileBinding` (Reality A) | Download from repo → cache path | PEFT `load_adapter` | PEFT `unload_adapter` |
| `EmbeddedBinding` (Reality B) | No-op (weights baked in) | Render `controls` field into chat template | Drop the `controls` field |
| `ServerMediatedBinding` (Reality C) | No-op (or push weights, depending on sub-case) | Set adapter identifier on API request | Unset identifier |

`call_intrinsic` becomes one flow, no branches on backend type:

```
adapter = backend.resolve_adapter(name)
with backend.adapter_scope(adapter):
    raw = backend.generate(adapter.io_contract.build_prompt(...))
return adapter.io_contract.parse(raw)
```

From this shape, the seven threads of #929 resolve cleanly. Full mapping is in Part II §8.

**What users see:** high-level helpers (`check_answerability` etc.) keep their current shape, with the `model_options=` addition that PR #1003 is introducing. Manual adapter construction collapses from four classes to one, with the binding as the pluggable part. Custom intrinsics no longer require monkey-patching the catalog. Detail in Part II §9.

**What cross-cutting concerns look like:** observability (spans + a schema-drift metric), docs rewrite (`intrinsics_and_adapters.md` is 39 lines describing classes this renames), and a test-parity commitment travel **with** the refactor, not after it. Detail in Part II §10–§11.

## 5. Decisions needed now

These gate decomposition; everything else can live in sub-issues once these land.

1. **Does the end-state shape (§4) hold?** Three realities, `Adapter = identity + io_contract + weights`, role-based lookup for rerouting. Yes / no / what's missing.
2. **Adapter lifecycle default — session-scoped or request-scoped?** Today's HF backend keeps adapters loaded once added; request-scoped load/unload is safer for multi-tenancy but costs latency on a 7B base.
3. **OpenAI Reality C — which concrete shape first?** vLLM-backed LoRA serving (client-known weight file, server-side load) or commercial fine-tunes (fully hosted)? The binding covers both; the first subclass sets the idiom.
4. **Telemetry coupling with #1035.** Land intrinsic spans as part of this refactor, or as a follow-on to PR #1036's Gap 5? Coupling avoids designing content capture twice; decoupling keeps #1036 moving.
5. **Deprecation window.** How long do `IntrinsicAdapter` / `EmbeddedIntrinsicAdapter` / `CustomIntrinsicAdapter` stay as shims before removal? One minor release is the default; confirm.

---

# Part II — Supporting detail

> For deeper review once Part I lands. Skim headings first.

## 6. Three realities of "where the weights live"

### 6.1 Reality A — Local PEFT adapter (today's `IntrinsicAdapter`)

- Weights are a distinct file Mellea downloads from HuggingFace into the local cache.
- At call time, the backend uses the PEFT library to plug those weights into the base model.
- After the call, the backend can unplug them.
- **Physical weights, runtime activation, downloadable lifecycle.**

### 6.2 Reality B — Embedded adapter (today's `EmbeddedIntrinsicAdapter`, used by Granite Switch)

- Weights were **baked into the base model during training**. No separate file to download.
- Activation is done by rendering a `controls` field (structured JSON) into the chat template. The Jinja template places it either at the beginning of the sequence (LoRA technology) or before the generation prompt (aLoRA technology). The model itself routes the request to the right baked-in weights.
- You still need the `io.yaml` for input/output formatting — that's the only artefact the client needs.
- **Pre-installed weights, prompt-level activation, no download lifecycle.**

### 6.3 Reality C — Server-mediated adapter (today's gap, #929 point 5)

Two plausible sub-cases; design must accommodate both.

- **C1 — Client-pulled, server-activated**: weights exist as a file on the client side (or somewhere pullable), but activation happens on a remote inference server (e.g. vLLM loads them and exposes them via a LoRA ID or per-request model alias). The client sends either `model=<adapter-id>` or a dedicated LoRA field in the API request. PR #543 removed this path because vLLM dropped aLoRA support; #929 point 5 anticipates its return as "a much different implementation." This is the likely near-term shape.
- **C2 — Provider-hosted**: weights live entirely on the provider's infrastructure. The client only ever passes an identifier. Applies to commercial fine-tunes behind OpenAI, Azure, etc.

Both share: **no local weight loading, API-parameter activation, `io.yaml` still required client-side.**

## 7. Why the current code is tangled (concrete example)

Inside `_util.call_intrinsic`:

```python
if getattr(backend, "_uses_embedded_adapters", False):
    adapters = EmbeddedIntrinsicAdapter.from_source(...)
else:
    intrinsic_adapter = IntrinsicAdapter(...)  # Reality A path
```

Three problems:

1. **`_uses_embedded_adapters` is a backend flag, not an adapter property.** It hard-codes "this backend type → always this adapter type." Reality C needs a third branch, then a fourth if a backend supports both.
2. **The `else` branch calls `obtain_lora` unconditionally** via `IntrinsicAdapter.__init__` → `download_and_get_path`. If the adapter was meant to be a different type, the user sees a misleading download-path error instead of the real cause.
3. **Output parsing assumes one schema.** `result_json["answerability"]` is hardcoded in helpers. When PR #1008 changed `requirement-check` output from `{"requirement_likelihood": 0.9}` to `{"requirement_check": {"score": 0.9}}`, the parsing helper had to be rewritten and the catalog gained a second entry (`requirement_check` for Granite 3.x, `requirement-check` for Granite 4.x) to support both.

## 8. Full #929 thread mapping

| Thread | Resolution |
| --- | --- |
| 1a. Loading/unloading divergence | One `WeightsBinding` verb set; control flow identical across realities. |
| 1b. `obtain_lora` always-called bug | Only `LocalFileBinding.prepare` calls `obtain_lora`; others no-op. |
| 1c. Backend- + adapter-type-specific abstraction | `WeightsBinding` is the adapter-type axis; `AdapterMixin` verbs are the backend axis. |
| 2a. Intrinsic rewriters overwrite options | `Adapter.resolve_model_options()` replaces the five-place merge with one documented stack. |
| 2b/2c. Model-option hierarchy | Five layers enforced in `resolve_model_options` (base model → adapter config → `io.yaml` defaults → `io.yaml` per-intrinsic → caller). |
| 3. Naming consistency | Three-axis identity (`name`, `adapter_type`, `version`) plus explicit `role`. |
| 4a. `call_intrinsic` assumes one output schema | `io_contract.parse()` dispatches on `(name, version)`; helpers see normalised shape. |
| 4b. Per-adapter vs standard schema | `io_contract.parse()` is per-adapter; helpers define the normalised post-parse shape. |
| 4c. Versioning | Schema version declared in `io.yaml` (`schema_version:`); defaults to `v1`. |
| 5. OpenAI backend support | Ships as one or two `ServerMediatedBinding` subclasses. |
| 6. Catalog cleanup | Catalog becomes optional resolver (`LocalFileBinding.from_catalog(name)`). Custom adapters bypass it; no monkey-patching. Duplicate `requirement_check` / `requirement-check` entries collapse into one entry with two schema versions. |
| 7. Hardcoded `requirement-check` refs | Callers look up by **role**, not name. |

## 9. What users see — detailed

**High-level helpers** keep their signatures. The `model_options=` parameter lands via PR #1003:

```python
score = check_answerability(question, documents, context, backend)
score = check_answerability(question, documents, context, backend,
                            model_options={"temperature": 0.1})
```

**Manual adapter construction** collapses from four classes (`IntrinsicAdapter`, `EmbeddedIntrinsicAdapter`, `CustomIntrinsicAdapter`, abstract base) to one `Adapter` + a binding:

```python
# Stock intrinsic from the catalogue:
adapter = Adapter(name="answerability",
                  weights=LocalFileBinding.from_catalog("answerability"))

# Custom intrinsic — no catalog monkey-patching:
adapter = Adapter(name="my-thing",
                  weights=LocalFileBinding(source="myuser/my-adapter",
                                           base_model_name="granite-4.1-3b"),
                  io_contract=IOContract.from_yaml("./io.yaml"))

# Granite Switch embedded:
adapter = Adapter(name="answerability",
                  weights=EmbeddedBinding.from_base_model(backend))
```

**Backend authors** keep `AdapterMixin` as the backend surface, but it exposes only the verbs a backend naturally has: `load_peft_adapter`, `unload_peft_adapter`, `render_controls`, `set_request_adapter`. Bindings call into these verbs. Adding a new reality = adding a new verb + new binding.

## 10. Observability

Intrinsic calls have no bespoke instrumentation today. Folding it into the redesign costs one span attribute per verb; retrofitting means re-editing every binding later.

**Spans** — each `adapter_scope` wraps a child span tree rooted at `intrinsic.call`, with children `intrinsic.prepare`, `intrinsic.activate`, `intrinsic.generate`, `intrinsic.parse`, `intrinsic.deactivate`. Standard attributes: `intrinsic.name`, `intrinsic.version`, `intrinsic.role`, `intrinsic.adapter_type`, `intrinsic.binding_type`, `intrinsic.source`, `intrinsic.target`. Errors set OTel `ERROR` status (aligns with #1035 gap 4).

**Metrics** — an `IntrinsicMetricsPlugin` alongside the existing Token / Latency / Error plugins:
- `mellea.intrinsic.invocations` — counter labelled by name, version, binding type, adapter type, outcome.
- `mellea.intrinsic.phase_duration_ms` — histogram labelled by name, phase.
- `mellea.intrinsic.parse_failures` — counter labelled by name, version. This is the **schema-drift detector**: a climbing counter against a specific `(name, version)` pair means an upstream adapter shipped a schema change without a version bump.

**Content capture** — gated behind PR #1036's `MELLEA_TRACE_CONTENT` flag. Intrinsics emit `intrinsic.input.kwargs` (structured dict), `intrinsic.output.raw` (raw JSON string), and `intrinsic.output.parsed` (normalised shape) as span events. Different shape from chat `gen_ai.*.message` events because intrinsics have different semantics.

## 11. Docs, tests, tutorials

First-class deliverables, not afterthoughts.

**Docs** — rewrite (not edit) for `docs/dev/intrinsics_and_adapters.md` (39 lines describing classes that get renamed). Update `docs/dev/requirement_aLoRA_rerouting.md` to describe role-based lookup instead of hardcoded strings. User-facing `docs/docs/advanced/intrinsics.md` and examples under `docs/examples/intrinsics/` are breaking-API touched. New dev doc for adapter observability. Update AGENTS.md §13 once normalised post-parse shapes are stable.

**Tests** — existing intrinsic tests stay green per phase. New tests cover: each binding × each verb (unit); integration matrix `{HF, OpenAI} × {applicable bindings} × {lora, alora where applicable} × {every existing intrinsic}`; per-version parse round-trips (with `requirement-check` v1 / v2 as the worked case); concurrency window correctness; span/metric emission assertions.

**Tutorials** — three worth writing alongside the refactor:
- "Adding a custom intrinsic in 20 lines" — replaces the `CustomIntrinsicAdapter` monkey-patch story.
- "Shipping a new schema version without breaking users" — worked example using `requirement-check` v1 → v2.
- "Reading intrinsic telemetry" — short dashboard-building guide.

**Release notes** separate: no-op for high-level helper users; deprecated-but-shimmed for direct adapter constructors; removed at Phase 4 (see below).

## 12. Migration (rough shape only)

Detail deferred until Part I §5 decisions land, but the intended phasing is:

1. **Phase 0 — parallel types.** Introduce `Adapter` / `WeightsBinding` / `IOContract` alongside existing classes. No call-site changes, tests unchanged.
2. **Phase 1 — callers move.** `_util.call_intrinsic`, requirement rerouting, and each helper switch to new types. Old classes become deprecation shims.
3. **Phase 2 — backends move.** `AdapterMixin` narrows to the new verb set. Backends drop per-call `_simplify_and_merge` in favour of `resolve_model_options`.
4. **Phase 3 — Reality C lands.** `ServerMediatedBinding` subclass(es) written; OpenAI backend drops `_uses_embedded_adapters` hard-code.
5. **Phase 4 — shim removal.** After one minor release with deprecation warnings.

Observability and docs deliverables attach to the phase that first exercises them.

## 13. Open questions (full list)

1. **Naming.** `WeightsBinding` vs `ResourceStrategy` vs `AdapterProvider`. Pick one; the term leaks into error messages.
2. **Lifecycle default** — session-scoped or request-scoped (also in Part I §5).
3. **Role vs name.** Free-form `role` string, or a small enum so users can't invent roles backends don't honour?
4. **Reality C idiom.** vLLM LoRA serving first or commercial fine-tunes first (also in Part I §5).
5. **Rewind interaction (PR #1028).** `factuality_detection` / `factuality_correction` mutate context via `context.previous_node`. Belongs on `io_contract.build_prompt` (cleaner) or stay in the helper (smaller migration blast radius)?
6. **Telemetry coupling with #1035** (also in Part I §5).
7. **Deprecation window** (also in Part I §5).

---

_Verified against: `mellea/backends/adapters/{adapter,catalog,__init__}.py`, `mellea/stdlib/components/intrinsic/{_util,intrinsic,core,rag,guardian}.py`, `mellea/backends/{openai,huggingface}.py`, `mellea/formatters/granite/intrinsics/input.py`, `mellea/stdlib/requirements/requirement.py`, `docs/dev/{intrinsics_and_adapters,requirement_aLoRA_rerouting}.md`, PRs #543 / #881 / #986 / #994 / #1003 / #1008 / #1028 / #1036, commits `666d646a`, `8b6b8d55`, `c57aba1d`, `8577d092`._
