# Adapter Lifecycle — Design Proposal

> **Addresses:** [Epic #929 — Fix Intrinsic Adapter Lifecycle & Consistency in Mellea](https://github.com/generative-computing/mellea/issues/929). Read the epic first if you haven't; it catalogues the specific threads this proposal tries to resolve coherently rather than individually.
>
> **Status:** proposal. Design docs produced during implementation will live under `docs/dev/`.
>
> **Structure:** **Part I** covers the problem, goals, terminology, end state, and the decisions that gate decomposition. **Part II** contains supporting detail — read after Part I is agreed, not before.
>
> **Terminology:** **"Adapter"** is the backend artefact: the weights loaded by a backend. The user-facing layer — helpers, input/output parsing, the AST component — is referred to as **`AdapterBasedComponent`** throughout this document. This is a placeholder name: IBM is retiring "Intrinsic" but has not yet confirmed the replacement; Mellea will adopt whatever name is settled upstream.
>
> **Related issues and prior work:** see the appendix at the end of this document for a linked index with annotations.

---

# Part I — Summary for agreement

## Proposal at a glance

**What changes:** three separate adapter classes (`IntrinsicAdapter`, `EmbeddedIntrinsicAdapter`, `CustomIntrinsicAdapter`) collapse into one `Adapter`:

```
Adapter = identity + io_contract + weights_binding
```

The `weights_binding` is pluggable — `LocalFileBinding`, `EmbeddedBinding`, or `ServerMediatedBinding` — each exposing the same four verbs (`prepare`, `activate`, `deactivate`, `release`). The backend calls these uniformly; it does not branch on adapter type.

**What stays the same:** all high-level helpers (`check_answerability`, `requirement_check`, etc.) keep their current signatures. Deprecated classes are shimmed for one release.

**Five decisions gate decomposition:**

| # | Question | Status |
| --- | --- | --- |
| Q1 | Does the `Adapter = identity + io_contract + weights` shape hold? | **Resolved** (Jake): shape holds |
| Q2 | Lifecycle default | **Resolved** (Jake): session-scoped loading; per-call auto-activate/deactivate |
| Q3 | Reality C (server-mediated): design slot or leave empty? | **Resolved**: design slot, leave empty (Paul; vLLM blocked) |
| Q4 | Deprecation window for old classes | **Resolved**: 1 minor release ≈ 4–6 weeks; longer if user impact warrants (Paul, Jake) |
| Q5 | Terminology: name for the user-facing layer | **Resolved** (Jake): two-layer split adopted; user-facing layer named `AdapterBasedComponent` as placeholder pending IBM's final decision |

Detail on each in §5.

---

## 1. The problem we are solving

Mellea intrinsics — `check_answerability`, `requirement_check`, `find_citations`, the Guardian helpers — let users add specialised capabilities to a base model. Under the hood each one is an **adapter**: a small artefact that specialises the base model for that one task.

Three sources of friction have accumulated:

1. **Three different kinds of adapter share one class hierarchy.** Local PEFT adapters (weights on disk), Granite Switch "embedded" adapters (weights baked into the base model), and the yet-to-return OpenAI-compatible adapters (weights served behind an API) all try to live under one base class. The code branches on backend identity (`if backend._uses_embedded_adapters:`) to route between them.
2. **Adapter lifecycle is not modelled.** `call_intrinsic` constructs an `IntrinsicAdapter` as a side effect of invoking one, which triggers an unconditional weight download even when no download is needed. The user sees a misleading download error; the real error is masked. There is no concept of "prepare," "activate," "deactivate" as distinct steps.
3. **Small, visible follow-on issues cluster around these two roots** — a five-place model-options hierarchy with a silent-overwrite bug; JSON output keys hardcoded in helpers (`result_json["answerability"]`) that break when an adapter ships a new output schema; the `"requirement-check"` string duplicated across four files; a `CustomIntrinsicAdapter` whose constructor monkey-patches the global catalog with a self-confessed "temporary hack."

Every thread in #929 is a symptom of not having separated the kinds of adapter and their lifecycles cleanly. This is not a theoretical concern: **seven fix-up commits have been merged in the adapter area in recent history** (full list in the appendix), alongside the `obtain_lora`-always-called masked error and the hardcoded `"requirement-check"` strings flagged by #929 point 7 / PR #1008 — the picture is of a subsystem that receives repeated small-scope fixes rather than a stable abstraction.

## 2. What we are trying to achieve

Four outcomes, in order of importance. Detail on each lives in Part II; this list is the ask.

1. **One adapter model, one code path.** Reasonable from the outside, unified from the inside — no more `if backend._uses_embedded_adapters:` branches.
2. **Safe evolution.** Model-option precedence is documented and enforced. Adapter weights are versioned by HF commit SHA — Mellea can pin to a specific revision for stability or track latest for newest weights (refresh policy in §17 Q5). Output schemas are stable in the common case (new weights, same schema); the rare breaking schema change is handled by pinning the HF revision and by helpers raising `AdapterSchemaMismatchError` when parse cannot yield the helper's declared output contract (Jake req 4). Forward-compatible additions (e.g. an extra optional field) do not trigger the error — only contract-breaking deltas do. Helpers like `check_answerability` see a normalised result regardless of underlying churn. Output-schema versioning beyond this is tracked in [#1111](https://github.com/generative-computing/mellea/issues/1111) (§17 Q4).
3. **First-class customer adapters.** Customers can ship their own against the same API as first-party ones — today it requires patching the catalog or subclassing a self-confessed "temporary hack" ([#424](https://github.com/generative-computing/mellea/issues/424)).
4. **Observable and parity-respecting.** Every lifecycle phase is a distinct span; high-level helpers (`check_answerability` etc.) keep their shape; manual adapter construction becomes simpler, not harder.

## 3. Key terms (brief)

Only the few terms needed to read Part I:

- **AdapterBasedComponent** *(placeholder name)* — the user-facing capability: helper functions like `check_answerability`, the AST component, and input/output parsing. Implemented by an adapter. IBM is retiring "Intrinsic" and the replacement name is not yet confirmed; this document uses `AdapterBasedComponent` until that decision lands.
- **Adapter** — the backend artefact: the weights loaded by a backend (LoRA / aLoRA / embedded), with its identity and I/O contract.
- **Base model** — the general-purpose LLM everything runs on top of (e.g. `ibm-granite/granite-4.1-3b`).
- **LoRA / aLoRA** — the two PEFT technologies adapters use. Both are supported.
- **Reality A / B / C** — shorthand introduced in §4 for the three "where the weights live" stories.

Full glossary (identity, I/O contract, weights binding, role, qualified name, catalog, io.yaml) is in §7 — needed only when you descend into the detail.

## 4. Rough end result

An **Adapter** is a small object composed of three parts:

```
Adapter
├── identity      — name, adapter type (lora/alora), optional role
├── io_contract   — parsed io.yaml: prompt building, output parsing, model options
└── weights       — one of three pluggable bindings (LocalFile, Embedded, ServerMediated)
```

**Sane defaults:** when an adapter's weights come from a HuggingFace repo, the `io_contract` defaults to the `io.yaml` in that same repo. Callers rarely pass `io_contract=` explicitly. Identity, I/O contract, and weights are tightly coupled by design; the defaults treat them as a unit.

The **weights binding** is where the three realities live. It exposes a single verb set — `prepare`, `activate`, `deactivate`, `release` — that every backend calls uniformly. What each verb does per reality lives in §9; the high-level picture is all three realities converging on one shared `io_contract`:

```mermaid
flowchart LR
    subgraph A["Reality A — Local PEFT"]
        direction TB
        A1["HF repo"] -->|"download"| A2["local cache"]
        A2 -->|"PEFT load"| A3["base model<br/>+ LoRA"]
    end
    subgraph B["Reality B — Embedded (Granite Switch)"]
        direction TB
        B1["base model<br/>(weights baked in)"] -->|"render <i>controls</i><br/>in chat template"| B2["base model<br/>(activated)"]
    end
    subgraph C["Reality C — Server-mediated"]
        direction TB
        C1["remote server<br/>(holds weights)"] -->|"adapter_id<br/>in API request"| C2["base model<br/>(remote)"]
    end
```

Adapter invocation becomes one flow, with no branches on backend type. From this shape, the seven threads of #929 resolve cleanly. The simplified invocation pseudocode, the per-binding verb semantics, the lifecycle sequence diagram, and the thread-by-thread mapping are in Part II (§9 and §12).

**What users see:** high-level helpers (`check_answerability` etc.) keep their current shape, with the `model_options=` and `documents=` additions that fold in here from #1003 (PR #1028 was closed 2026-05-15 in favour of this epic). Manual adapter construction collapses from four classes to one, with the binding as the pluggable part. Custom intrinsics no longer require monkey-patching the catalog. Detail in Part II §13.

**What cross-cutting concerns look like:** observability (spans + a schema-drift metric), docs rewrite (`intrinsics_and_adapters.md` is 39 lines describing classes this renames), and a test-parity commitment travel **with** the refactor, not after it. Detail in Part II §14–§15.

### 4.1 Backend scope

Of Mellea's five backends (`LocalHFBackend`, `OpenAIBackend`, `OllamaBackend`, `WatsonxBackend`, `LiteLLMBackend`), **the two primary adapter backends are `LocalHFBackend` and `OpenAIBackend`** — those are what this design targets. The remaining three are out of scope for adapter support because the underlying providers do not support the mechanisms Mellea's adapters need today. The `WeightsBinding` abstraction does not preclude adding them later. Full backend × reality matrix with per-backend reasoning is in §10.

## 5. Decisions needed now

These gate decomposition; everything else can live in sub-issues once these are agreed.

1. **Does the end-state shape (§4) hold?** Three realities, `Adapter = identity + io_contract + weights`, role-based lookup for rerouting. **Resolved (Jake):** shape holds. In most cases identity / io_contract / weights will be colocated, but allowing divergence is the point — it enables separation of how weights are fetched from the adapter's functional definition.
2. **Adapter lifecycle — session-scoped loading, per-call activate/deactivate.** **Resolved (Jake):** two-level lifecycle. Weight loading (`prepare`) is session-scoped — the adapter is loaded once and held until explicit `release()` at session teardown. Activation/deactivation (`activate`/`deactivate`) is call-scoped — auto-wrapped around each generation. This matches the §9.3 sequence diagram. The multi-tenancy concern is reduced because `LocalHFBackend` is primarily a single-user/local backend (see §10). Request-scoped lifecycle (including `prepare`/`release` per call) remains an opt-in for deployments that need per-call isolation.
3. **Reality C target shape — design slot, leave empty.** **Resolved (Paul):** the aLoRA-on-vLLM path ([#27](https://github.com/generative-computing/mellea/issues/27)) is currently blocked — vLLM has declined to upstream aLoRA support (see §8.3 for history). The `ServerMediatedBinding` slot is designed so the interface is clean if the upstream situation ever changes, but the implementation stays empty and we don't invest in stubs.
4. **Deprecation window — at least 1 minor release; longer if user impact warrants.** **Resolved (Paul, Jake):** Paul confirms 1 minor release ≈ 4–6 weeks is sufficient, extendable if needed; Jake notes the final length depends on how many users are impacted. **Sub-question (open):** can this ship without breakage at all? IBM is retiring "Intrinsic" (see Q5), so `IntrinsicAdapter` cannot stay as a permanent re-export — it will eventually need to go. The question is whether the deprecation shim for `IntrinsicAdapter` → `AdapterBasedComponent` (placeholder) can be deferred until the upstream name is confirmed, effectively separating the structural refactor from the naming change.
5. **Terminology — two-layer split, with `AdapterBasedComponent` as placeholder.** **Resolved (Jake):** the conceptual split is agreed. "Adapter" is the backend artefact (weights loaded by a backend). The user-facing layer — helper functions, input/output parsing, the AST component — is a distinct abstraction. IBM is retiring the "Intrinsic" name but has not yet confirmed its replacement; until that decision lands, Mellea will use **`AdapterBasedComponent`** as the working placeholder name throughout the codebase and docs.

   Three implementation sub-questions follow from this:
   - **Q5a. Prose rename** — shift docs, error messages, and help text away from "Intrinsic" to `AdapterBasedComponent` (or the final IBM name once known). Zero breakage.
   - **Q5b. Module rename** — rename `mellea.stdlib.components.intrinsic` → `mellea.stdlib.components.adapter_based_component` (placeholder path), with the old path re-exported for one release. Breaking for submodule importers.
   - **Q5c. AST class rename** — rename `Intrinsic` AST component → `AdapterBasedComponent`, with `Intrinsic` as a deprecation alias for one release.

> **Implementation note, not a reviewer question:** intrinsic-level observability (§14) should coordinate with the in-flight [#1035](https://github.com/generative-computing/mellea/issues/1035) / [PR #1036](https://github.com/generative-computing/mellea/pull/1036) work so content capture uses the same `MELLEA_TRACE_CONTENT` flag and doesn't get designed twice. Flagged here for awareness; sequenced during implementation.

## 6. Impact and blast radius

Scope of this refactor in concrete terms so reviewers can weigh the cost.

### API surface

- **Unchanged** — every high-level helper (`check_answerability` etc.) keeps its signature. `m.instruct`, `m.validate`, `m.chat` unaffected. The `model_options=` and `documents=` additions from [#1003](https://github.com/generative-computing/mellea/issues/1003) (PR #1028 closed 2026-05-15 in favour of this epic) ship as part of Phase 1.
- **Deprecated but shimmed for one release** — `IntrinsicAdapter`, `EmbeddedIntrinsicAdapter`, `CustomIntrinsicAdapter` public classes. Direct users get `DeprecationWarning` pointing to the new constructor. *(Applies if Q5 settles on rename or new-API-alongside; under Jake's split, the old types stay as re-exports indefinitely with no deprecation needed.)*
- **Optional, was mandatory** — the adapter catalogue. Callers no longer have to register custom adapters in `catalog.py` before use; the catalogue stays as a convenience resolver for first-party names, not a precondition.
- **Possibly moved/renamed** — depends on §5 Q5 (terminology rename scope).

### User-archetype impact

| Audience | Impact |
| --- | --- |
| Helper user (`check_answerability`-style calls) | None beyond the `model_options=` / `documents=` additions from [#1003](https://github.com/generative-computing/mellea/issues/1003) and clearer error messages. |
| Advanced user constructing adapters directly | One release of deprecation warnings, then adopt the new `Adapter(name=…, weights=…)` constructor. |
| Customer writing their own adapter | First-class path; no more `CustomIntrinsicAdapter` monkey-patching; no forced catalogue upload. Resolves [#424](https://github.com/generative-computing/mellea/issues/424). |
| Backend author | `AdapterMixin` verb set narrows to the natural operations each backend can perform; existing implementations update or use shim methods. |
| Operator / SRE | New spans and metrics per §14; easier diagnosis of adapter failures and cost attribution. |

### Code reach

Files and modules touched, approximate: `mellea/backends/adapters/{adapter,catalog,__init__}.py`, `mellea/backends/{huggingface,openai}.py`, `mellea/stdlib/components/intrinsic/*`, `mellea/formatters/granite/intrinsics/*`, `mellea/stdlib/requirements/requirement.py`, `docs/examples/intrinsics/*`, `docs/dev/{intrinsics_and_adapters,requirement_aLoRA_rerouting}.md`. Larger than a typical feature PR; phased per §16 so individual PRs stay reviewable.

### Release planning

- **Target release (minor, exact number TBD)**: §5 agreement plus Phases 0–2 of the migration (new `Adapter` / `WeightsBinding` / `IOContract` types, call-site adoption, backend narrowing, deprecation shims for old classes, unified model-option precedence, observability per §14, tests per §15).
- **Follow-on minor release**: [#1018](https://github.com/generative-computing/mellea/issues/1018) (embedded adapters on `LocalHFBackend`), Phase 4 shim removal.
- **Deferred until upstream moves**: Reality C / [#27](https://github.com/generative-computing/mellea/issues/27).

### Blocking and unblocking

- **Blocks** [#1018](https://github.com/generative-computing/mellea/issues/1018) (explicitly stated in its issue body).
- **Substantially addresses** [#423](https://github.com/generative-computing/mellea/issues/423) (adapter code undocumented and over-specialised), [#424](https://github.com/generative-computing/mellea/issues/424) (cannot use intrinsics without uploading), all seven threads of [#929](https://github.com/generative-computing/mellea/issues/929).
- **Coordinates with** [PR #1036](https://github.com/generative-computing/mellea/pull/1036) on content-capture semantics.
- **Blocked by** upstream vLLM position on aLoRA ([#27](https://github.com/generative-computing/mellea/issues/27)) — and only for Reality C. Parts I–II of this design are not gated on upstream.

### Performance

- **Likely neutral or improved.** Session-scoped lifecycle is the proposed default (matches current `LocalHFBackend` behaviour); no additional load/unload cost per call. Unified parsing avoids the double-parse that the current output-normalisation sometimes does.
- **Regression watch**: if §5 Q2 chooses request-scoped, per-call PEFT load/unload becomes a visible cost. Measure before adoption.

### Risk

- **Biggest unknown**: whether the unified `resolve_model_options` handles every combination currently in use. Mitigation: keep the five-layer precedence explicit, add per-adapter override documentation, and assert resolved values in tests.
- **Second biggest**: handling breaking schema changes from upstream. Two layers: pinning (avoid the risk), and helpers raising `AdapterSchemaMismatchError` when parse cannot yield the helper's declared output contract (Jake req 4, loud safety net). Forward-compatible additions do not trigger the error. Worked example: the [#1008](https://github.com/generative-computing/mellea/pull/1008) `requirement-check` change would have surfaced as `AdapterSchemaMismatchError` on the first call after the schema change, rather than silently returning `False`. (Output-schema versioning is tracked separately in [#1111](https://github.com/generative-computing/mellea/issues/1111) — §17 Q4.)
- **Mitigated by**: per-phase test-parity commitment (nothing merges if existing tests regress); observability introduced alongside the refactor so production regressions surface as dashboard signals rather than silent behavioural drift.

---

# Part II — Supporting detail

> For deeper review once Part I is agreed. Part II expands the definitions and the design so that reviewers can pressure-test the specifics. Sections are roughly ordered from "what exactly are we talking about" (terminology, realities, end-state detail) through "why this shape is right" (current tangle, thread mapping) to "what it looks like in practice" (user-facing, observability, docs/tests, migration, open questions).

## 7. Terminology (full glossary)

Names matter because they appear in user-facing error messages, docs, and telemetry attributes. The short list for quick reading is in Part I §3; this is the complete reference.

| Term | Meaning |
| --- | --- |
| **Base model** | The general-purpose LLM (e.g. `ibm-granite/granite-4.1-3b`) that everything runs on top of. |
| **AdapterBasedComponent** *(placeholder)* | The user-facing capability: helper functions (`check_answerability`, `requirement_check`, the Guardian helpers), the AST component, input/output parsing. Backed by an adapter. IBM is retiring "Intrinsic" and has not yet confirmed the replacement name; `AdapterBasedComponent` is used throughout this document as a placeholder (see Part I §5). |
| **Adapter** | The backend artefact: the weights loaded by a backend (LoRA / aLoRA / embedded), with its identity, I/O contract, and weights binding. The user-facing **Intrinsic** wraps an adapter to provide helpers and parsing. In the redesign, the class hierarchy collapses from four (`IntrinsicAdapter` / `EmbeddedIntrinsicAdapter` / `CustomIntrinsicAdapter` + abstract base) to one `Adapter` + a pluggable binding. |
| **Identity** | The part of an adapter that says *what it is*: name (e.g. `answerability`), adapter type (`lora` / `alora`), and optional role. |
| **I/O contract** | The parsed `io.yaml` — prompt template, output parser, model-option defaults. Always present, same shape regardless of reality. *Name under discussion: Jacob prefers `io_config`; `io_contract` is used throughout this proposal but is not final.* |
| **Weights binding** | The part of an adapter that says *how its weights are made available*. Three subclasses, one per reality. Exposes `prepare`, `activate`, `deactivate`, `release`. |
| **Reality A / B / C** | Shorthand for the three "where the weights live" stories: A = local PEFT file, B = shipped with the base model (Granite Switch), C = server-mediated (future OpenAI/vLLM). |
| **LoRA / aLoRA** | Two PEFT technologies. LoRA weights always participate; aLoRA only participates after an activation token is seen. A single adapter ships as one or the other (some intrinsics as either); both are supported across all three realities (including embedded — Granite Switch has LoRA and aLoRA adapters in the same repo, `technology` field on each). |
| **Role** | A *semantic* label on an adapter distinct from its name — e.g. `requirement_check`, `context_attribution`. Used by callers (the `Requirement` rerouting path) to find "the adapter that plays this role" without hardcoding a name string. |
| **Qualified name** | Today's disambiguator: `<name>_<adapter_type>`. In the redesign, derived on demand from `identity` rather than stored as a field. |
| **Catalog** | The registry of known adapters at `mellea/backends/adapters/catalog.py`. Becomes optional and advisory rather than mandatory and monkey-patched. |
| **`io.yaml`** | The YAML file that declares an adapter's input template, output schema, and generation parameters. Lives in the adapter's HuggingFace repo. |

## 8. Three realities of "where the weights live"

### 8.1 Reality A — Local PEFT adapter (today's `IntrinsicAdapter`)

- Weights are a distinct file Mellea downloads from HuggingFace into the local cache.
- At call time, the backend uses the PEFT library to plug those weights into the base model.
- After the call, the backend can unplug them.
- **Physical weights, runtime activation, downloadable lifecycle.**

### 8.2 Reality B — Embedded adapter (today's `EmbeddedIntrinsicAdapter`, used by Granite Switch)

- Adapter weights **ship in the same HuggingFace repo as the base model**. They come down with the base-model snapshot and are not fetched separately — confirmed by the fact that `EmbeddedIntrinsicAdapter.from_hub` downloads only `adapter_index.json` + `io_configs/**`, not weight files. The phrase "baked into the base model" is a useful shorthand but imprecise: the weights are still distinct PEFT modules, just co-located and pre-loaded by the inference runtime.
- **Both LoRA and aLoRA are supported.** `adapter_index.json` lists each embedded adapter with a `technology` field (`"lora"` or `"alora"`). The chat template uses that field to place the `controls` JSON at the correct position — beginning of sequence for LoRA, before the generation prompt for aLoRA — so the right adapter is active for the right span of tokens. Granite Switch therefore genuinely carries both technologies; it is not a LoRA-only reality.
- On the client side, only `io.yaml` is needed to format inputs and parse outputs.
- **Pre-installed weights, prompt-level activation, no separate download lifecycle.**

### 8.3 Reality C — Server-mediated adapter (partially gap today)

The OpenAI-compatible backend **already supports adapters** — but only embedded ones (Granite Switch via Reality B, added in [PR #881](https://github.com/generative-computing/mellea/pull/881)). What's missing is *non-embedded* server-side adapters.

**The history (corrected):** Mellea previously ran aLoRA adapters through the OpenAI backend against a **custom vLLM build** that carried an aLoRA patch. The upstream vLLM project declined to merge that patch (confirmed in [PR #543](https://github.com/generative-computing/mellea/pull/543)'s review: "the vLLM aLoRA PR will not [be] accepted, so the alora/intrinsics code for openai is now all dead code"), so PR #543 removed the dead path. Upstream vLLM has therefore **never carried** aLoRA support — the right framing is "declined upstream," not "dropped."

**Current status (confirmed by Paul):** The aLoRA-on-vLLM path is blocked. vLLM has declined the upstream aLoRA patch, and there is no known path to change this. [Issue #27](https://github.com/generative-computing/mellea/issues/27) remains open to track any change in upstream position, but it is not a near-term delivery target. The design slot in this proposal exists as an interface commitment — if the upstream situation ever changes, here is the clean implementation path — not as an active work item.

**Scope of this reality:** whatever the eventual technology path, the design slot is the same. Two sub-cases the binding must accommodate when the path becomes viable:

- **C1 — Client-pulled, server-activated**: weights exist as a file client-side (or somewhere pullable), but activation happens on a remote inference server which loads them and exposes them via a LoRA ID or per-request model alias. This is the vLLM-shaped path, paced by #27 being unblocked.
- **C2 — Provider-hosted**: weights live entirely on the provider's infrastructure. The client only ever passes an identifier. Applies to commercial fine-tunes behind OpenAI, Azure, etc. Not currently a known target in Mellea.

Both share: **no local weight loading, API-parameter activation, `io.yaml` still required client-side.** The first concrete `ServerMediatedBinding` subclass sets the idiom for the API shape.

**Intent summary for OpenAI-compatible support:** keep and extend. Embedded support stays. The design leaves a clean slot for C1 to be populated when #27 is unblocked upstream; C2 is noted for completeness but not a near-term target.

## 9. End-state design detail

### 9.1 Simplified invocation

Adapter invocation collapses to a single flow with no branching on backend type:

```
adapter = backend.resolve_adapter(name)
with backend.adapter_scope(adapter):
    raw = backend.generate(adapter.io_contract.build_prompt(...))
return adapter.io_contract.parse(raw)
```

Every verb that varies per reality lives inside `adapter_scope` (see §9.3); the outer flow is the same whether the adapter is a local PEFT file, an embedded Granite Switch adapter, or a server-mediated one.

> **Boundary constraint:** `io_contract.build_prompt()` and `io_contract.parse()` must delegate to `granite-common` / `granite-formatters` for all `io.yaml` handling and parsing. The `IOContract` class in Mellea wraps these libraries; it does not re-implement their logic. (Jacob's requirement — keep `io.yaml` parsing in the granite-common / granite-formatters boundary.) `build_prompt()` returns a `Component`-compatible prompt object — not a raw string — consistent with the rest of Mellea's prompt pipeline.

### 9.2 Weights binding verbs per reality

Each concrete binding implements the four-verb set from Part I §4. The column meanings do not change between realities — only what happens inside the verb does.

| Binding | `prepare` | `activate` | `deactivate` |
| --- | --- | --- | --- |
| `LocalFileBinding` (Reality A) | Download from repo → cache path | PEFT `load_adapter` | PEFT `unload_adapter` |
| `EmbeddedBinding` (Reality B) | No-op (weights shipped with base model) | Render `controls` field into chat template | Drop the `controls` field |
| `ServerMediatedBinding` (Reality C) | No-op (or push weights, depending on sub-case) | Set adapter identifier on API request | Unset identifier |

`release()` is implemented per-binding as needed (cache eviction for LocalFile; no-op for the others).

> **Which class knows an adapter doesn't need PEFT activation? The binding does — not the backend.** `EmbeddedBinding.activate()` renders `controls` JSON into the chat template; `LocalFileBinding.activate()` calls PEFT `load_adapter`. The backend calls `binding.activate()` uniformly and has no conditional on binding type. This is the mechanism that eliminates the `if getattr(backend, "_uses_embedded_adapters", False):` branch (§11). When embedded-adapter support is later added to `LocalHFBackend` ([#1018](https://github.com/generative-computing/mellea/issues/1018)), the backend does not need to learn about embedding — it calls the same verbs, and `EmbeddedBinding` handles the difference. The backend only needs the verb interface. (Addressing Jacob's review question on backend consumption.)

> **Weight updates:** weights are versioned by HF commit SHA. `prepare()` resolves the configured revision (`main` by default, or a pinned SHA) and refreshes the local cache when upstream has moved. Refresh policy and the long-running-process exception are open (§17 Q5).

### 9.3 Lifecycle sequence

The lifecycle inside `adapter_scope` is the same for every binding — only the verbs do reality-specific work:

```mermaid
sequenceDiagram
    participant C as Caller
    participant B as Backend
    participant A as Adapter
    participant W as WeightsBinding
    participant M as Base Model

    C->>B: check_answerability(...)
    B->>A: resolve_adapter(name)

    rect rgb(245, 245, 245)
    Note over B,W: adapter_scope(adapter)
    B->>W: prepare()
    W-->>M: download / no-op
    B->>W: activate()
    W-->>M: load / render controls / set adapter_id
    B->>A: io_contract.build_prompt(...)
    B->>M: generate(prompt)
    M-->>B: raw output
    B->>A: io_contract.parse(raw)
    A-->>B: normalised result
    B->>W: deactivate()
    W-->>M: unload / drop controls / unset
    end

    B-->>C: score
```

## 10. Backend × reality matrix

Mellea currently exposes five backends. Adapter support varies — and is not a goal for every backend.

| Backend             | Reality A (Local PEFT) | Reality B (Embedded) | Reality C (Server-mediated) | Notes |
| ------------------- | :--------------------: | :------------------: | :-------------------------: | --- |
| `LocalHFBackend`    | ✅ today                | ⏳🔽 [#1018](https://github.com/generative-computing/mellea/issues/1018) | — | Primary local backend; only one with aLoRA support today. Primarily used for individual/local deployments rather than multi-tenant environments (Paul). |
| `OpenAIBackend`     | —                      | ✅ today ([#881](https://github.com/generative-computing/mellea/pull/881)) | ⏳🔼 [#27](https://github.com/generative-computing/mellea/issues/27) | OpenAI-compatible endpoint, including vLLM servers. |
| `OllamaBackend`     | —                      | —                    | —                           | Ollama's LoRA/PEFT story is GGUF-based and immature; not a current target. |
| `WatsonxBackend`    | —                      | —                    | —                           | Would require watsonx-side adapter support; no current plan. |
| `LiteLLMBackend`    | —                      | —                    | —                           | Multi-provider shim; adapter support would depend on the underlying provider and is not a coherent single-backend target. Could opportunistically inherit C2 if any wrapped provider exposes fine-tuned identifiers. |

Legend: ✅ supported today; ⏳🔽 planned, blocked by this proposal (downstream); ⏳🔼 planned, blocked by an upstream dependency outside Mellea; — not applicable or not planned.

**What this says about intent:**

- The two **primary adapter backends are `LocalHFBackend` and `OpenAIBackend`.** The refactor targets these first.
- Granite Switch (embedded) is the newest addition but is **not** "the premier option": local PEFT via `LocalHFBackend` remains the development/on-prem path and is the only reality that ships with both LoRA and aLoRA today.
- The remaining three backends (`OllamaBackend`, `WatsonxBackend`, `LiteLLMBackend`) are **out of scope for adapter support under this design**. The `WeightsBinding` abstraction does not preclude adding them later, but no issue currently tracks the intent and the underlying providers do not support the mechanisms Mellea's adapters need.
- The design keeps every ✅ cell working, adds clean paths for the ⏳ cells without ad-hoc branching, and leaves empty cells empty rather than stubbing them speculatively.

## 11. Why the current code is tangled (concrete example)

Part I §1 listed the symptoms; this section names the *structural* cause. The single piece of code that most clearly shows it is the branch in `_util.call_intrinsic`:

```python
if getattr(backend, "_uses_embedded_adapters", False):
    adapters = EmbeddedIntrinsicAdapter.from_source(...)
else:
    intrinsic_adapter = IntrinsicAdapter(...)  # Reality A path
```

This is a **backend-keyed dispatch** where the branching key (`_uses_embedded_adapters`) is a property of the backend rather than of the adapter. Every new reality forces a new branch, and the `else` path is not a generic fallback — it is the Reality A path, so it unconditionally calls `obtain_lora` whether or not the adapter needs downloading. The three symptoms in §1 (misleading download errors, rigid output parsing, hardcoded role strings) are *all* consequences of this same shape: "the adapter doesn't know what kind it is, so the call site guesses." The new design flips this: the **binding** says what kind it is, and the backend simply executes its verbs.


## 12. Full #929 thread mapping

| Thread | Resolution |
| --- | --- |
| 1a. Loading/unloading divergence | One `WeightsBinding` verb set; control flow identical across realities. |
| 1b. `obtain_lora` always-called bug | Only `LocalFileBinding.prepare` calls `obtain_lora`; others no-op. |
| 1c. Backend- + adapter-type-specific abstraction | `WeightsBinding` is the adapter-type axis; `AdapterMixin` verbs are the backend axis. |
| 2a. Intrinsic rewriters overwrite options | `Adapter.resolve_model_options()` replaces the five-place merge with one documented stack. |
| 2b/2c. Model-option hierarchy | Five layers enforced in `resolve_model_options` (base model → adapter config → `io.yaml` defaults → `io.yaml` per-intrinsic → caller). |
| 3. Naming consistency | Three-axis identity (`name`, `adapter_type`, `revision`) plus explicit `role`. |
| 4a. `call_intrinsic` assumes one output schema | `io_contract.parse()` validates the output shape and raises `AdapterSchemaMismatchError` when parse cannot yield the declared contract (Jake req 4); forward-compatible additions do not trigger the error. Helpers see a normalised shape. |
| 4b. Per-adapter vs standard schema | `io_contract.parse()` is per-adapter; helpers define the normalised post-parse shape. |
| 4c. Versioning | HF commit SHA is the version (every push = new revision; pin via `revision="..."` for stability). Breaking schema changes (rare) handled by pinning and by helpers raising `AdapterSchemaMismatchError` when parse cannot yield the declared contract (Jake req 4). Output-schema versioning beyond this is tracked separately in [#1111](https://github.com/generative-computing/mellea/issues/1111) (§17 Q4). |
| 5. OpenAI backend support | Ships as one or two `ServerMediatedBinding` subclasses. |
| 6. Catalog cleanup | Catalog becomes optional resolver (`LocalFileBinding.from_catalog(name)`). Custom adapters bypass it; no monkey-patching. Duplicate `requirement_check` / `requirement-check` entries collapse into one entry; the v1 → v2 output-schema change (PR #1008) is handled by Jake req 4 (helper raises when parse cannot yield the declared contract); pinning the prior HF revision is the avoidance path. |
| 7. Hardcoded `requirement-check` refs | Callers look up by **role**, not name. |

## 13. What users see — detailed

**High-level helpers** keep their signatures. The `model_options=` parameter (and `documents=` keyword on `factuality_detection` / `factuality_correction`) is added in Phase 1, folding in #1003 (PR #1028 closed in favour of this epic):

```python
score = check_answerability(question, documents, context, backend)
score = check_answerability(question, documents, context, backend,
                            model_options={"temperature": 0.1})
```

**Validation on parse.** Helpers declare their expected output shape; `io_contract.parse()` validates against it and raises `AdapterSchemaMismatchError` when the parse cannot yield the helper's declared output contract — with `name`, observed keys, and expected keys in the message. Forward-compatible additions (an extra optional field the parser ignores) do not trigger the error; contract-breaking deltas (missing required field, type change on a depended-on key) do. Schema drift is loud, not silent. (Jake req 4.)

**Manual adapter construction** collapses from four classes (`IntrinsicAdapter`, `EmbeddedIntrinsicAdapter`, `CustomIntrinsicAdapter`, abstract base) to one `Adapter` + a binding:

```python
# Adapter for the answerability intrinsic (auto-loaded from catalogue; pinned revision):
adapter = Adapter(name="answerability",
                  weights=LocalFileBinding.from_catalog("answerability"))
# Catalogue entry includes a pinned HF commit SHA (Jake req 5).
# Pass revision="main" to LocalFileBinding directly to override and track latest.

# Adapter for a custom intrinsic — io.yaml auto-loaded from the same HF repo:
adapter = Adapter(name="my-thing",
                  weights=LocalFileBinding(source="myuser/my-adapter",
                                           base_model_name="granite-4.1-3b"))
# To override io_contract with a local file:
# adapter = Adapter(..., io_contract=IOContract.from_yaml("./io.yaml"))

# Adapter for the Granite Switch embedded variant:
adapter = Adapter(name="answerability",
                  weights=EmbeddedBinding.from_base_model(backend))
```

**Backend authors** keep `AdapterMixin` as the backend surface, but it exposes only the verbs a backend naturally has: `load_peft_adapter`, `unload_peft_adapter`, `render_controls`, `set_request_adapter`. Bindings call into these verbs. Adding a new reality = adding a new verb + new binding.

## 14. Observability

### 14.1 Why adapters need bespoke observability

Adapter calls hide the complexity that matters most when something goes wrong (weight fetching, activation side-effects, schema contracts). Without per-phase instrumentation, four failure modes are hard or impossible to diagnose — and Mellea has already hit the first two in production:

1. **Masked errors.** The `obtain_lora`-always-called bug (#929 point 1b) showed users a misleading download error while the real cause (adapter-type mismatch) stayed invisible. A span at the `prepare` boundary recording the exception would have surfaced the actual cause on first run.
2. **Silent schema drift.** When PR #1008 changed `requirement-check` output from `{"requirement_likelihood": 0.9}` to `{"requirement_check": {"score": 0.9}}`, `requirement_check_to_bool` silently returned `False` for every call until someone noticed. Under Jake req 4 (helpers raise when parse cannot yield the declared contract), this would have surfaced as `AdapterSchemaMismatchError` on the first call after the schema change — the caller gets a named error instead of a silently wrong value. The `parse_failures` counter labelled by `(name, revision)` is the dashboard signal; the exception is the runtime signal.
3. **Latency attribution.** "`check_answerability` is slow" is unanswerable today — download, PEFT load, generation, and JSON parse collapse into one backend span. Phase-level spans make the culprit obvious in any trace viewer.
4. **Alerting and cost attribution.** OTel `ERROR` status on failed download/activation makes generic dashboards and alerts work. Token counts labelled by adapter answer "which capability is 30% of our spend?" Both impossible today.

Adding instrumentation now costs one span attribute per verb. Retrofitting after the refactor means re-editing every binding. And during a refactor this wide, the fastest way to spot a regression in a specific reality is a dashboard, not a bug report.

### 14.2 Spans and metrics

**Spans** — each `adapter_scope` wraps a child span tree rooted at `intrinsic.call`:

```mermaid
graph TD
    root["intrinsic.call<br/><i>name, revision, role,<br/>binding_type, adapter_type</i>"]
    root --> prep["intrinsic.prepare<br/><i>LocalFile: download ms</i>"]
    root --> act["intrinsic.activate<br/><i>peft_name / controls / api_id</i>"]
    root --> gen["intrinsic.generate<br/><i>(regular backend span:<br/>tokens, latency)</i>"]
    root --> par["intrinsic.parse<br/><i>revision, parse_ok, raw_len</i>"]
    root --> deact["intrinsic.deactivate"]
```

Standard attributes: `intrinsic.name`, `intrinsic.revision`, `intrinsic.role`, `intrinsic.adapter_type`, `intrinsic.binding_type`, `intrinsic.source`, `intrinsic.target`. Errors set OTel `ERROR` status (aligns with #1035 gap 4).

**Metrics** — an `IntrinsicMetricsPlugin` alongside the existing Token / Latency / Error plugins:
- `mellea.intrinsic.invocations` — counter labelled by name, revision, binding type, adapter type, outcome.
- `mellea.intrinsic.phase_duration_ms` — histogram labelled by name, phase.
- `mellea.intrinsic.parse_failures` — counter labelled by name, revision. This is the **schema-drift detector**: a climbing counter against a specific `(name, revision)` pair means an upstream adapter pushed a breaking schema change at a new HF revision that the local parser doesn't yet handle. Each increment matches an `AdapterSchemaMismatchError` raised at the call site (Jake req 4).

**Content capture** — gated behind PR #1036's `MELLEA_TRACE_CONTENT` flag. Intrinsics emit `intrinsic.input.kwargs` (structured dict), `intrinsic.output.raw` (raw JSON string), and `intrinsic.output.parsed` (normalised shape) as span events. Different shape from chat `gen_ai.*.message` events because intrinsics have different semantics.

## 15. Docs, tests, tutorials

First-class deliverables, not afterthoughts.

**Docs** — rewrite (not edit) for `docs/dev/intrinsics_and_adapters.md` (39 lines describing classes that get renamed). Update `docs/dev/requirement_aLoRA_rerouting.md` to describe role-based lookup instead of hardcoded strings. User-facing `docs/docs/advanced/intrinsics.md` and examples under `docs/examples/intrinsics/` are breaking-API touched. New dev doc for adapter observability. Update AGENTS.md §13 once normalised post-parse shapes are stable.

**Tests** — existing adapter tests stay green per phase. New tests cover: each binding × each verb (unit); integration matrix `{HF, OpenAI} × {applicable bindings} × {lora, alora where applicable} × {every existing adapter}`; per-version parse round-trips (with `requirement-check` v1 / v2 as the worked case); concurrency window correctness; span/metric emission assertions.

**Qualitative effectiveness suite (optional, per-adapter).** The tests above verify plumbing. They do *not* answer "does the answerability adapter actually judge answerability correctly?" A per-adapter qualitative suite (`@pytest.mark.qualitative`, opt-in, kept out of the fast loop) takes a small canonical dataset per adapter and asserts an accuracy floor on its outputs. Without this, a refactor can pass every structural test while silently degrading the behaviour users care about.

The implementation approach for this suite is intentionally left open — start simple and file a separate proposal if a more structured approach (e.g. `TestBasedEval`, `BenchDrift`) is warranted.

**Tutorials** — three worth writing alongside the refactor:
- "Adding a custom intrinsic in 20 lines" — replaces the `CustomIntrinsicAdapter` monkey-patch story.
- "Handling a breaking schema change without breaking users" — worked example using `requirement-check` v1 → v2; covers HF revision pinning and `AdapterSchemaMismatchError` (Jake req 4).
- "Reading intrinsic telemetry" — short dashboard-building guide.

**Release notes** separate: no-op for high-level helper users; deprecated-but-shimmed for direct adapter constructors; removed at Phase 4 (see below).

## 16. Migration (rough shape only)

Detail deferred until Part I §5 decisions are agreed, but the intended phasing is:

1. **Phase 0 — parallel types.** Introduce the new types (`Adapter`, `WeightsBinding`, `IOContract`, plus a user-facing `Intrinsic` class if Q5 is settled on Jake's split) alongside existing classes. Catalogue entries gain pinned HF revision SHAs (Jake req 5; §17 Q6). No call-site changes, tests unchanged.
2. **Phase 1 — callers move.** `_util.call_intrinsic`, requirement rerouting, and each helper switch to new types. Helpers gain output validation raising `AdapterSchemaMismatchError` when parse cannot yield the declared contract (Jake req 4). #1003 helper signature work folds in here: `model_options=` on all top-level helpers; `documents=` keyword-only on `factuality_detection` / `factuality_correction`. Auto-context document discovery for `documents=None` lifted from PR #1028 (§17 Q3); mechanism refined per intrinsics-team guidance — helpers read documents from ordinary conversation context, not from a `_docs`-specific scan path. Old classes become deprecation shims.
3. **Phase 2 — backends move.** `AdapterMixin` narrows to the new verb set. Bindings implement `prepare` / `activate` / `deactivate` / `release` per reality; `LocalFileBinding.prepare` resolves the configured HF revision (§17 Q5 weight-refresh policy). Backends drop per-call `_simplify_and_merge` in favour of `resolve_model_options`.
4. **Phase 3 — Reality C ships.** `ServerMediatedBinding` subclass(es) written; OpenAI backend drops `_uses_embedded_adapters` hard-code.
5. **Phase 4 — shim removal.** After one minor release with deprecation warnings. *(Skipped if Q5 settles on Jake's split — re-exports stay.)*

Observability and docs deliverables attach to the phase that first exercises them.

## 17. Open questions and implementation positions

Items marked **[Open]** need decision; **[Position]** is the proposal's working answer (reviewers can push back); **[Resolved]** has explicit reviewer agreement. Decisions that gate decomposition are in Part I §5; this section is for implementation-level questions and positions.

1. **Naming `WeightsBinding`** [Position]. Used throughout the doc; alternatives `ResourceStrategy` / `AdapterProvider` were considered. `WeightsBinding` is concrete (says what it binds) and unambiguous in error messages.
2. **Role vs name** [Position]. `role` is a free-form string with an advisory known-roles registry (e.g., `mellea.backends.adapters.roles.KNOWN_ROLES`). Backends warn on unknown roles but accept any string. Pure enum was considered but rejected — it would lock role names at library-release time.
3. **Rewind interaction (formerly PR #1028).** Two parts:
   - **Where rewind logic lives** [Resolved] (Jake on PR #1080): the rewind in `_resolve_question` / `_resolve_response` stays in the helpers. Phase 1 can revisit moving it to `io_contract.build_prompt` if cleaner separation is wanted; not gating.
   - **Document discovery when `documents=None`** [Resolved] (Jake on PR #1080, 2026-05-20). When `documents=None` is passed to `factuality_detection` / `factuality_correction`, the helper auto-discovers user-supplied documents from the conversation context rather than requiring an explicit `documents=` argument. The auto-discovery direction was the contribution of PR #1028; intrinsics-team guidance refines the *mechanism*: documents flow through ordinary conversation context (no `_docs`-scanning fallback path needed). Phase 1 implements helpers that read whatever documents are present in the context they receive; populating that context is the caller's responsibility (explicit `documents=`, prior `Message`s, retrieval, …). PR #1028's specific `_resolve_response` `_docs`-scanning code is shelved.
4. **Output-schema versioning** [Resolved — Defer to [#1111](https://github.com/generative-computing/mellea/issues/1111)]. This refactor assumes `io.yaml` does **not** carry a `schema_version` field, and Mellea does not introduce one. Forward-compatibility is preserved: helpers only raise `AdapterSchemaMismatchError` when parse cannot yield the declared contract, not on benign additions. Versioning is tracked separately in #1111; promote to in-progress when the trigger conditions documented there are hit.
5. **Weight-refresh policy** [Position]. Adapter weights are versioned by HF commit SHA. `prepare()` re-resolves the upstream revision at session start; long-running processes (sessions spanning a release) opt into an explicit `refresh()` API. Default cadence matches the session-scoped lifecycle (Part I §5 Q2 Resolved).
6. **Version pinning for auto-loaded adapters** [Position]. When an adapter is auto-loaded from the catalogue (caller didn't specify a revision), Mellea pins to the catalogue entry's recorded SHA. `revision="main"` is an explicit opt-in to track latest. Pinning gives reproducibility; explicit tracking gives latest weights at the cost of behaviour drift between runs. (Jake req 5; coupled to Q5 weight-refresh policy.)

---

# Appendix — Referenced issues and PRs

Linked index of every issue, PR, and commit cited in this document. Use this to jump to primary sources.

### Tracking items (open, design-relevant)

| Ref | Title | Relevance |
| --- | --- | --- |
| [Epic #929](https://github.com/generative-computing/mellea/issues/929) | Fix Intrinsic Adapter Lifecycle & Consistency in Mellea | *the epic this proposal addresses* |
| [#27](https://github.com/generative-computing/mellea/issues/27) | Add support for aloras to remote vllm when vllm supports it | live tracking item for Reality C |
| [#423](https://github.com/generative-computing/mellea/issues/423) | Adapter code is undocumented and over-specialized to Intrinsics | Priority-labelled; overlaps this refactor |
| [#424](https://github.com/generative-computing/mellea/issues/424) | Cannot use intrinsics without uploading them | customer-adapter friction |
| [#1018](https://github.com/generative-computing/mellea/issues/1018) | add support for granite-switch / embedded adapters on HF backend | explicitly sequenced after this refactor |

### History and rework evidence

| Ref | Title | Role in this doc |
| --- | --- | --- |
| [#543](https://github.com/generative-computing/mellea/pull/543) | revert: remove adapters/intrinsics/alora/lora from openai code | why OpenAI backend lost adapter support (upstream vLLM declined aLoRA PR) |
| [#881](https://github.com/generative-computing/mellea/pull/881) | feat: add embedded adapters (granite switch) to openai backend | why OpenAI backend got Reality B back |
| [#946](https://github.com/generative-computing/mellea/pull/946) | feat: simplify intrinsics | rework evidence |
| [#972](https://github.com/generative-computing/mellea/pull/972) | fix: model options with intrinsics | rework evidence for #929 point 2 |
| [#979](https://github.com/generative-computing/mellea/pull/979) | fix: key in json returned by policy_guardrails intrinsic | rework evidence for output parsing |
| [#986](https://github.com/generative-computing/mellea/pull/986) | fix: issues introduced by intrinsic changes | rework evidence |
| [#994](https://github.com/generative-computing/mellea/pull/994) | fix: default intrinsic adapter types; granite-switch tests | rework evidence |
| [#1008](https://github.com/generative-computing/mellea/pull/1008) | fix: rewrite requirement_check_to_bool for new schema | worked example for the contract-mismatch story (Jake req 4) |
| [#1028](https://github.com/generative-computing/mellea/pull/1028) | feat: normalize intrinsics interfaces | introduces the factuality rewind path |

#### Rework evidence in detail

Seven recent fix-up commits in the adapter area, all symptomatic of the design gaps described in §1 rather than straightforward feature work. Referenced from §1 as evidence that this is friction, not theory:

| Commit / PR | What it fixed |
| --- | --- |
| `1734900d` | Remove `answer_relevance*` intrinsics and unrelated intrinsic issues. |
| `8b6b8d55` ([#972](https://github.com/generative-computing/mellea/pull/972)) | Model options with intrinsics (precedence bug surfaced). |
| `c57aba1d` ([#986](https://github.com/generative-computing/mellea/pull/986)) | Issues introduced by preceding intrinsic changes. |
| `8577d092` ([#994](https://github.com/generative-computing/mellea/pull/994)) | Default intrinsic adapter types; canned I/O with temperature. |
| `4d372b0e` ([#979](https://github.com/generative-computing/mellea/pull/979)) | Key in JSON returned by `policy_guardrails` intrinsic. |
| `0617bd96` ([#1008](https://github.com/generative-computing/mellea/pull/1008)) | Rewrote `requirement_check_to_bool` for a changed output schema; flipped `"requirement_check"` → `"requirement-check"` in four files. |
| `75465d29` ([#946](https://github.com/generative-computing/mellea/pull/946)) | "Simplify intrinsics" — reacting to accumulated complexity. |

### Related in-flight and planned work

| Ref | Title | Role in this doc |
| --- | --- | --- |
| [#1003](https://github.com/generative-computing/mellea/issues/1003) | fix: intrinsic function signatures | folded into Phase 1 of this epic; PR #1028 closed 2026-05-15 |
| [PR #1028](https://github.com/generative-computing/mellea/pull/1028) | feat: normalize intrinsics interfaces | closed 2026-05-15 in favour of folding into this epic. Two threads inherited: (1) #1003 helper signatures → Phase 1 (already scoped); (2) auto-context document discovery → Phase 1 (§17 Q3 Resolved 2026-05-20; mechanism refined to ordinary-context reading, not `_docs` scanning). |
| [#1035](https://github.com/generative-computing/mellea/issues/1035) | OTel emission gaps | parent for telemetry coordination |
| [PR #1036](https://github.com/generative-computing/mellea/pull/1036) | feat(telemetry): close five OTel GenAI semconv gaps | in-flight telemetry work to coordinate with |

### Sequencing

**Why [#1018](https://github.com/generative-computing/mellea/issues/1018) waits for this proposal:**

- #1018's own body states: *"May require sorting out some of the issues in #929 first. Or at least creating a comprehensive plan."*
- Once Part I is agreed and Phase 0–2 of the migration have merged, #1018 reduces to *"add the `EmbeddedBinding` path to `LocalHFBackend`"* following the pattern already used for `OpenAIBackend`.
- Attempting #1018 without this refactor re-creates the same branching problem on a second backend.

### Verification trail

Verified against: `mellea/backends/adapters/{adapter,catalog,__init__}.py`, `mellea/stdlib/components/intrinsic/{_util,intrinsic,core,rag,guardian}.py`, `mellea/backends/{openai,huggingface}.py`, `mellea/formatters/granite/intrinsics/input.py`, `mellea/stdlib/requirements/requirement.py`, `docs/dev/{intrinsics_and_adapters,requirement_aLoRA_rerouting}.md`; commits `666d646a`, `8b6b8d55`, `c57aba1d`, `8577d092`, `c6a3e643` (aLoRA → PEFT 0.18.1 migration).
