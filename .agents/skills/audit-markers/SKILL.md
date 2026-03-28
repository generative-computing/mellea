---
name: audit-markers
description: >
  Audit and fix pytest markers on test files and examples. Classifies tests as
  unit/integration/e2e/qualitative using general heuristics and project-specific
  marker rules. Estimates GPU VRAM and RAM requirements by tracing model
  identifiers and looking up parameter counts.
  Use when: writing a new test and unsure which markers to apply; reviewing or
  auditing existing test markers; a test is unexpectedly skipped or not collected;
  a test is consuming too much GPU/RAM and you want to check its resource gates;
  checking marker correctness before committing; or any question about why a test
  does or doesn't run in a given configuration.
argument-hint: "[file-or-directory] [--dry-run | --apply]"
compatibility: "Claude Code, IBM Bob"
metadata:
  version: "2026-03-26"
  capabilities: [read_file, write_file, bash, grep, glob]
---

# Audit & Fix Pytest Markers

Classify tests, validate markers, and fix issues. Works in two layers:
general test classification (applicable to any project) plus project-specific
marker rules for **mellea**.

## Inputs

- `$ARGUMENTS` — file path, directory, or glob. If empty, audit `test/` and `docs/examples/`.
- **No flags (default)** — produce report, then ask user to confirm before applying.
- `--apply` — produce report and apply fixes without asking.
- `--dry-run` — report only, do not offer to apply.

Single-file use: skip the triage phase (Step 0) entirely — deep-read the file directly and proceed from Step 1.

## Modes of use

This skill has two modes. Choose based on what the user asked:

### Audit mode (default)
User wants markers classified or fixed — for a new test, an existing file, or a
pre-commit check. Follow the full Audit Procedure (Steps 0–4).

### Diagnostic mode
User wants to know **why** a specific test is not running, is being skipped, or
is consuming unexpected resources. Do NOT produce an audit report. Instead:

1. **Read the test** — identify its markers and any predicate decorators.
2. **Check the default filter** — read `pyproject.toml` `[tool.pytest.ini_options]`
   `addopts`. The project default is `-m "not slow"`. If the test has `slow`, it
   is excluded from a plain `uv run pytest` run.
3. **Check backend auto-skip** — read `test/conftest.py` `pytest_configure` and
   the `pytest_collection_modifyitems` hook. Backend markers (`ollama`, `huggingface`,
   etc.) trigger auto-skip when the backend is unavailable. Check whether the
   relevant service or credentials are present on the user's machine.
4. **Evaluate predicates** — if the test has a predicate decorator (`require_gpu`,
   `require_api_key`, `require_ram`, etc.), read `test/predicates.py` and explain
   what condition would cause the skip. For `require_gpu(min_vram_gb=N)`, compare N
   against the system's detected VRAM (run `get_system_capabilities()` logic or
   check `sysctl hw.memsize` on Apple Silicon / `nvidia-smi` on CUDA).
5. **Report directly** — answer "this test is skipped because X" with the specific
   condition, the value it evaluated to, and how to override if appropriate (e.g.
   `uv run pytest test/path/test_foo.py` bypasses the `-m "not slow"` default filter).

For resource overload (test consuming too much GPU/RAM): classify the test's
resource gates using the VRAM heuristics in Part 2, compare against what the
test actually loads, and report whether the gate is correctly set or too loose.

## Project References

Read these before auditing — they are the authoritative source for marker conventions:

- **Marker guide:** `test/MARKERS_GUIDE.md`
- **Marker registration:** `test/conftest.py` (`pytest_configure`) and `pyproject.toml` (`[tool.pytest.ini_options]`)
- **Resource predicates:** `test/predicates.py` (predicate functions for resource gating)
- **Example marker format:** `docs/examples/conftest.py` (`_extract_markers_from_file`)

---

# Part 1: General Test Classification

These principles apply to any test suite, not just mellea. Use them as the
foundation for classifying every test function.

## The Four Granularity Tiers

Tests fall into exactly one tier based on what they exercise and what they need:

### Unit

**Entirely self-contained** — no services, no I/O, no fixtures that connect
to anything external. Pure logic testing.

**Recognise by:**
- Imports only from the project and stdlib — no external service clients
- Creates objects directly, calls methods, checks return values
- If it uses test doubles, they replace all external boundaries (network, DB, services, third-party SDKs)
- Third-party library is imported only as a type or helper, not as a real collaborator being asserted against
- No fixture that starts/connects to a real or fixture-managed service
- Runs in milliseconds to low seconds
- Would pass identically if you replaced a real SDK import with a stub of the same interface

**Examples of unit assertions:**
```python
assert str(cb) == "hello"
assert len(items) == 3
assert raises(ValueError)
mock_backend.generate.assert_called_once()
```

### Integration

**Verifies that your code correctly communicates across a real boundary.**
The boundary may be a third-party SDK/library whose API contract you are
asserting against, multiple internal components wired together, or a
fixture-managed local service. What distinguishes integration from unit is
that at least one real external component — not a mock or stub — is on the
other side of the boundary being tested.

**Key distinction from unit:** The boundary is not limited to network or
hardware. A test that wires project code against a real third-party SDK
object to assert on output format or values is integration — even when
entirely in-memory with no network I/O. The question is whether a real
external component's API contract is being verified, not whether there is
network activity.

**Key distinction from e2e:** Integration controls or provides its
dependencies (mocks, in-memory SDK components, fixture-managed local
services). E2E depends on real backends that exist independently (Ollama,
cloud APIs, GPU-loaded models).

**Positive indicators:**

- Uses a real third-party SDK object to *capture and assert* on output —
  e.g. `InMemoryMetricReader`, `InMemorySpanExporter`, `LoggingHandler` —
  rather than patching the SDK away
- Asserts on format or content of data as received by the external component
  (semantic conventions, attribute names, accumulated values)
- Wires multiple real project components together and mocks only at the
  outermost boundary (LLM call, network, hardware)
- Breaking the interface between your code and the external component (e.g.
  a changed attribute name, a missing SDK method call) would cause the test
  to fail
- Fixture-managed dependencies that stand up or configure real local services

**Negative indicators (likely unit instead):**

- All external boundaries replaced with `MagicMock`, `patch`, or `AsyncMock`
- Third-party library imported only as a type or helper, not as a real
  collaborator being asserted against
- Toggles env vars and checks booleans or config state with no real SDK
  objects instantiated
- Only one real component under test; everything else is faked

**Tie-breaker:** If you changed the contract between your code and the
external component (e.g. renamed an attribute, stopped calling the right
SDK method), would this test catch it? If yes → integration. If no → unit.

**Examples:**

```python
# Integration — real OTel InMemoryMetricReader, asserting SDK contract
@pytest.mark.integration
def test_token_metrics_attributes(clean_metrics_env):
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    record_token_usage_metrics(input_tokens=10, output_tokens=5, ...)
    # Asserts against real OTel output — breaking the attribute name would fail this
    assert attrs["gen_ai.provider.name"] == "ollama"

# Integration — multiple real project components, only LLM call mocked
@pytest.mark.integration
def test_session_chains_components(mock_backend):
    session = start_session(backend=mock_backend)
    result = session.instruct("hello")
    assert mock_backend.generate.called

# Unit — real OTel SDK imported but only for isinstance check on a no-op
def test_instruments_are_noop_when_disabled(clean_metrics_env):
    counter = create_counter("test.counter")
    assert counter.__class__.__name__ == "_NoOpCounter"
```

### E2E (End-to-End)

Tests against **real backends** — cloud APIs, local servers, or GPU-loaded
models. No mocks on the critical path.

**Recognise by:**
- Uses a real backend (however started — cloud API, local server, script-launched, GPU-loaded)
- Needs infrastructure: running server, API key, GPU, sufficient RAM
- Fixtures create real service connections, not mocks
- Assertions check that the real service behaved correctly
- Assertions are **deterministic** — structural, type-based, or functional

**Examples of e2e assertions:**
```python
assert isinstance(result, CBlock)              # type check
assert json.loads(result.value)                 # valid JSON
assert result._meta["status"] == "complete"     # status check
assert tool_call.function.name == "get_weather" # tool was invoked
assert result.parsed_repr is not None           # output exists
```

### Qualitative

Subset of e2e. Same infrastructure requirements, but assertions are on
**non-deterministic output content** that may vary across model versions,
temperatures, or runs.

**Recognise by:**
- Same as e2e (real backend, real calls)
- Assertions check semantic content, natural language output, or quality
- A different model version could break the assertion even if the system works correctly

**Examples of qualitative assertions:**
```python
assert "hello" in result.value.lower()          # content check
assert result.value.startswith("Subject")       # format of generated text
assert len(result.value.split()) > 50           # output length
assert "error" not in result.value.lower()      # absence of bad content
```

**The decision rule:** If swapping the model version could break the assertion
despite the system working correctly → `qualitative`. If the assertion checks
structure, types, or functional correctness → `e2e`.

## Behavioural Signal Detection

Before deep-reading a file, grep-able signals reveal whether it is likely unit
or non-unit. Use these to triage at scale (see Audit Procedure, Step 0).

### Live-backend signals (test is likely NOT unit)

| Category | Grep patterns | Notes |
|---|---|---|
| Network literals | `localhost`, `127.0.0.1`, `0.0.0.0`, port numbers (`:11434`, `:8000`) | Direct infra dependency |
| HTTP clients | `requests.get(`, `httpx.`, `aiohttp.ClientSession`, `urllib.request.urlopen` | Real network unless mocked |
| Raw networking | `socket.socket(`, `socket.connect(` | Low-level network |
| Subprocess | `subprocess.Popen(`, `subprocess.run(`, `subprocess.call(`, `os.system(` | Spawns external process |
| API credentials | `_API_KEY`, `_TOKEN`, `_SECRET` in `os.environ`/`os.getenv` calls | Credential dependency |
| GPU / model loading | `import torch`, `.to("cuda")`, `.from_pretrained(` | Hardware dependency |
| External downloads | URL literals (`http://`, `https://`), `urlopen`, `requests.get` with URLs | Network fetch |

### Training memory signals (check `require_gpu` threshold)

Training tests consume significantly more memory than inference. When these
patterns appear, verify that `require_gpu(min_vram_gb=N)` uses the **training
peak**, not just the model parameter size:

| Category | Grep patterns | Notes |
|---|---|---|
| Model load | `from_pretrained(`, `AutoModelForCausalLM`, `AutoModelForSeq2SeqLM`, `AutoTokenizer` | Downloading/loading a real model |
| Training | `train_model(`, `Trainer(`, `trainer.train(`, `epochs=`, `num_train_epochs=` | Any training loop |
| Inference on real model | `.generate(` in a test body without a mock | Full model forward pass |
| HF dataset download | `load_dataset(` | Dataset fetch + tokenisation |

**Training memory rule:** Training requires ~2× the base model weight memory
(activations, optimizer states, gradient temporaries). A test that trains then
reloads the model for inference has two separate peaks — use the training peak
plus headroom:

```
min_vram_gb = (model_param_bytes_in_bfloat16 * 2) + headroom
            ≈ (params_B * 2 GB) * 2  + ~4 GB headroom
```

Example: 3B bfloat16 model → 6 GB weights → training peak ~12 GB → set
`require_gpu(min_vram_gb=20)` so the gate fires on machines where available
GPU memory is below that, rather than letting the test OOM mid-run.

The VRAM heuristic (`predicates.py`) reports *estimated available* memory, not
total RAM. A 32 GB Apple Silicon machine reports ~16 GB available. Setting
`min_vram_gb=20` correctly skips on that machine while running on 48 GB+.

### Adapter accumulation signals (module-scoped backend with multiple intrinsics)

When a test module uses a **module-scoped backend fixture** and calls multiple different intrinsic
functions (`call_intrinsic`, `guardian_check`, `policy_guardrails`, `guardian_check_harm`, etc.)
against the same backend, each intrinsic loads a separate LoRA adapter that stays resident in
`_loaded_adapters` for the lifetime of the fixture. Adapters do NOT auto-unload between tests.

| Signal | Pattern | Notes |
|---|---|---|
| Module-scoped HF backend | `scope="module"` on a `LocalHFBackend` fixture | Adapter accumulation possible |
| Multiple intrinsic calls | `call_intrinsic(`, `guardian_check(`, `policy_guardrails(`, `factuality_` | Each loads a distinct adapter |
| No `unload_adapter` | Absence of `backend.unload_adapter(` in fixture teardown | Adapters pile up |

**Memory estimate for modules with adapter accumulation:**

```
min_vram_gb = base_model_gb + (N_distinct_intrinsics × ~0.2 GB) + inference_overhead_gb
```

For `test_guardian.py` (6 tests, 3B base model, ~4 distinct adapters):
- Base model: ~6 GB
- 4 adapters: ~0.8 GB
- Inference overhead: ~2 GB
- Total: ~9 GB minimum → use `require_gpu(min_vram_gb=12)` for headroom

Flag any module where `scope="module"` backend + multiple distinct intrinsic calls has
`require_gpu(min_vram_gb=N)` set only to cover the base model size.

### SDK-boundary signals (test is likely integration, not unit)

These patterns indicate real third-party SDK objects are being used as
collaborators to assert on output — not mocked away:

| Category | Grep patterns | Notes |
|---|---|---|
| OTel metrics | `InMemoryMetricReader`, `MeterProvider(metric_readers=` | Asserting against real OTel metrics output |
| OTel tracing | `InMemorySpanExporter`, `SimpleSpanProcessor`, `TracerProvider(` | Asserting against real OTel span output |
| OTel logging | `LoggingHandler`, `LoggerProvider`, `set_logger_provider` | Asserting against real OTel log output |
| Real SDK setup | `provider.force_flush()`, `reader.get_metrics_data()` | Consuming real SDK output |

Cross-reference: if these appear alongside `@patch(` that patches the SDK
itself away → unit. If the SDK objects are instantiated and used directly →
integration.

### Mock signals (test is likely unit)

| Category | Grep patterns |
|---|---|
| Mock objects | `MagicMock`, `Mock(`, `AsyncMock`, `create_autospec` |
| Patching | `@patch(`, `@mock.patch`, `monkeypatch`, `mocker` fixture |
| HTTP mocks | `responses`, `respx`, `httpx_mock`, `aioresponses`, `vcr` |

### Fixture signals (need chain tracing to resolve)

| Signal | Likely tier |
|---|---|
| `tmp_path`, `capsys`, `monkeypatch`, `caplog` only | Unit |
| Custom fixtures named `session`, `backend`, `m_session` | Could be real or mock — trace the chain |
| Session/module-scoped fixtures (`scope="session"`) | Usually infra setup → e2e |
| Fixture name starts with `mock_`, `fake_`, `stub_` | Unit |

### Cross-referencing signals

A single file may contain multiple signal types. Cross-reference to determine
the correct bucket:

| Live-backend signals? | SDK-boundary signals? | Mock signals? | Classification |
|---|---|---|---|
| Yes | Any | No | Almost certainly e2e — deep-read to confirm |
| Yes | Any | Yes | Needs inspection — partial mock = integration, or mixed file |
| No | Yes | No | Likely integration — deep-read to confirm SDK objects are asserted against, not just imported |
| No | Yes | Yes | Needs inspection — if SDK is patched away → unit; if used directly → integration |
| No | No | Yes | Likely unit — skip deep read |
| No | No | No | Likely unit — skip deep read |

## When to Ask the User

Some classifications are ambiguous. **Ask for confirmation** when:

- A test mixes structural and content assertions (e2e vs qualitative)
- A test uses a real backend but only checks that no exception was raised (could be e2e or integration if the backend call is incidental)
- A test patches some but not all external boundaries (partial mock — unit or integration?)
- An assertion is borderline: `assert len(result.value) > 0` could be structural (e2e) or content-dependent (qualitative)

When asking, present the test code and your reasoning so the user can make an informed decision.

---

# Part 2: Project-Specific Rules

Read `test/MARKERS_GUIDE.md` for the full marker reference (marker tables,
resource gates, auto-skip logic, common patterns). This section covers only
the **code analysis heuristics** the skill needs to classify tests — things
that require reading the test source code rather than looking up a table.

## Key project rules

- `unit` is auto-applied by conftest — **never write it explicitly**
- `llm` is deprecated (synonym for `e2e`) — **flag and recommend replacing**.
  This applies to both `pytestmark` lists and `# pytest:` comments in examples.
- Backend/resource markers only go on `e2e`/`qualitative` tests
- `qualitative` is per-function; module carries `e2e` + backend markers.
  **Exception:** if every test function in the file is qualitative, module-level
  `qualitative` is acceptable to avoid repetitive per-function decorators.
- If a file mixes unit and non-unit tests, apply markers per-function, not module-level

## Backend detection heuristics

When classifying a test file, check ALL of the following to determine which
backend(s) it uses:

- **Imports:** `from mellea.backends.ollama import ...` → `ollama`
- **Session creation:** `start_session("ollama", ...)` → `ollama`; bare `start_session()` with no backend arg → `ollama` (default backend)
- **Backend constructors:** `OllamaModelBackend(...)` → `ollama`; `OpenAIBackend(...)` → `openai`
- **Environment variables checked:** `OPENAI_API_KEY` → `openai`; `WATSONX_API_KEY` → `watsonx`
- **Dual backends:** `OpenAIBackend` pointed at Ollama's `/v1` endpoint → both `openai` AND `ollama` (but NOT `requires_api_key`)

### Project-specific triage signals

These supplement the general behavioural signals (Part 1) with mellea patterns:

| Signal | Grep pattern | Implies |
|---|---|---|
| Backend import | `from mellea.backends.` | e2e (which backend depends on module) |
| Session creation | `start_session(` | e2e, default ollama |
| Backend constructor | `OllamaModelBackend(\|OpenAIBackend(\|LocalHFBackend(\|LocalVLLMBackend(\|WatsonxAIBackend(\|LiteLLMBackend(` | e2e |
| Example marker comment | `# pytest:` | Already classified — validate |
| Ollama port | `11434` | e2e, ollama |

## Fixture chain tracing

**This is the most important analysis step.** A test's tier depends on what its
fixtures actually provide. The test function signature alone is not enough — you
must trace each fixture back to its definition to determine whether it connects
to a real backend, a mock, or nothing external.

### How to trace

1. **Read the test function signature.** List every fixture parameter
   (e.g., `session`, `backend`, `m_session`, `gh_run`).
2. **Locate each fixture definition.** Check (in order):
   - Same file (local `@pytest.fixture` functions)
   - Nearest `conftest.py` in the same directory
   - Parent `conftest.py` files up to `test/conftest.py`
   - Root `conftest.py` or plugin-provided fixtures
3. **Follow the chain recursively.** If a fixture depends on another fixture,
   trace that one too. Stop when you reach a leaf: a constructor, a mock, or
   a conftest-provided value.
4. **Classify the leaf.** The leaf determines the tier:
   - **Real backend constructor** (`OllamaModelBackend()`, `LocalHFBackend()`,
     `LocalVLLMBackend()`, `OpenAIBackend()`, `WatsonxAIBackend()`,
     `LiteLLMBackend()`) → **e2e**
   - **`start_session()`** (no mock involved) → **e2e** (default backend is ollama)
   - **Subprocess that starts a server** (`subprocess.Popen(["vllm", "serve", ...])`) → **e2e**
   - **Mock/MagicMock/patch** replacing the backend → **unit** (if self-contained)
     or **integration** (if wiring multiple real components around the mock)
   - **No external dependency at all** → **unit**

### Common fixture chain patterns in this project

**Pattern 1 — Direct session creation (e2e):**
```
test_func(session) → session fixture → start_session() → real ollama
```
Backend: `ollama`. Tier: e2e.

**Pattern 2 — Backend → session chain (e2e):**
```
test_func(session) → session(backend) → backend fixture → LocalHFBackend(...)
```
Backend: `huggingface`. Tier: e2e.

**Pattern 3 — Process → backend → session chain (e2e):**
```
test_func(m_session) → m_session(backend) → backend(vllm_process) → vllm_process spawns subprocess
```
Backend: `vllm` (via OpenAI client). Tier: e2e.

**Pattern 4 — OpenAI-via-Ollama (e2e, dual markers):**
```
test_func(m_session) → m_session(backend) → OpenAIBackend(base_url="...ollama.../v1", api_key="ollama")
```
Backend markers: `openai` + `ollama`. NOT `requires_api_key`.

**Pattern 5 — Mocked backend (unit or integration):**
```
test_func(session) → session uses MagicMock/MockBackend/patch
```
If the test only checks the mock was called → **unit**.
If the test wires real components around the mock → **integration**.

**Pattern 5b — Real SDK collaborator (integration):**
```
test_func(clean_metrics_env) → creates InMemoryMetricReader() + MeterProvider()
                              → calls project code → asserts on reader output
```
No network, no backend — but a real OTel SDK object is on the other side of
the boundary being asserted against → **integration**.

**Pattern 6 — No backend at all (unit):**
```
test_func() — or test_func(tmp_path, capsys, ...)
```
Only uses pytest built-in fixtures. Tier: **unit**.

### What to watch for

- **`gh_run` fixture** (from root conftest) — provides CI flag, does NOT indicate
  a backend. Ignore for classification purposes.
- **`autouse` fixtures** — `aggressive_cleanup`, `normalize_ollama_host`,
  `auto_register_acceptance_sets` are infrastructure. They do not affect tier.
- **Conditional fixture bodies** — some fixtures branch on `gh_run` to choose
  model IDs or options. The backend is still real in both branches → still e2e.
- **`pytest.skip()` inside fixtures** — a fixture that skips on CI
  (e.g., watsonx) is still e2e when it runs.
- **`MagicMock` vs real instance** — if a fixture returns `MagicMock(spec=Backend)`,
  the test is NOT e2e regardless of what the test function does with it.
- **Mixed files** — a file might define both a real `backend` fixture (used by
  some tests) and have other tests that don't use any fixture. Classify
  per-function, not per-file.

## Resource gating

E2e and qualitative tests need gating so they skip cleanly when the required
infrastructure is absent. The preferred mechanism is **predicate functions**
— reusable decorators that encapsulate availability checks. Test authors
apply the predicate that matches their test's actual requirements.

### The predicate factory pattern (general)

Projects should provide a shared module of predicate functions that return
`pytest.mark.skipif(...)` decorators. This gives test authors precision
(exact thresholds, specific env vars) without ad-hoc `skipif` or blunt
resource markers scattered across files.

### Determining `min_vram_gb` and `min_gb` values

When migrating legacy `requires_gpu` markers to predicates, do not guess or use
blanket thresholds. Determine the correct `min_vram_gb` by tracing the model each
test loads and computing VRAM requirements from parameter counts. Legacy
`requires_heavy_ram` markers should simply be removed (see "What to audit" below).

#### Trace the model identifier

For each file needing GPU/RAM gating, determine which model(s) it loads. Check in order:

1. **Module-level constants** — e.g. `BASE_MODEL = "ibm-granite/..."` or
   `MODEL_ID = model_ids.QWEN3_0_6B`.
2. **Fixture definitions** — trace `@pytest.fixture` functions for:
   - `LocalHFBackend(model_id=...)` — extract the `model_id` argument
   - `LocalVLLMBackend(model_id=...)` — extract the `model_id` argument
   - `start_session("hf", model_id=...)` — extract `model_id`
3. **ModelIdentifier resolution** — if the model_id is a constant like
   `model_ids.QWEN3_0_6B`, read `mellea/backends/model_ids.py` and extract
   the `hf_model_name` field.
4. **Conftest fixtures** — check `conftest.py` files up the directory tree for
   fixture definitions that provide model/backend instances.
5. **Per-function overrides** — some files have different models per test function.
   Track per-function when this occurs.

#### Look up parameter count

Use these strategies in priority order. Stop at the first that succeeds.

**Strategy A: HuggingFace Hub API** (preferred, requires network)

```python
from huggingface_hub.utils._safetensors import get_safetensors_metadata
meta = get_safetensors_metadata("ibm-granite/granite-3.3-8b-instruct")
total_params = sum(meta.parameter_count.values())
```

Run via `uv run python -c "..."` — only needs `huggingface_hub` (in the `[hf]` extra).

**Strategy B: Ollama model info** (for Ollama-tagged models)

```bash
ollama show <model_name> --modelfile 2>/dev/null | grep -i 'parameter'
```

**Strategy C: Model name parsing** (offline fallback)

| Pattern | Extract | Example match |
|---------|---------|---------------|
| `(\d+\.?\d*)b[-_.]` or `-(\d+\.?\d*)b` | N billion | `granite-3.3-8b` → 8B |
| `(\d+\.?\d*)B` (capital B in HF names) | N billion | `Qwen3-0.6B` → 0.6B |
| `-(\d+)m[-_.]` or `(\d+)m-` | N million ÷ 1000 | `granite-4.0-h-350m` → 0.35B |
| `micro` without explicit size | 0.35B–3B | Check ModelIdentifier catalog |

When the name is ambiguous (e.g. `granite4:micro-h`), resolve via the
`ModelIdentifier` constant in `model_ids.py` — the HF name usually contains
the explicit size.

**Strategy D: Conservative default** (last resort)
- Assume **8B parameters** (16 GB at fp16)
- Flag as **"model unidentified — manual review needed"**

#### Backend determines GPU gating need

| Backend | GPU loaded locally? | Predicate needed |
|---------|--------------------|--------------------|
| `LocalHFBackend` | Yes | `require_gpu(min_vram_gb=N)` |
| `LocalVLLMBackend` | Yes | `require_gpu(min_vram_gb=N)` |
| `OllamaModelBackend` | Managed by Ollama | None — `ollama` backend marker + conftest auto-skip handles availability |
| `OpenAIBackend` (real API) | No | No GPU gate |
| `OpenAIBackend` → Ollama `/v1` | Managed by Ollama | None — `ollama` backend marker handles it |
| `WatsonxAIBackend` / `LiteLLMBackend` / Cloud | No | No GPU gate |

**Key rule:** Ollama manages its own GPU memory. Tests using Ollama backends
should use the `ollama` backend marker only, NOT `require_gpu()`.

#### Compute VRAM estimate

**VRAM formula:**
```
vram_gb = params_B × bytes_per_param × 1.2
```

Where `bytes_per_param` depends on precision: fp32=4.0, fp16/bf16=2.0 (default),
int8=1.0, int4=0.5. The 1.2 multiplier covers KV cache, activations, and framework
buffers. Round `min_vram_gb` **up** to the next even integer.

Models load into GPU VRAM, not system RAM — do **not** add `require_ram()` to GPU
tests. The `require_ram()` predicate exists for tests that are genuinely RAM-bound
(large dataset processing, etc.), not as a companion to `require_gpu()`.

### What to audit

Check the project's predicate module (see Project References) for available
predicates, then apply the following checks to every e2e/qualitative file:

1. **Legacy resource markers → migrate to predicates.** If a test uses
   `@pytest.mark.requires_gpu` or `@pytest.mark.requires_api_key`,
   replace with the equivalent predicate from the project's predicate module.
   If a test uses `@pytest.mark.requires_heavy_ram` or
   `@pytest.mark.requires_gpu_isolation`, **remove** them — these are
   deprecated markers with no direct replacement (`requires_heavy_ram` was a
   blanket 48 GB threshold that conflated VRAM with RAM; GPU isolation is now
   automatic). Resource markers are deprecated in favour of predicates. This
   is a **fix** (same priority as `llm` → `e2e`), not just a recommendation —
   apply it in Step 4 like any other marker fix. The replacement requires
   adding an import for the predicate and swapping the marker in the
   `pytestmark` list or decorator.
2. **Ad-hoc `skipif` → migrate to predicate.** If a predicate exists for
   the same check (e.g., `require_gpu()` exists but the test has a raw
   `skipif(not torch.cuda.is_available())`), replace with the predicate.
3. **Missing gating.** A test that uses a GPU backend but has no GPU
   predicate and no `skipif` — add the appropriate predicate.
4. **Imprecise gating.** A predicate that's too broad for the actual model
   being loaded — tighten the `min_vram_gb` threshold.
5. **Redundant CICD `skipif`.** `skipif(CICD == 1)` is usually redundant
   when conftest auto-skip or predicates already handle the condition.
   Flag as removable.

### What NOT to flag

Not every `skipif` needs migrating. Leave these alone:

- **Python version gates** (`skipif(sys.version_info < (3, 11))`) — one-off,
  or use `require_python()` predicate if available.
- **`importorskip` for optional deps** — idiomatic pytest, or use
  `require_package()` predicate if available and a decorator style is preferred.
- **Truly one-off conditions** with no predicate equivalent and no pattern
  of recurrence across files.

For any inline `skipif` that IS NOT covered above, check whether a matching
predicate exists. If it does → recommend migration. If it doesn't and the
same condition appears in multiple files → flag as an infrastructure note
("consider adding a predicate for this condition").

Resource gating is orthogonal to tier classification — a test gated by
`require_gpu()` is still e2e/qualitative based on what it exercises.

### Project predicates (`test/predicates.py`)

Read `test/predicates.py` for the available predicates. Expected patterns:

| Predicate | Use when test needs |
|---|---|
| `require_gpu()` | Any GPU (CUDA or MPS) |
| `require_gpu(min_vram_gb=N)` | GPU with at least N GB VRAM |
| `require_ram(min_gb=N)` | N GB+ system RAM (for genuinely RAM-bound tests, not GPU model loading) |
| `require_api_key("OPENAI_API_KEY")` | Specific API credentials |
| `require_api_key("WATSONX_API_KEY", "WATSONX_URL", "WATSONX_PROJECT_ID")` | Multiple credentials |
| `require_package("cpex.framework")` | Optional dependency |
| `require_python((3, 11))` | Minimum Python version |

Typical combinations for backends:

- `huggingface` → `require_gpu(min_vram_gb=N)` (compute N from model params)
- `vllm` → `require_gpu(min_vram_gb=N)` (compute N from model params)
- `watsonx` → `require_api_key("WATSONX_API_KEY", "WATSONX_URL", "WATSONX_PROJECT_ID")`
- `openai` → `require_api_key("OPENAI_API_KEY")` only for real OpenAI (not Ollama-compat)

## Example files (`docs/examples/`)

Examples use a comment-based marker format (not `pytestmark`):

```python
# pytest: e2e, ollama, qualitative
```

Same classification rules apply. Parser: `docs/examples/conftest.py`
(`_extract_markers_from_file`).

**Legacy markers in examples:** The same deprecation rules apply to
`# pytest:` comments. Remove `requires_heavy_ram`, `requires_gpu_isolation`,
and `llm` when found. Replace `requires_gpu` with the appropriate predicate
marker if the comment format supports it, or just remove it and rely on the
backend marker (e.g., `huggingface` already triggers GPU checks in the
examples conftest).

---

# Audit Procedure

## Step 0 — Triage (for scopes larger than ~5 files)

When auditing a directory or the full repo, do NOT deep-read every file.
Use behavioural signal detection (Part 1) to bucket files first, then
deep-read only files that need inspection.

### Phase 0: Fixture discovery

Read `conftest.py` files in the target scope to catalog fixture names.
Classify each fixture as **live** (returns a real backend/connection) or
**mock** (returns a MagicMock, patch, or fake). Record these lists — they
become additional grep patterns for the next phase.

### Phase 1: Signal grep

Run grep across all target files for:

1. **Live-backend signals** — backend imports, constructors, `start_session(`,
   network literals (`localhost`, `127.0.0.1`, port numbers), HTTP client
   usage, subprocess calls, `_API_KEY`/`_TOKEN`/`_SECRET` in env var checks,
   GPU/model loading (`torch`, `.from_pretrained(`), URL literals.
1a. **Training signals** — `from_pretrained(`, `train_model(`, `Trainer(`, `epochs=`,
    `.generate(` (non-mock), `load_dataset(`. Cross-reference against
    `require_gpu(min_vram_gb=N)` — check N uses the 2× training memory rule.
2. **SDK-boundary signals** — real third-party SDK objects used as collaborators:
   `InMemoryMetricReader`, `InMemorySpanExporter`, `MeterProvider(metric_readers=`,
   `TracerProvider(`, `LoggingHandler`, `provider.force_flush()`,
   `reader.get_metrics_data()`. See the SDK-boundary signal table in the
   Behavioural Signal Detection section.
3. **Mock signals** — `MagicMock`, `Mock(`, `AsyncMock`, `@patch(`,
   `monkeypatch`, `mocker`, HTTP mock libraries.
4. **Existing markers** — `pytestmark`, `@pytest.mark.`, `# pytest:`.
5. **Live/mock fixture names** from Phase 0.

### Phase 2: Bucket and prioritise

Cross-reference the signal hits into four priority buckets:

| Priority | Condition | Action |
|---|---|---|
| **P1 — Missing markers** | Live-backend or SDK-boundary signals present, NO existing markers | Deep-read and classify. These are the most likely gaps. |
| **P2 — Mixed signals** | Both live/SDK signals AND mock signals present | Deep-read to determine if integration, partial mock, or mixed file. |
| **P3 — Validate existing** | Live-backend or SDK-boundary signals present, markers already exist | Spot-check that markers match. Replace deprecated `llm`. |
| **P4 — Skip** | No live-backend or SDK-boundary signals (mock-only or no signals) | Likely unit. Report as clean without deep-reading. Spot-check a sample if the count is large. |

### Phase 3: Deep-read

Process P1 → P2 → P3 files using Steps 1–5 below. For P4 files, list them
in the summary as "N files — no live-backend signals, assumed unit."

## Step 1 — Read and identify

Read the file fully. Identify:
- Module-level `pytestmark` (test files) or `# pytest:` comment (examples)
- Per-function `@pytest.mark.*` decorators
- Fixtures and their backend dependencies (trace the fixture chain — see above)
- Any use of the deprecated `llm` marker

**For example files (`docs/examples/`):** Examples are standalone scripts, not
fixture-based tests. Classification comes from reading the code directly —
look for backend imports, `start_session()` calls, and constructor usage.
The `# pytest:` comment is the only marker mechanism (no `pytestmark`).

## Step 2 — Classify each test function

For each `def test_*` or `async def test_*`, apply the general classification
from Part 1 using the project-specific heuristics from Part 2:

1. **Real backend, SDK collaborator, or fully mocked?**
   - Real LLM backend (Ollama, HF, cloud API) → **e2e**
   - Real third-party SDK object asserted against (OTel reader, logging handler) → **integration**
   - All external boundaries mocked/patched → **unit** (single component) or **integration** (multiple real components wired)
2. **Which backend(s)?** → backend markers (e2e only)
3. **Deterministic or content-dependent assertions?** → e2e vs qualitative
4. **What resources?** → resource markers
5. **Training memory?** → if training signals present (`train_model(`, `Trainer(`,
   `epochs=`), verify `require_gpu(min_vram_gb=N)` uses 2× the model inference
   memory + headroom (see Training memory signals table).

If uncertain about a classification (especially qualitative vs e2e), note it
and ask the user to confirm.

## Step 3 — Compare and report

### Output tiers

Scale the report detail to the scope of the audit:

**Tier 1 — Summary table (always).**  Print first so the user sees the big
picture before any detail:

```
| Category | Files | Functions |
|----------|------:|----------:|
| Correct (no changes) | 42 | — |
| Deprecated `llm` → `e2e` (simple) | 27 | — |
| Missing tier marker | 8 | 12 |
| Wrong marker | 3 | 5 |
| Over-marked | 2 | 4 |
| Missing resource gating | 4 | 6 |
| Legacy resource marker → predicate | 5 | 9 |
| Infrastructure notes | 3 | — |
```

**Tier 2 — Issues-only detail.**  For each file with at least one issue,
print the file header and **only the functions that need changes**.  Omit
functions that are already correct — they are noise at scale:

```
## test/backends/test_ollama.py

Module markers — Current: [llm, ollama] → Proposed: [e2e, ollama]
  ↳ replace deprecated `llm` with `e2e`

  test_structured_output — WRONG: asserts JSON schema, remove `qualitative`
```

Functions without issues (`test_simple_instruct ✓`, `test_chat ✓`) are
**not listed**.  Files where everything is correct appear only in the Tier 1
count.

**Tier 3 — Batch groups (for mechanical fixes).**  When many files share the
same fix (e.g. `llm` → `e2e` in `pytestmark`), collapse them into a single
block instead of repeating the per-file template:

```
### Deprecated `llm` → `e2e` (27 files, module-level pytestmark)

test/backends/test_ollama.py
test/backends/test_openai.py
test/backends/test_watsonx.py
... (24 more)
```

The agent should list all files (not truncate) so the user can review before
applying, but one line per file is sufficient when the fix is identical.

## Step 4 — Apply fixes

The apply behaviour depends on the flags passed:

| Flag | Behaviour |
|------|-----------|
| *(none)* | Output the full report (Steps 1–3 + Output Summary), then **ask the user** "Apply these N changes?" before writing any files. |
| `--apply` | Output the full report, then apply all fixes **without asking**. |
| `--dry-run` | Output the full report. Do NOT write any files or offer to apply. |

**When applying fixes:**

Surgical edits only — change specific marker lines, do not reformat surrounding code.
When replacing `llm` with `e2e` in `pytestmark` lists, keep the same list structure.
When replacing legacy resource markers with predicates, add the necessary import
(`from test.predicates import ...`) at the top of the file and swap the marker
in the `pytestmark` list or decorator.

## Step 5 — Backend registry audit

Check that every backend used in test files has a registered marker.
The project's backend registry is `BACKEND_MARKERS` in `test/conftest.py`
(single source of truth). Markers must also appear in `pyproject.toml`
`[tool.pytest.ini_options].markers` and in `test/MARKERS_GUIDE.md`.

For each backend constructor or `start_session(backend_name=...)` call
found during classification, verify:

1. A marker exists in `BACKEND_MARKERS` for that backend.
2. The marker appears in `pyproject.toml`.
3. The marker appears in the MARKERS_GUIDE.md backend table.

If a backend is used in tests but has no registered marker, flag it as
a **missing backend marker** issue and add it to the registry, pyproject.toml,
and MARKERS_GUIDE.md (same apply/confirm rules as other fixes in Step 4).

## Step 6 — Flag infrastructure notes

Report issues outside marker-edit scope as **notes**. Do NOT fix these:
- Missing conftest skip logic for a backend
- Tests with no assertions
- Files mixing unit and integration tests that could be split
- Files mixing unit and e2e tests that could be split

## Output Summary

The output is the Tier 1 summary table (always printed first) followed by
Tier 2 issues-only detail and Tier 3 batch groups as described in Step 3.
End the report with:

The summary table should include a row for `Missing backend marker` when
backends are used in tests but not registered in `BACKEND_MARKERS`.

```
---
Files audited: N | Correct: N | With issues: N
Changes: N applied / N pending confirmation / N dry-run
Infrastructure notes: N (see notes section)
```

## Infrastructure (already in place — do not re-add)

The following infrastructure was set up in #727 and should NOT be recreated
by this skill.  If an audit finds these missing, something has regressed —
flag as a blocker, don't silently re-add:

- **Auto-unit hook:** `test/conftest.py` `pytest_collection_modifyitems` adds
  `pytest.mark.unit` to any test without `integration`, `e2e`, or `qualitative`.
- **Backend marker registry:** `BACKEND_MARKERS` dict in `test/conftest.py` is
  the single source of truth for backend markers. `pytest_configure` iterates
  over it. New backends are added by inserting one entry into the dict.
  `pyproject.toml` and `test/MARKERS_GUIDE.md` must stay in sync manually.
- **Resource predicates:** `test/predicates.py` provides `require_gpu`,
  `require_ram`, `require_api_key`, `require_package`,
  `require_python`.
