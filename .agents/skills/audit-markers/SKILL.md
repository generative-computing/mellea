---
name: audit-markers
description: >
  Audit and fix pytest markers on test files and examples. Classifies tests as
  unit/integration/e2e/qualitative using general heuristics and project-specific
  marker rules. Use when reviewing markers, auditing test files, or checking
  before commit. References test/MARKERS_GUIDE.md for project conventions.
argument-hint: "[file-or-directory] [--dry-run]"
compatibility: "Claude Code, IBM Bob"
metadata:
  version: "2026-03-25"
  capabilities: [read_file, write_file, bash, grep, glob]
---

# Audit & Fix Pytest Markers

Classify tests, validate markers, and fix issues. Works in two layers:
general test classification (applicable to any project) plus project-specific
marker rules for **mellea**.

## Inputs

- `$ARGUMENTS` — file path, directory, or glob. If empty, audit `test/` and `docs/examples/`.
- `--dry-run` — report only, do not edit files.

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
- If it uses test doubles, they replace external boundaries (network, DB, services)
- No fixture that starts/connects to a real or fixture-managed service
- Runs in milliseconds to low seconds
- Would pass on any machine with just the language runtime and project deps

**Examples of unit assertions:**
```python
assert str(cb) == "hello"
assert len(items) == 3
assert raises(ValueError)
mock_backend.generate.assert_called_once()
```

### Integration

Tests **multiple components working together**, potentially needing additional
services or fixture-managed dependencies. Backends may be mocked, stubbed, or
stood up by test fixtures.

**Recognise by:**
- Creates real instances of multiple project components and wires them together
- External service boundaries may be mocked, stubbed, or managed by fixtures
- Tests that the components interact correctly — data flows, callbacks fire, errors propagate
- May need additional services, but the test controls or provides its dependencies
- Slower than unit (fixture setup, service lifecycle) and may consume more memory

**Key distinction from unit:** Count the real (non-mock) project components
being exercised. Unit isolates **one** class or function — all collaborators
are faked. Integration wires up **multiple** real components and mocks only
at the external perimeter (network, backend, database).

**Key distinction from e2e:** Integration controls its dependencies (mocks, stubs,
fixture-managed services). E2E uses real backends that exist independently.

**Borderline: unit vs integration (the "scope of mocks" rule)**

When a test uses mocks, look at *what* is mocked to decide:

- **Mock replaces external I/O only, multiple real internal components wired
  together** → **integration**. Example: a test that registers a real `Plugin`,
  calls real `invoke_hook()` and `register()`, but passes `MagicMock()` for
  the backend. The plugin-manager wiring executes for real; only the LLM call
  is faked.
- **Mock replaces internal collaborators too, only one real component under
  test** → **unit**. Example: a test that instantiates one `Plugin` but
  passes `MagicMock()` for the session, the backend, and the hook dispatcher.
  Only the plugin's own logic executes.

When in doubt, ask: "if I broke the *wiring* between two components, would
this test catch it?" If yes → integration. If no → unit.

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

A single file may contain both live and mock signals. Cross-reference to
determine the correct bucket:

| Live signals? | Mock signals? | Classification |
|---|---|---|
| Yes | No | Almost certainly e2e — deep-read to confirm |
| Yes | Yes | Needs inspection — partial mock = integration, or mixed file |
| No | Yes | Likely unit — skip deep read |
| No | No | Likely unit — skip deep read |

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

### What to audit

Check the project's predicate module (see Project References) for available
predicates, then apply the following checks to every e2e/qualitative file:

1. **Legacy resource markers → migrate to predicates.** If a test uses
   `@pytest.mark.requires_gpu`, `@pytest.mark.requires_heavy_ram`,
   `@pytest.mark.requires_api_key`, or `@pytest.mark.requires_gpu_isolation`,
   recommend replacing with the equivalent predicate from the project's
   predicate module. Resource markers are deprecated in favour of predicates.
2. **Ad-hoc `skipif` → migrate to predicate.** If a predicate exists for
   the same check (e.g., `require_gpu()` exists but the test has a raw
   `skipif(not torch.cuda.is_available())`), recommend the predicate.
3. **Missing gating.** A test that uses a GPU backend but has no GPU
   predicate and no `skipif` — recommend adding the appropriate predicate.
4. **Imprecise gating.** A predicate that's too broad (e.g., `require_ram(48)`
   on a test that only needs 16 GB) — suggest tightening the threshold.
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
| `require_ram(min_gb=N)` | N GB+ system RAM |
| `require_gpu_isolation()` | Subprocess isolation for CUDA memory |
| `require_api_key("OPENAI_API_KEY")` | Specific API credentials |
| `require_api_key("WATSONX_API_KEY", "WATSONX_URL", "WATSONX_PROJECT_ID")` | Multiple credentials |
| `require_package("cpex.framework")` | Optional dependency |
| `require_ollama()` | Running Ollama server |
| `require_python((3, 11))` | Minimum Python version |

Typical combinations for backends:

- `huggingface` → `require_gpu()` + `require_ram(48)` (adjust RAM per model)
- `vllm` → `require_gpu(min_vram_gb=24)` + `require_ram(48)`
- `watsonx` → `require_api_key("WATSONX_API_KEY", "WATSONX_URL", "WATSONX_PROJECT_ID")`
- `openai` → `require_api_key("OPENAI_API_KEY")` only for real OpenAI (not Ollama-compat)

## Example files (`docs/examples/`)

Examples use a comment-based marker format (not `pytestmark`):

```python
# pytest: e2e, ollama, qualitative
```

Same classification rules apply. Parser: `docs/examples/conftest.py`
(`_extract_markers_from_file`).

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
2. **Mock signals** — `MagicMock`, `Mock(`, `AsyncMock`, `@patch(`,
   `monkeypatch`, `mocker`, HTTP mock libraries.
3. **Existing markers** — `pytestmark`, `@pytest.mark.`, `# pytest:`.
4. **Live/mock fixture names** from Phase 0.

### Phase 2: Bucket and prioritise

Cross-reference the signal hits into four priority buckets:

| Priority | Condition | Action |
|---|---|---|
| **P1 — Missing markers** | Live signals present, NO existing markers | Deep-read and classify. These are the most likely gaps. |
| **P2 — Mixed signals** | Both live AND mock signals present | Deep-read to determine if integration, partial mock, or mixed file. |
| **P3 — Validate existing** | Live signals present, markers already exist | Spot-check that markers match the actual backend. Replace deprecated `llm`. |
| **P4 — Skip** | No live signals (mock-only or no signals at all) | Likely unit. Report as clean without deep-reading. Spot-check a sample if the count is large. |

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

1. **Real backend or mocked?** → determines unit/integration vs e2e
2. **Which backend(s)?** → backend markers (e2e only)
3. **Deterministic or content-dependent assertions?** → e2e vs qualitative
4. **What resources?** → resource markers

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

## Step 4 — Apply fixes (unless `--dry-run`)

**If `--dry-run` is active, do NOT execute any file writes or bash commands that
modify code. Output the report only.**

Surgical edits only — change specific marker lines, do not reformat surrounding code.

When replacing `llm` with `e2e` in `pytestmark` lists, keep the same list structure.

## Step 5 — Flag infrastructure notes

Report issues outside marker-edit scope as **notes**. Do NOT fix these:
- Missing conftest skip logic for a backend
- Unregistered markers in pyproject.toml
- MARKERS_GUIDE.md gaps
- Tests with no assertions
- Files mixing unit and e2e tests that could be split

## Output Summary

The output is the Tier 1 summary table (always printed first) followed by
Tier 2 issues-only detail and Tier 3 batch groups as described in Step 3.
End the report with:

```
---
Files audited: N | Correct: N | With issues: N
Changes: N applied / N dry-run
Infrastructure notes: N (see notes section)
```

## Infrastructure (already in place — do not re-add)

The following infrastructure was set up in #727 and should NOT be recreated
by this skill.  If an audit finds these missing, something has regressed —
flag as a blocker, don't silently re-add:

- **Auto-unit hook:** `test/conftest.py` `pytest_collection_modifyitems` adds
  `pytest.mark.unit` to any test without `integration`, `e2e`, or `qualitative`.
- **Marker registration:** all tier, backend, and resource markers registered in
  `pytest_configure` and `pyproject.toml`.
- **Resource predicates:** `test/predicates.py` provides `require_gpu`,
  `require_ram`, `require_gpu_isolation`, `require_api_key`, `require_package`,
  `require_ollama`, `require_python`.
