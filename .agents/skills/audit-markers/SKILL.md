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
- **Example marker format:** `docs/examples/conftest.py` (`_extract_markers_from_file`)
- **Epic context:** GitHub issues #726 (epic), #727 (granularity), #728 (backend/resource)

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

**Key distinction from unit:** Unit is entirely self-contained with no services.
Integration wires up components and may need services (even fixture-managed ones).

**Key distinction from e2e:** Integration controls its dependencies (mocks, stubs,
fixture-managed services). E2E uses real backends that exist independently.

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
- `llm` is deprecated (synonym for `e2e`) — **flag and recommend replacing**
- Backend/resource markers only go on `e2e`/`qualitative` tests
- `qualitative` is always per-function; module carries `e2e` + backend markers
- If a file mixes unit and non-unit tests, apply markers per-function, not module-level

## Backend detection heuristics

When classifying a test file, check ALL of the following to determine which
backend(s) it uses:

- **Imports:** `from mellea.backends.ollama import ...` → `ollama`
- **Session creation:** `start_session("ollama", ...)` → `ollama`; bare `start_session()` with no backend arg → `ollama` (default backend)
- **Backend constructors:** `OllamaModelBackend(...)` → `ollama`; `OpenAIBackend(...)` → `openai`
- **Environment variables checked:** `OPENAI_API_KEY` → `openai`; `WATSONX_API_KEY` → `watsonx`
- **Dual backends:** `OpenAIBackend` pointed at Ollama's `/v1` endpoint → both `openai` AND `ollama` (but NOT `requires_api_key`)

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

## Resource marker inference

These are not automatic — verify by reading the code:

- `huggingface` usually → `requires_gpu` + `requires_heavy_ram` + `requires_gpu_isolation`
- `vllm` usually → `requires_gpu` + `requires_heavy_ram` + `requires_gpu_isolation`
- `watsonx` usually → `requires_api_key`
- `openai` → `requires_api_key` ONLY when using the real OpenAI API (not Ollama-compatible)

## Example files (`docs/examples/`)

Examples use a comment-based marker format (not `pytestmark`):

```python
# pytest: e2e, ollama, qualitative
```

Same classification rules apply. Parser: `docs/examples/conftest.py`
(`_extract_markers_from_file`).

---

# Audit Procedure

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

Per-file report format:

```
## test/backends/test_ollama.py

Module markers — Current: [llm, ollama] → Proposed: [e2e, ollama]
  Note: replace deprecated `llm` with `e2e`

  test_simple_instruct   — qualitative ✓
  test_structured_output — Current: qualitative → WRONG: asserts JSON schema, remove qualitative
  test_chat              — qualitative ✓
```

## Step 4 — Apply fixes (unless `--dry-run`)

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

```
## Audit Summary

Files audited: N
Files correct: N
Files with issues: N

Issues by type:
  Missing markers:     N
  Wrong markers:       N
  Over-marked:         N
  Deprecated (llm):    N

Changes: N applied / N dry-run
Infrastructure notes: N (see notes section)
```

## Infrastructure Note (not part of this skill's scope)

For `pytest -m unit` to work, the project needs a conftest hook:

```python
# In test/conftest.py pytest_collection_modifyitems:
_NON_UNIT = ("integration", "e2e", "qualitative", "llm")
for item in items:
    if not any(item.get_closest_marker(m) for m in _NON_UNIT):
        item.add_marker(pytest.mark.unit)
```

The `e2e` and `integration` markers also need registering in `pytest_configure`
and `pyproject.toml`. These are one-time infrastructure changes tracked in
issue #727, not performed by this skill.
