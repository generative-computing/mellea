# Mellea Test Suite

This file covers everything you need to contribute tests: strategy, classification,
marker reference, authoring guide, CI pipeline, and operational notes.

## Quick start

**First-time setup.** Follow the
[Contributing Guide](../docs/docs/community/contributing-guide.md#development-setup)
to install `uv`, sync deps, and install pre-commit hooks. For anything beyond
unit tests you also need Ollama running locally:

```bash
ollama serve &
ollama pull granite4:micro
ollama pull granite4:micro-h
```

**Running tests during development.**

```bash
uv run pytest -m "not qualitative"            # ~2 min fast loop
uv run pytest test/path/to/test_thing.py       # focus on one file
uv run pytest -rs                              # show why anything skipped
```

**Adding a new test.**

1. **Classify** it (see [Test tiers](#test-tiers)). If it doesn't call a real
   backend or external SDK, it's `unit` and needs no marker.
2. **Place** the file at `test/<mirror of source path>/test_<module>.py`.
3. **Add the granularity marker** — `integration`, `e2e`, or `qualitative` are
   explicit; `unit` is auto-applied by conftest, never write it.
4. **Add backend marker(s)** — only for `e2e`/`qualitative` tests
   (`ollama`, `huggingface`, etc.). See [Backend markers](#backend-markers).
5. **Add resource predicates** — only for `e2e`/`qualitative`, use
   `test/predicates.py`. See [Resource predicates](#resource-predicates).
6. **Verify collection** — `uv run pytest --collect-only -m "your_marker"`.
7. **Run it** — `uv run pytest path::test_name -v`.

**Before opening a PR.**

```bash
uv run pre-commit run --all-files              # what CI runs first
CICD=1 uv run pytest test                      # what CI runs second
```

## Philosophy

Mellea tests assert **observable contracts**, not implementation details.

- Test the public API surface, not private helpers.
- Test cross-backend behaviour where it needs to be consistent.
- A test that passes while the system is broken has negative value; prefer
  fewer, more honest tests over coverage padding.
- When a test fails, fix the **code**. Adjusting an assertion to silence a
  failure is almost always wrong; fixing the test is acceptable only if the
  test was never correctly written.
- All tiers run locally by default. Qualitative tests (assertions on LLM output
  *content*) are the one tier that CI skips, so non-deterministic checks never
  block CI.

## Test tiers

Every test belongs to exactly one granularity tier. Apply the decision rules
below in order:

| Question | Answer → tier |
|----------|---------------|
| Does it call a real LLM backend or external API? | Yes → **e2e** (or **qualitative**, see below) |
| Does it assert against a real third-party SDK object (OTel reader, metrics collector)? | Yes → **integration** |
| Does it wire multiple real project components together without external I/O? | Yes → **integration** |
| Does everything happen in-process with no real external collaborators? | Yes → **unit** (auto-applied) |

### Unit

**Entirely self-contained** — no services, no I/O, no network. Pure logic:
formatters, parsers, schema validation, config loading, pure helper functions.
Runs in milliseconds on any machine.

The `unit` marker is **auto-applied by conftest** to every test that has no
other granularity marker. Never write `@pytest.mark.unit` yourself.

```python
# No markers needed — auto-applied as unit
def test_cblock_repr():
    assert str(CBlock(value="hi")) == "hi"
```

### Integration

**Verifies that your code correctly communicates across a real boundary.**
The boundary may be a third-party SDK/library whose API contract you are
asserting against, multiple internal components wired together, or a
fixture-managed local service. What distinguishes integration from unit is
that at least one real external component — not a mock or stub — is on the
other side of the boundary being tested.

Add `@pytest.mark.integration` explicitly; no backend marker is needed.

**Positive indicators:**

- Uses a real third-party SDK object to *capture and assert* on output —
  e.g. `InMemoryMetricReader`, `InMemorySpanExporter`, `LoggingHandler` —
  rather than patching the SDK away
- Asserts on the format or content of data as received by an external
  component (semantic conventions, attribute names, accumulated values)
- Wires multiple real project components together and mocks only at the
  outermost boundary
- Breaking the interface between your code and the external component
  would cause the test to fail

**Negative indicators (likely unit instead):**

- All external boundaries replaced with `MagicMock`, `patch`, or `AsyncMock`
- Third-party library imported only as a type or helper, not as a real
  collaborator being asserted against

**Tie-breaker:** if you changed the contract between your code and the external
component, would this test catch it? If yes → integration. If no → unit.

```python
@pytest.mark.integration
def test_token_metrics_format(clean_metrics_env):
    # Real InMemoryMetricReader — asserting against the OTel SDK contract
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    record_token_usage_metrics(input_tokens=10, output_tokens=5, ...)
    metrics_data = reader.get_metrics_data()
    assert metrics_data.resource_metrics[0]...name == "mellea.llm.tokens.input"

@pytest.mark.integration
def test_session_chains_components(mock_backend):
    # Multiple real project components wired together; only LLM call mocked
    session = start_session(backend=mock_backend)
    result = session.instruct("hello")
    assert mock_backend.generate.called
```

### E2E

**Tests against real backends** — cloud APIs, local servers (Ollama), or
GPU-loaded models (HuggingFace, vLLM). No mocks on the critical path.

Add `@pytest.mark.e2e` explicitly, always combined with backend marker(s).
Assertions must be **deterministic** — structural, type-based, or functional.
Assertions on generated text content belong in qualitative tests, not e2e.

```python
pytestmark = [pytest.mark.e2e, pytest.mark.ollama]

def test_structured_output_returns_valid_json(session):
    result = session.format(Person, "Make up a person")
    assert isinstance(json.loads(result.value), dict)
```

### Qualitative

**A sub-tier of e2e**: same infrastructure requirements, but assertions check
**non-deterministic output content** that may vary across model versions or runs.

Add `@pytest.mark.qualitative` per-function (not at module level). The module
still needs `e2e` and the backend marker. Qualitative tests are included in the
default local run but skipped in CI (`CICD=1`).

```python
pytestmark = [pytest.mark.e2e, pytest.mark.ollama]

@pytest.mark.qualitative
def test_greeting_contains_salutation(session):
    result = session.instruct("Write a greeting")
    assert "hello" in result.value.lower()    # content check — qualitative
```

**Decision rule:** if swapping the model version could break the assertion
despite the system working correctly, it is `qualitative`. If the assertion
checks structure, types, or functional correctness, it is `e2e`.

### Deprecated: the `llm` marker

`llm` is a legacy alias for `e2e`. It remains registered for backwards
compatibility but must not be used in new tests. The conftest auto-apply hook
treats `llm` the same as `e2e`.

## Backend markers

Backend markers identify which backend a test needs. They enable selective test
runs (`pytest -m ollama`) and drive auto-skip logic. **Only apply to `e2e` and
`qualitative` tests.**

| Marker | Backend | Resources |
|--------|---------|-----------|
| `ollama` | Ollama (`OLLAMA_HOST`, default `:11434`) | Local, light (~2–4 GB RAM) |
| `openai` | OpenAI API or any OpenAI-compatible endpoint | API calls (may use Ollama `/v1`) |
| `watsonx` | IBM Watsonx API | API calls, requires credentials |
| `huggingface` | HuggingFace transformers | Local, GPU required (VRAM varies) |
| `vllm` | vLLM via an OpenAI-compatible server | Local, GPU required (combine with `openai`) |
| `litellm` | LiteLLM (wraps other backends) | Depends on underlying backend |
| `bedrock` | AWS Bedrock | API calls, requires credentials |

### OpenAI-via-Ollama pattern

Some tests use the OpenAI client pointed at Ollama's `/v1` endpoint. Mark these
with **both** `openai` and `ollama`, but do **not** add `require_api_key`:

```python
pytestmark = [pytest.mark.e2e, pytest.mark.openai, pytest.mark.ollama]
```

### vLLM

vLLM is tested through an OpenAI-compatible server, so vLLM tests carry the
`openai` and `e2e` markers **plus** the `vllm` marker. The `vllm` marker is the
selection axis for the vLLM-specific subset (`pytest -m vllm` picks exactly those
modules, a strict subset of `-m openai`) and drives GPU skip-gating for
`# pytest: vllm` example files in `docs/examples/conftest.py`. Backend grouping
under `--group-by-backend` keys on the `openai` marker (the `openai_vllm` group),
not on `vllm`.

Resource gating is handled by `require_gpu(min_vram_gb=N)` plus the
`vllm_process` fixture in `test/backends/test_openai_vllm.py`, which connects to
an externally started server (`VLLM_TEST_BASE_URL`, set by the nightly script) or
spawns one on a CUDA host, and skips otherwise.

```python
pytestmark = [pytest.mark.e2e, pytest.mark.openai, pytest.mark.vllm,
              require_gpu(min_vram_gb=8)]
```

## Resource predicates

Fine-grained resource gating uses predicate decorators from `test/predicates.py`.
They compose with `pytestmark` and produce self-documenting skip reasons:

```python
from test.predicates import require_gpu, require_api_key
```

| Predicate | Use when test needs |
|-----------|---------------------|
| `require_gpu()` | Any GPU (CUDA or MPS) |
| `require_gpu(min_vram_gb=N)` | GPU with at least N GB VRAM |
| `require_ram(min_gb=N)` | N GB+ system RAM (genuinely RAM-bound tests only) |
| `require_api_key("ENV_VAR")` | Specific API credentials |
| `require_package("pkg")` | Optional dependency |
| `require_python((3, 11))` | Minimum Python version |

**Typical combinations:**

- `huggingface` → `require_gpu(min_vram_gb=N)` (compute N from model parameters)
- vLLM (`openai` + `vllm` markers) → `require_gpu(min_vram_gb=N)` (compute N from model parameters)
- `watsonx` → `require_api_key("WATSONX_API_KEY", "WATSONX_URL", "WATSONX_PROJECT_ID")`
- `openai` → `require_api_key("OPENAI_API_KEY")` only for real OpenAI, not Ollama-compat

**Other gating markers** (not resource predicates, but still control selection):

| Marker | Gate | Auto-skip when |
|--------|------|----------------|
| `slow` | Tests taking >1 minute | Excluded by default via `pyproject.toml` `addopts` |
| `qualitative` | Non-deterministic output | Skipped when `CICD=1` |

**Removed markers:** `requires_gpu`, `requires_heavy_ram`, and
`requires_gpu_isolation` have been removed. Use `require_gpu(min_vram_gb=N)`
from `test.predicates` instead.

**Bypassing resource checks:** pass `--skip-resource-checks` to bypass `require_gpu`
and `require_ram` gates — useful for running test logic on under-spec hardware or
reproducing failures from higher-spec machines. API credential and Ollama checks are
unaffected. On machines with no GPU at all, gated tests will run and may fail naturally.
The env var `_MELLEA_SKIP_RESOURCE_CHECKS=1` has the same effect and can be used in CI
environments without modifying the pytest invocation.

## Auto-detection

The test suite automatically detects system capabilities and skips tests whose
requirements are not met. No manual configuration needed.

| Capability | How detected |
|------------|--------------|
| Ollama | TCP check at collection time (`OLLAMA_HOST`, default `:11434`) |
| GPU / VRAM | `torch` + `sysctl hw.memsize` |
| API keys | Environment variable check |

Run `pytest -rs` to see skip reasons for each skipped test.

## Common patterns

```python
# Unit — no markers needed (auto-applied by conftest)
def test_cblock_repr():
    assert str(CBlock(value="hi")) == "hi"

# Integration — mocked backend, real project components
@pytest.mark.integration
def test_session_with_mock(mock_backend):
    session = start_session(backend=mock_backend)
    result = session.instruct("hello")
    assert mock_backend.generate.called

# E2E — real Ollama backend, deterministic assertion
pytestmark = [pytest.mark.e2e, pytest.mark.ollama]

def test_structured_output(session):
    result = session.format(Person, "Make up a person")
    assert isinstance(json.loads(result.value), dict)

# Qualitative — real backend, non-deterministic content check
pytestmark = [pytest.mark.e2e, pytest.mark.ollama]

@pytest.mark.qualitative
def test_greeting_content(session):
    result = session.instruct("Write a greeting")
    assert "hello" in result.value.lower()

# Heavy GPU e2e — resource predicate for gating
from test.predicates import require_gpu

pytestmark = [pytest.mark.e2e, pytest.mark.huggingface, require_gpu(min_vram_gb=20)]
```

## Authoring guide

### Naming and structure

- File: `test_<module>.py` in a directory mirroring the source (e.g.
  `test/backends/test_ollama.py` for `mellea/backends/ollama.py`).
- Files must be named `test_*.py` so that pydocstyle ignores them.
- Function: `test_<subject>_<scenario>_<expected>`, written so the name reads
  as a sentence.
- One behavioural claim per test. If a test has `and` in the name, split it.

### Fixture discipline

The global `test/conftest.py` provides:

| Fixture | Scope | Use |
|---------|-------|-----|
| `gh_run` | session | Returns `1` when `CICD=1` is set; use for CI-conditional behaviour |
| `system_capabilities` | session | Detected hardware/service capabilities (GPU, Ollama, API keys) |

Backend-specific fixtures (e.g. a pre-configured `session` against
`granite4:micro`, or a `mock_backend` for unit/integration tests) are defined
per test module or per-directory conftest — check the test files closest to
what you're adding before creating new fixtures.

Rules:

- Reuse existing fixtures before creating new ones.
- Do not create session-scoped fixtures that depend on real backends — they
  prevent test isolation and make skip logic unreliable.
- For Ollama-backed tests, the conftest evicts models between test modules
  automatically. Do not add `keep_alive` management in individual tests.

### Mock discipline

- Do not mock what you can replace with a real test double.
- Do not mock internal project components unless the test is explicitly testing
  the boundary *around* that component.
- When you must mock a backend for a unit or integration test, mock at the
  backend's public method boundary (`generate_from_chat_context`,
  `generate_from_raw`), not by patching internal Mellea classes.

### Assertions

- Assert one observable outcome per test.
- Prefer specific assertions (`isinstance(result.value, str)`) over broad ones
  (`result is not None`).
- Do not assert on `repr()` strings — they break on whitespace changes.

### Slow tests

Mark any test taking more than one minute with `@pytest.mark.slow`. Slow tests
are excluded from the default `pytest` invocation and from CI. Run them
explicitly with `pytest -m slow`.

## Running tests

### Environment variables

| Variable | Effect |
|----------|--------|
| `CICD=1` | Skips qualitative tests (mirrors CI behaviour) |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Helps with GPU memory fragmentation |

### Common command lines

```bash
# Fast loop — unit + integration + e2e, no qualitative (~2 min)
uv run pytest -m "not qualitative"

# Default — all tiers including qualitative, skips slow
uv run pytest

# Slow tests only
uv run pytest -m slow

# Single backend
uv run pytest -m ollama
uv run pytest -m "e2e and ollama and not qualitative"

# Specific file or test
uv run pytest test/backends/test_ollama.py
uv run pytest test/backends/test_ollama.py::test_structured_output_returns_valid_json

# See why tests were skipped
uv run pytest -rs

# CI mode locally (mirrors what PR CI does)
CICD=1 uv run pytest test

# Nightly-style local run on a GPU host
./test/scripts/run_tests_with_ollama_and_vllm.sh --group-by-backend -v -s
```

### Scoping a test run

A pytest run can be scoped along four independent axes; combine them as needed.

| Axis | Flag / form | Examples |
|------|-------------|---------|
| **By tier** | `-m <marker>` | `-m unit`, `-m integration`, `-m e2e`, `-m qualitative`, `-m slow` |
| **By backend** | `-m <backend>` | `-m ollama`, `-m huggingface`, `-m "openai or watsonx"` |
| **By compound expression** | `-m "<expr>"` | `-m "e2e and ollama and not qualitative"` |
| **By path / node id** | positional | `pytest test/backends/test_ollama.py`, `pytest test/foo.py::test_bar` |

The `addopts` in `pyproject.toml` adds `-m "not slow"` to every invocation, so
slow tests are always excluded unless you pass `-m slow` yourself. Qualitative
tests run by default locally and are skipped only when `CICD=1` is set.

### Auto-skip behaviour

Tests skip automatically when requirements are not met:

- **Ollama tests** skip at *collection time* if Ollama is not reachable
  (`OLLAMA_HOST`, default port 11434), preventing fixture setup errors before
  the skip decision.
- **GPU/HuggingFace/vLLM tests** skip if no GPU is detected or VRAM is below
  the test's requirement.
- **Cloud API tests** skip if required environment variables are unset.

## CI pipeline

| Tier | Trigger | Where | What runs |
|------|---------|-------|-----------|
| **Pre-commit** | Every commit (local) | Local hook | ruff, mypy, uv-lock, codespell, markdownlint |
| **PR CI** | Every push / merge group | GitHub Actions, Ubuntu | `pytest test/` on Python 3.11/3.12/3.13 with Ollama. `CICD=1` (qualitative skipped). `slow` excluded. |
| **Nightly** | Scheduled | IBM internal LSF cluster (GPU) | Full `pytest test/ --group-by-backend`, Ollama + vLLM, qualitative enabled. Failures file an auto-issue. |
| **On-demand nightly** | Not yet available | IBM internal LSF cluster | Comment-triggered nightly against a PR branch. Tracked in [#734](https://github.com/generative-computing/mellea/issues/734); ask a maintainer if you need pre-merge GPU validation today. |

**PR CI** (`ci.yml` → `quality.yml`): pre-commit checks, then Ollama installed
and `granite4:micro` + `granite4:micro-h` pulled, then `uv run -m pytest -v
--junit-xml=... test`. `docs/examples/` is not collected in PR CI.

**Nightly** (`test/scripts/run_tests_with_ollama_and_vllm.sh`): starts local
Ollama and (when GPU present) a local vLLM server, then runs
`pytest test/ --group-by-backend`. The `--group-by-backend` flag reorders tests
to run each backend as a contiguous group, reducing GPU memory fragmentation.

## Coverage

Branch coverage is enabled and runs automatically with every test invocation.
Reports are written to `htmlcov/` and `coverage.json`.

```bash
uv run pytest
open htmlcov/index.html        # macOS
xdg-open htmlcov/index.html    # Linux
```

Coverage is measured over `mellea/` and `cli/`. Test files and `docs/` are
excluded. There is no enforced minimum threshold; use coverage locally to
identify untested paths. Uploading artifacts and trend reporting is an open gap
([#737](https://github.com/generative-computing/mellea/issues/737)).

## Examples as tests

Files in `docs/examples/` are not auto-collected. A file is only executed by
pytest if it has an opt-in comment near the top:

```python
# pytest: e2e, ollama, qualitative
"""Greeting example — demonstrates session.instruct()."""
```

The comment lists comma-separated marker names (not `-m` expression syntax —
no `and`/`or`/`not`). Files without this comment are silently ignored and do
not appear in skip summaries or collection output.

The same classification rules and marker conventions apply as for `test/`
files. Only add the `# pytest:` comment when the example has the necessary
dependencies documented and should be part of the regression suite.

Parser: `docs/examples/conftest.py` (`_extract_markers_from_file`).

## Ollama model eviction

When pytest orchestrates many Ollama-backed tests in sequence, the default 5-minute
keep-alive means models from earlier tests stay resident and accumulate, eventually
starving later tests of memory.

Two mechanisms in `test/conftest.py` handle this:

- **Per-module eviction** (`pytest_runtest_teardown`) — when crossing a file
  boundary between Ollama-marked tests, queries `/api/ps` for all loaded models
  and evicts them with `keep_alive=0`. Covers both `test/` and `docs/examples/`.
  Always active, no flags required.
- **Group warm-up/eviction** (`pytest_runtest_setup`) — warms up a fixed set of CI
  models (`keep_alive=-1`) when entering the Ollama backend group and evicts them
  when leaving. Requires `--group-by-backend`.

**Trade-off:** if two consecutive test files use the same model, it will be unloaded
and reloaded (~5–15 s overhead). Predictable memory behaviour is more important
than saving a reload, especially on constrained CI runners. Tests within a single
file share the loaded model with no overhead.

**Caveat:** eviction targets *all* loaded Ollama models, not just those loaded by
the test. If you are using Ollama interactively while the suite runs, your model
will be evicted between test modules.

## GPU testing on CUDA systems

### The problem: CUDA EXCLUSIVE_PROCESS mode

When running GPU tests on systems with `EXCLUSIVE_PROCESS` mode (common on HPC
clusters), you may encounter "CUDA device busy" errors. This happens because:

1. The pytest parent process creates a CUDA context when running regular tests.
2. Example tests run in subprocesses (via `docs/examples/conftest.py`).
3. In `EXCLUSIVE_PROCESS` mode, only one process can hold a CUDA context per GPU.
4. Subprocesses fail with "CUDA device busy" when the parent still holds the context.

### Solution 1: NVIDIA MPS (recommended)

**NVIDIA Multi-Process Service (MPS)** allows multiple processes to share a GPU
in `EXCLUSIVE_PROCESS` mode. Enable it via your job scheduler configuration;
consult your HPC documentation for specific syntax.

### Solution 2: run smaller test subsets

If MPS is unavailable, run `test/` and `docs/examples/` in separate invocations:

```bash
pytest -m huggingface test/
pytest -m huggingface docs/examples/
```

If conflicts persist, continue breaking down into smaller subsets.

### Why this matters

The test infrastructure runs examples in subprocesses to isolate execution and
capture stdout/stderr cleanly, but this creates the "parent trap": the parent
pytest process holds a CUDA context from running regular tests, blocking
subprocesses from accessing the GPU.

**Approaches that do not work:** `torch.cuda.empty_cache()` (only affects the
PyTorch allocator, not the driver context), `cudaDeviceReset()` in subprocesses
(parent still holds the context), inter-example delays.
