---
title: "Test Strategy"
description: "How Mellea tests are classified, written, and run — the contributor guide to the test suite."
sidebarTitle: "Test Strategy"
---

This document explains how Mellea's test suite is organised, how to classify
new tests, and what runs in CI. It is aimed at contributors to this repository.
Sections build on each other; you can stop when you have what you need.

> **Writing tests for your own `@generative` code?** See
> [Unit Test Generative Code](../how-to/unit-test-generative-code) instead.

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

## Classification

Every test belongs to exactly one granularity tier. Apply the decision rules
below in order:

| Question | Answer → tier |
|----------|---------------|
| Does it call a real LLM backend or external API? | Yes → **e2e** (or **qualitative**, see below) |
| Does it assert against a real third-party SDK object (OTel reader, metrics collector)? | Yes → **integration** |
| Does it wire multiple real project components together without external I/O? | Yes → **integration** |
| Does everything happen in-process with no real external collaborators? | Yes → **unit** (auto-applied) |

### About backends

Mellea supports multiple **backends**: the providers that actually run language
models. Which backend a test uses determines its tier, what hardware or
credentials it needs, and how the test suite skips it automatically.

| Backend | What it is | Infrastructure |
|---------|-----------|----------------|
| `ollama` | Locally-served models via the Ollama runtime | Ollama process on port 11434; light RAM (~2–4 GB) |
| `huggingface` | Models loaded directly via HuggingFace `transformers` | Local GPU required (VRAM varies by model) |
| `vllm` | High-throughput inference via vLLM | Local GPU required |
| `openai` | OpenAI API or any OpenAI-compatible endpoint | API key; some tests point this at Ollama's `/v1` |
| `watsonx` | IBM Watsonx API | API key + project ID |
| `litellm` | LiteLLM unified Python client (wraps other backends) | Depends on underlying backend |
| `bedrock` | AWS Bedrock API | AWS credentials |

Tests that use any of these are **e2e** tests. Tests that don't touch a backend
(pure logic, formatters, schema validation, or the telemetry pipeline) are
**unit** or **integration**, and run on any machine with just Python and the
project dependencies installed.

### Unit

**Entirely self-contained** — no services, no I/O, no network. Pure logic:
formatters, parsers, schema validation, config loading, pure helper functions.
Runs in milliseconds on any machine.

The `unit` marker is **auto-applied by conftest** to every test that has no
other granularity marker. Never write `@pytest.mark.unit` yourself.

### Integration

**Verifies that your code correctly communicates across a real boundary.**
The boundary may be a third-party SDK/library (e.g. the OTel metrics SDK, where
you assert on what was actually emitted), multiple internal project components
wired together, or a fixture-managed local service. What distinguishes
integration from unit is that at least one real external component — not a mock
or stub — is on the other side of the boundary being tested.

Integration tests do **not** use real LLM backends. Add `@pytest.mark.integration`
explicitly; no backend marker is needed.

**Positive indicators:**

- Wires multiple real Mellea components (e.g. a Session with a real Formatter
  or Requirement) and mocks only at the outermost LLM call
- Uses a real third-party SDK object to assert on its output — e.g.
  `InMemoryMetricReader` to verify what the OTel metrics SDK actually emitted
- Breaking the interface between your code and the external component would
  cause the test to fail

**Negative indicators:**

- All external boundaries replaced with `MagicMock`, `patch`, or `AsyncMock`
- No real SDK objects instantiated; the external library is only present as a type

**Tie-breaker:** if you changed the contract between your code and the external
component, would this test catch it? If yes → integration. If no → unit.

### E2E

**Tests against a real backend**: cloud APIs, local servers (Ollama), or
GPU-loaded models (HuggingFace, vLLM). No mocks on the critical path.

Add `@pytest.mark.e2e` explicitly, always combined with the appropriate backend
marker (`ollama`, `huggingface`, etc.). Assertions must be **deterministic**:
structural, type-based, or functional. Assertions on generated text content
belong in qualitative tests, not e2e.

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
# Module already carries e2e + backend markers from the example above
pytestmark = [pytest.mark.e2e, pytest.mark.ollama]

@pytest.mark.qualitative
def test_greeting_contains_salutation(session):
    result = session.instruct("Write a greeting")
    assert "hello" in result.value.lower()    # content check — model could legitimately
                                              # return "Hi there" and the assertion fails
```

**Decision rule:** if swapping the model version could break the assertion
despite the system working correctly, it is `qualitative`. If the assertion
checks structure, types, or functional correctness, it is `e2e`.

### What to test at each tier

| Tier | Good candidates | Not appropriate |
|------|-----------------|-----------------|
| unit | Formatters, parsers, schema validation, config logic, pure helpers | Anything that calls a real backend |
| integration | Telemetry pipeline (OTel readers), multi-component session wiring, assertions on what an external SDK actually emitted | Tests where all boundaries are mocked |
| e2e | Backend protocol correctness, streaming, structured output | Assertions on generated text content |
| qualitative | Factual accuracy, instruction-following, output style | Deterministic structural checks |

### The `llm` marker (deprecated)

`llm` is a legacy alias for `e2e`. It remains registered for backwards
compatibility but must not be used in new tests. Issue
[#729](https://github.com/generative-computing/mellea/issues/729) tracks
the ongoing migration of legacy `llm`/`e2e` tests that should be reclassified
as `integration`; you may encounter these in the existing test suite.

## Authoring guide

### Naming and structure

- File: `test_<module>.py` in a directory mirroring the source (e.g.
  `test/backends/test_ollama.py` for `mellea/backends/ollama.py`).
- Function: `test_<subject>_<scenario>_<expected>`, written so the name reads
  as a sentence.
- One behavioural claim per test. If a test has `and` in the name, split it.

### Markers

Never write `@pytest.mark.unit` — it is auto-applied. All other granularity and
backend markers are explicit. See
[test/MARKERS_GUIDE.md](https://github.com/generative-computing/mellea/blob/main/test/MARKERS_GUIDE.md)
for the complete marker reference, including the full backend table, common
patterns, and the `/audit-markers` skill for validating marker coverage.

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
  backend's public method boundary (e.g. `generate_from_chat_context`,
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

## Local dev workflow

### Scoping a test run

A pytest run can be scoped along four independent axes; combine them as needed.

| Axis | Flag / form | Examples |
|------|-------------|----------|
| **By tier** | `-m <marker>` | `-m unit`, `-m integration`, `-m e2e`, `-m qualitative`, `-m slow` |
| **By backend** | `-m <backend>` | `-m ollama`, `-m huggingface`, `-m "openai or watsonx"` |
| **By compound expression** | `-m "<expr>"` | `-m "e2e and ollama and not qualitative"`, `-m "not qualitative and not slow"` |
| **By path / node id** | positional argument | `pytest test/backends/test_ollama.py`, `pytest test/foo.py::test_bar` |

On top of these, **resource gating** narrows the set further at runtime: even
if a test matches your `-m` expression, it will skip itself if it declares
`require_gpu`, `require_ram`, `require_api_key`, `require_package`, or
`require_python` and the host doesn't satisfy it. The skip reasons are
self-documenting; pass `-rs` to see them. See
[Backend & resource gating](#backend--resource-gating) for the predicate
reference.

The `addopts` in `pyproject.toml` adds `-m "not slow"` and coverage flags to
every invocation, so `slow` tests are always excluded unless you pass `-m slow`
yourself (which overrides). `qualitative` tests run by default locally and are
skipped only when `CICD=1` is set.

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

# Nightly-style local run on a GPU host (Ollama + vLLM, grouped by backend)
./test/scripts/run_tests_with_ollama_and_vllm.sh --group-by-backend -v -s
```

### Test collection scope

`pyproject.toml` sets `testpaths = ["test", "docs"]`. When running without an
explicit path argument, both directories are collected. `docs/examples/` uses
the opt-in `# pytest:` comment filter; only files that declare themselves
runnable are executed (see [Examples as tests](#examples-as-tests)).
PR CI passes `test` explicitly and does not collect `docs/`. Nightlies run
`pytest test/` (also no `docs/`) but otherwise leave the full marker set in.

### Auto-skip behaviour

Tests are automatically skipped when their requirements are not met. No manual
configuration is needed:

- **Ollama tests** skip at *collection time* if port 11434 is not reachable,
  preventing fixture setup errors before the skip decision.
- **GPU/HuggingFace/vLLM tests** skip if no GPU is detected or if VRAM is
  below the test's requirement.
- **Cloud API tests** skip if the required environment variables are unset.

Run `pytest -rs` to see the skip reasons for each skipped test.

### GPU tests

On machines with a GPU, e2e tests with `huggingface` or `vllm` markers run
automatically when resources are sufficient. The `--group-by-backend` flag
(used by nightlies) reorders tests so backends do not interleave; this
reduces GPU memory fragmentation enough to make full runs viable on shared
hosts. On HPC clusters with `EXCLUSIVE_PROCESS` GPU mode, run `test/` and
`docs/examples/` in separate invocations to avoid CUDA context conflicts;
see [test/README.md](https://github.com/generative-computing/mellea/blob/main/test/README.md)
for NVIDIA MPS setup.

## CI pipeline

| Tier | Trigger | Where it runs | What runs |
|------|---------|---------------|-----------|
| **Pre-commit** | Every commit (local) | Local hook | ruff, mypy, uv-lock, codespell, markdownlint |
| **PR CI** | Every push / merge group | GitHub Actions, Ubuntu | `pytest test/` on Python 3.11/3.12/3.13 with Ollama running. `CICD=1` set, so qualitative is skipped. `slow` excluded via `addopts`. |
| **Nightly** | Scheduled | Bluevela LSF (GPU) | Full `pytest test/` with `--group-by-backend`, Ollama + vLLM, no `CICD=1` so qualitative runs. Failures file an auto-issue (e.g. [#985](https://github.com/generative-computing/mellea/issues/985)). |
| **On-demand nightly for a PR** | *planned* | Bluevela | Comment-triggered full nightly against a PR branch. Tracked in [#734](https://github.com/generative-computing/mellea/issues/734). |
| **Pre-release** | *planned* | Bluevela | Nightly suite against a release candidate. Tracked under epic [#726](https://github.com/generative-computing/mellea/issues/726). |
| **Local dev** | Ad-hoc | Your machine | Any subset — see [Local dev workflow](#local-dev-workflow) and [Scoping a test run](#scoping-a-test-run). |

### PR CI in detail

`ci.yml` invokes the reusable `quality.yml` on every pull request and merge
group event. For each Python version in the matrix:

1. Pre-commit checks (ruff, mypy, uv-lock, codespell, markdownlint).
2. Ollama installed and served; `granite4:micro` and `granite4:micro-h` pulled.
3. `uv run -m pytest -v --junit-xml=... test` runs the `test/` directory only
   (not `docs/examples/`). `pyproject.toml` `addopts` excludes `slow`.
4. Job summary lists pass/fail counts from the JUnit XML.

`CICD=1` is set in the workflow environment, which triggers the conftest skip
for all `qualitative`-marked tests. `docs/examples/` is **not** collected in
PR CI; examples require variable model and hardware dependencies that are
not installed in the GitHub-hosted runners.

### Nightly in detail

Nightlies run on the Bluevela LSF cluster (GPU-equipped), orchestrated outside
this repo by a `nightly.py` driver that ultimately invokes
[`test/scripts/run_tests_with_ollama_and_vllm.sh`](https://github.com/generative-computing/mellea/blob/main/test/scripts/run_tests_with_ollama_and_vllm.sh).
The script:

- Starts a local Ollama and pulls `granite4:micro`, `granite4:micro-h`,
  `granite3.2-vision`.
- When a CUDA GPU is detected (or `WITH_VLLM=1` is set), starts a local vLLM
  server with `ibm-granite/granite-4.0-micro` by default.
- Runs `pytest test/ --group-by-backend -v -s`. The `--group-by-backend` flag
  reorders tests so all `huggingface`, then `openai`/vLLM, then `ollama`,
  then API-only backends run as contiguous groups, which dramatically reduces
  GPU memory fragmentation between backends.
- Does **not** set `CICD=1`, so qualitative tests *do* run in nightlies.

When a nightly fails, an issue is filed automatically tagged with the date and
commit SHA (see [#985](https://github.com/generative-computing/mellea/issues/985)
for the format). On-demand nightlies for an in-flight PR (#734) are not yet
available; if you need pre-merge GPU validation today, ask a maintainer with
Bluevela access.

## Coverage

Branch coverage is enabled and runs automatically as part of every test
invocation. Reports are written to `htmlcov/` (browsable HTML) and
`coverage.json` (machine-readable) in your working directory.

```bash
uv run pytest              # coverage runs automatically
# macOS:
open htmlcov/index.html
# Linux:
xdg-open htmlcov/index.html
```

What is measured: `mellea/` and `cli/`. Test files and `docs/` are excluded.

Coverage runs in PR CI but the reports are not currently uploaded or surfaced;
the CI job summary shows only pass/fail counts. Uploading artifacts and adding
trend reporting is tracked in issue
[#737](https://github.com/generative-computing/mellea/issues/737). There is
no enforced minimum threshold; use coverage locally to identify untested paths.

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

## Backend & resource gating

### Backend markers

Backend markers (`ollama`, `openai`, `huggingface`, etc.) identify which
backend a test needs and drive auto-skip logic. Apply them only to `e2e` and
`qualitative` tests; unit and integration tests do not use real backends.

See the full backend marker table and common pattern combinations in
[test/MARKERS_GUIDE.md](https://github.com/generative-computing/mellea/blob/main/test/MARKERS_GUIDE.md#backend-markers).

### Resource predicates

Fine-grained resource gating uses predicate decorators from `test/predicates.py`.
They compose with `pytestmark` and produce self-documenting skip reasons:

```python
from test.predicates import require_gpu, require_api_key

# Module-level gating — applies to every test in the file
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.huggingface,
    require_gpu(min_vram_gb=20),
]
```

See [test/MARKERS_GUIDE.md — Resource Gating](https://github.com/generative-computing/mellea/blob/main/test/MARKERS_GUIDE.md#resource-gating-predicates)
for the full predicate reference, typical backend/predicate combinations, and
removed legacy markers.

## See also

- [test/MARKERS_GUIDE.md](https://github.com/generative-computing/mellea/blob/main/test/MARKERS_GUIDE.md) — full marker reference: tier definitions, backend matrix, common patterns, and the `/audit-markers` skill
- [test/README.md](https://github.com/generative-computing/mellea/blob/main/test/README.md) — operational notes: model eviction, GPU CUDA conflicts
- [Epic #726](https://github.com/generative-computing/mellea/issues/726) — Testing Infrastructure & Strategy Overhaul (parent epic, CI tier roadmap)
