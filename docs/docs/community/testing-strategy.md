---
title: "Test Strategy"
description: "How Mellea tests are classified, written, and run — the contributor guide to the test suite."
sidebarTitle: "Test Strategy"
---

This document explains how Mellea's test suite is organised, how to classify
new tests, and what runs in CI. It is aimed at contributors to this repository.

> **Writing tests for your own `@generative` code?** See
> [Unit Test Generative Code](../how-to/unit-test-generative-code) instead.

## Background: backends and why they matter for testing

Mellea is an LLM framework that supports multiple **backends** — the providers
that actually run language models. Understanding them is essential because they
determine which tier a test belongs to, what hardware or credentials it needs,
and how tests are automatically skipped.

| Backend | What it is | Infrastructure |
|---------|-----------|----------------|
| `ollama` | Locally-served models via the Ollama runtime | Ollama process on port 11434; light RAM (~2–4 GB) |
| `huggingface` | Models loaded directly via HuggingFace `transformers` | Local GPU required (VRAM varies by model) |
| `vllm` | High-throughput inference via vLLM | Local GPU required |
| `openai` | OpenAI API or any OpenAI-compatible endpoint | API key; some tests point this at Ollama's `/v1` |
| `watsonx` | IBM Watsonx API | API key + project ID |
| `litellm` | LiteLLM proxy (wraps other backends) | Depends on underlying backend |
| `bedrock` | AWS Bedrock API | AWS credentials |

Tests that use any of these are **e2e** tests. Tests that don't touch a backend
at all — they test pure logic, formatters, schema validation, or the telemetry
pipeline — are **unit** or **integration** tests and run on any machine with
just Python and the project dependencies.

## Philosophy

Mellea tests assert **observable contracts**, not implementation details.

- Test the public API surface, not private helpers.
- Test cross-backend behaviour where it needs to be consistent.
- A test that passes while the system is broken has negative value — prefer
  fewer, more honest tests over coverage padding.
- When a test fails, fix the **code**. Adjusting an assertion to silence a
  failure is almost always wrong; fixing the test is acceptable only if the
  test was never correctly written.
- Qualitative tests (assertions on LLM output *content*) run locally by default
  but are always skipped in CI — they use a dedicated tier so they never block
  fast feedback loops.

## Classification

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

### Integration

**Verifies that your code correctly communicates across a real boundary.**
The boundary may be a third-party SDK/library (e.g. the OTel metrics SDK, where
you assert on what was actually emitted), multiple internal project components
wired together, or a fixture-managed local service. What distinguishes
integration from unit is that at least one real external component — not a mock
or stub — is on the other side of the boundary being tested.

Integration tests do **not** use real LLM backends. Add `@pytest.mark.integration`
explicitly; no backend marker is needed.

**Positive indicators:** uses a real `InMemoryMetricReader` or `InMemorySpanExporter`
to assert on OTel output; wires multiple real Mellea components together and mocks
only at the outermost LLM boundary; breaking the interface between your code and
the external component would cause the test to fail.

**Negative indicators:** all external boundaries are replaced with `MagicMock`,
`patch`, or `AsyncMock`; no real SDK objects are instantiated.

**Tie-breaker:** if you changed the contract between your code and the external
component, would this test catch it? If yes → integration. If no → unit.

### E2E

**Tests against a real backend** — cloud APIs, local servers (Ollama), or
GPU-loaded models (HuggingFace, vLLM). No mocks on the critical path.

Add `@pytest.mark.e2e` explicitly, always combined with the appropriate backend
marker (`ollama`, `huggingface`, etc.). Assertions must be **deterministic** —
structural, type-based, or functional (not assertions on generated text content;
that's qualitative).

```python
pytestmark = [pytest.mark.e2e, pytest.mark.ollama]

def test_structured_output_returns_valid_json(session):
    result = session.format(Person, "Make up a person")
    assert isinstance(json.loads(result.value), dict)
```

### Qualitative

**A sub-tier of e2e** — same infrastructure requirements, but assertions check
**non-deterministic output content** that may vary across model versions or runs.

Add `@pytest.mark.qualitative` per-function (not at module level). The module
still needs `e2e` and the backend marker. Qualitative tests are skipped in CI
(`CICD=1`) but **run by default locally**.

```python
pytestmark = [pytest.mark.e2e, pytest.mark.ollama]

def test_structured_output_returns_valid_json(session):      # e2e — deterministic
    result = session.format(Person, "Make up a person")
    assert isinstance(json.loads(result.value), dict)

@pytest.mark.qualitative
def test_greeting_contains_salutation(session):              # qualitative — content check
    result = session.instruct("Write a greeting")
    assert "hello" in result.value.lower()
```

**Decision rule:** if swapping the model version could break the assertion
despite the system working correctly, it is `qualitative`. If the assertion
checks structure, types, or functional correctness, it is `e2e`.

### What to test at each tier

| Tier | Good candidates | Not appropriate |
|------|-----------------|-----------------|
| unit | Formatters, parsers, schema validation, config logic, pure helpers | Anything that calls a real backend |
| integration | OTel/metrics pipeline, multi-component session wiring, SDK boundary assertions | Tests where all boundaries are mocked |
| e2e | Backend protocol correctness, streaming, structured output | Assertions on generated text content |
| qualitative | Factual accuracy, instruction-following, output style | Deterministic structural checks |

### The `llm` marker (deprecated)

`llm` is a legacy alias for `e2e`. It remains registered for backwards
compatibility but must not be used in new tests.

## Authoring guide

### Naming and structure

- File: `test_<module>.py` in a directory mirroring the source (e.g.
  `test/backends/test_ollama.py` for `mellea/backends/ollama.py`).
- Function: `test_<unit>_<scenario>_<expected>` — reads as a sentence.
- One behavioural claim per test. If a test has `and` in the name, split it.

### Markers

Never write `@pytest.mark.unit` — it is auto-applied. All other granularity and
backend markers are explicit. See
[test/MARKERS_GUIDE.md](https://github.com/generative-computing/mellea/blob/main/test/MARKERS_GUIDE.md)
for the complete marker reference, including the full backend table, common
patterns, and the `/audit-markers` skill for validating marker coverage.

### Fixture discipline

Key fixtures available in `test/conftest.py`:

| Fixture | Scope | Use |
|---------|-------|-----|
| `session` | function | Pre-configured Ollama session using `granite4:micro` |
| `mock_backend` | function | Minimal in-memory backend; call count and args accessible |
| `gh_run` | session | Returns `1` when `CICD=1` is set; use for CI-conditional behaviour |
| `clean_metrics_env` | function | Resets the OTel metrics environment for telemetry tests |
| `plugins` | function | Opt-in; adds plugin instances to a session |

Rules:

- Reuse existing fixtures before creating new ones — check conftest first.
- Do not create session-scoped fixtures that depend on real backends — they
  prevent test isolation and make skip logic unreliable.
- For Ollama-backed tests, the conftest evicts models between test modules
  automatically. Do not add `keep_alive` management in individual tests.

### Mock discipline

- Do not mock what you can replace with a real test double.
- Do not mock internal project components unless the test is explicitly testing
  the boundary *around* that component.
- When you must mock a backend for a unit or integration test, mock at the
  backend protocol boundary (the `generate` / `astream` method), not by
  patching internal Mellea classes.

### Assertions

- Assert one observable outcome per test.
- Prefer specific assertions (`isinstance(result.value, str)`) over broad ones
  (`result is not None`).
- Do not assert on `repr()` strings — they break on whitespace changes.

### Slow tests

Mark any test taking more than one minute with `@pytest.mark.slow`. Slow tests
are excluded from the default `pytest` invocation and from CI. Run them
explicitly with `pytest -m slow`.

## Coverage

Coverage is measured automatically as part of the standard test run. Reports
are generated into `htmlcov/` (browsable HTML) and `coverage.json` (machine-
readable). Branch coverage is enabled.

```bash
# Run tests with coverage (default — coverage runs automatically)
uv run pytest

# Open the HTML report
open htmlcov/index.html
```

What is covered: `mellea/` and `cli/`. Test files themselves are excluded.

Coverage reports are generated in every PR CI run. There is currently no
enforced minimum threshold — the goal is to use coverage to identify untested
paths, not to optimise for a percentage.

## CI pipeline tiers

The table below describes what runs at each stage. **Currently**, only the
pre-commit and PR CI tiers are implemented. Nightly and pre-release tiers are
planned as part of epic [#726](https://github.com/generative-computing/mellea/issues/726).

| Tier | Trigger | Budget | What runs |
|------|---------|--------|-----------|
| **Pre-commit** | Every commit | < 60 s | ruff, mypy, uv-lock, codespell, markdownlint |
| **PR CI** | Every push / merge group | ≤ 3 h | pre-commit + `pytest test/` on Python 3.11/3.12/3.13 with Ollama running. `CICD=1` skips qualitative. `slow` tests excluded by default. |
| **Nightly** | Scheduled — *planned* | ~60 min | Every test, all backends, qualitative included. Tracked in [#733](https://github.com/generative-computing/mellea/issues/733). |
| **Pre-release** | Manual — *planned* | ~90 min | Nightly suite on release candidate. Tracked in [#733](https://github.com/generative-computing/mellea/issues/733). |
| **Local dev** | Ad-hoc | varies | Any subset — see [Local dev workflow](#local-dev-workflow) |

### PR CI in detail

CI runs `quality.yml`, a reusable workflow invoked by `ci.yml` on every pull
request and merge group event. For each Python version in the matrix:

1. Pre-commit checks (ruff, mypy, uv-lock, codespell, markdownlint).
2. Ollama installed and served; `granite4:micro` and `granite4:micro-h` pulled.
3. `uv run -m pytest -v --junit-xml=... test` — runs the `test/` directory only
   (not `docs/examples/`). `pyproject.toml` `addopts` excludes `slow` and
   enables coverage.
4. Job summary written with pass/fail counts and coverage.

`CICD=1` is set in the workflow environment, which triggers the conftest skip
for all `qualitative`-marked tests.

Note: `docs/examples/` is **not** collected in PR CI — examples run with their
own variable dependencies that are not installed in the CI environment.

## Local dev workflow

```bash
# Fast loop — unit + integration + e2e, no qualitative (~2 min)
uv run pytest -m "not qualitative"

# Default — adds qualitative tests, still skips slow
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
```

### Test collection scope

`pyproject.toml` sets `testpaths = ["test", "docs"]`. When running without an
explicit path argument (i.e. `uv run pytest`), both directories are collected.
`docs/examples/` uses the opt-in `# pytest:` comment filter — only files that
declare themselves runnable are executed (see [Examples as tests](#examples-as-tests)).
PR CI passes `test` explicitly and does not collect `docs/`.

### Auto-skip behaviour

When you run the suite locally, tests are automatically skipped when their
requirements are not met — no manual configuration needed:

- **Ollama tests** skip if port 11434 is not reachable.
- **GPU/HuggingFace/vLLM tests** skip if no GPU is detected or if VRAM is
  below the test's requirement.
- **Cloud API tests** skip if the required environment variables are unset.

Run `pytest -rs` to see the skip reasons for each skipped test.

### GPU tests

On machines with a GPU, e2e tests with `huggingface` or `vllm` markers run
automatically when resources are sufficient. On HPC clusters with
`EXCLUSIVE_PROCESS` GPU mode, run `test/` and `docs/examples/` in separate
invocations to avoid CUDA context conflicts. See
[test/README.md](https://github.com/generative-computing/mellea/blob/main/test/README.md)
for NVIDIA MPS setup.

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
`qualitative` tests — unit and integration tests do not use real backends.

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

### Auto-detection

The conftest detects system capabilities once per session (Ollama port check,
GPU/VRAM via `torch`, API keys via environment variables) and caches the
results. Ollama tests are skipped **at collection time** (not setup time) —
this prevents fixture setup errors from backend-dependent fixtures firing
before the skip decision is made.

### Auto-applied `unit` marker

The `conftest.pytest_collection_modifyitems` hook applies `pytest.mark.unit`
to every test that carries no explicit granularity marker (`integration`,
`e2e`, `qualitative`, or the deprecated `llm`). This means `pytest -m unit`
works without any per-file maintenance burden.

## See also

- [test/MARKERS_GUIDE.md](https://github.com/generative-computing/mellea/blob/main/test/MARKERS_GUIDE.md) — full marker reference: tier definitions, backend matrix, common patterns, and the `/audit-markers` skill
- [test/README.md](https://github.com/generative-computing/mellea/blob/main/test/README.md) — operational notes: model eviction, GPU CUDA conflicts, coverage
- [Epic #726](https://github.com/generative-computing/mellea/issues/726) — Testing Infrastructure & Strategy Overhaul (parent epic, CI tier roadmap)
