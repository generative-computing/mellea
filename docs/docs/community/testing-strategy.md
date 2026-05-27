---
title: "Test Strategy"
description: "How Mellea tests are classified, written, and run — the contributor guide to the test suite."
sidebarTitle: "Test Strategy"
---

This document explains how Mellea's test suite is organised, how to classify new
tests, and what runs in CI. It is aimed at contributors to this repository.

> **Writing tests for your own `@generative` code?** See
> [Unit Test Generative Code](../how-to/unit-test-generative-code) instead.

## Philosophy

Mellea tests assert **observable contracts**, not implementation details.

- Test the public API surface, not private helpers.
- Test cross-backend behaviour where it needs to be consistent.
- A test that passes while the system is broken has negative value — prefer
  fewer, more honest tests over coverage padding.
- When a test fails, fix the **code**. Adjusting an assertion to silence a
  failure is almost always wrong; fixing the test is acceptable only if the
  test was never correctly written.
- Qualitative tests (which check LLM output content) use a separate tier so
  they never block fast feedback loops.

## Classification

Every test belongs to exactly one granularity tier. Apply the decision rules
below in order:

| Question | Answer → tier |
|----------|---------------|
| Does it touch a real LLM backend or external API? | Yes → **e2e** (or **qualitative**) |
| Does it assert against a real third-party SDK object (OTel reader, metrics collector)? | Yes → **integration** |
| Does it wire multiple real project components together without external I/O? | Yes → **integration** |
| Does everything happen in-process with no real external collaborators? | Yes → **unit** (auto-applied) |

**Qualitative** is a sub-tier of e2e, not a separate granularity. Use it only
when swapping the model could break the assertion even though the system is
working correctly. If the assertion is structural or type-based, it is `e2e`.

**Tie-breaker for integration vs unit:** if you changed the contract between
your code and the external component, would this test catch it? If yes →
integration. If no → unit.

### What to test at each tier

| Tier | Good candidates | Not appropriate |
|------|-----------------|-----------------|
| unit | Formatters, parsers, schema validation, config logic, pure helpers | Anything that calls a real backend |
| integration | OTel/metrics pipeline, multi-component session wiring, SDK boundary assertions | Tests where all boundaries are mocked |
| e2e | Backend protocol correctness, streaming, structured output | Assertions on generated text content |
| qualitative | Factual accuracy, instruction-following, output style | Deterministic structural checks |

## Authoring guide

### Naming and structure

- File: `test_<module>.py` in a directory mirroring the source (e.g.
  `test/backends/test_ollama.py` for `mellea/backends/ollama.py`).
- Function: `test_<unit>_<scenario>_<expected>` — reads as a sentence.
- One behavioural claim per test. If a test has `and` in the name, split it.

### Markers

Never write `@pytest.mark.unit` — it is auto-applied. All other granularity
markers are explicit. See [test/MARKERS_GUIDE.md](https://github.com/generative-computing/mellea/blob/main/test/MARKERS_GUIDE.md)
for the complete marker reference (tier definitions, backend markers, patterns).

```python
# Unit — no marker needed
def test_cblock_repr():
    assert str(CBlock(value="hi")) == "hi"

# Integration — explicit
@pytest.mark.integration
def test_token_metrics_emits_correct_attributes(clean_metrics_env):
    reader = InMemoryMetricReader()
    ...

# E2E — explicit, with backend marker
pytestmark = [pytest.mark.e2e, pytest.mark.ollama]

def test_structured_output_returns_valid_json(session):
    result = session.format(Person, "Make up a person")
    assert isinstance(json.loads(result.value), dict)

# Qualitative — per-function, module still carries e2e + backend
pytestmark = [pytest.mark.e2e, pytest.mark.ollama]

@pytest.mark.qualitative
def test_greeting_contains_salutation(session):
    result = session.instruct("Write a greeting")
    assert "hello" in result.value.lower()
```

### Fixture discipline

- Reuse fixtures from `test/conftest.py` before creating new ones.
- Use the `gh_run` fixture (or `CICD=1`) for CI-aware conditional logic.
- Do not create session-scoped fixtures that depend on real backends — they
  prevent test isolation and make skip logic unreliable.
- For Ollama-backed tests, the conftest evicts models between test modules
  automatically. Do not add `keep_alive` management in individual tests.

### Mock discipline

- Do not mock what you can replace with a real test double.
- Do not mock internal project components unless the test is explicitly testing
  the boundary *around* that component.
- When you must mock a backend for a unit or integration test, mock at the
  **transport** boundary, not at the Python class level.

### Assertions

- Assert one observable outcome per test.
- Prefer specific assertions (`isinstance(result.value, str)`) over broad ones
  (`result is not None`).
- Do not assert on `repr()` strings — they break on whitespace changes.

### Slow tests

Mark any test taking more than one minute with `@pytest.mark.slow`. Slow tests
are excluded from the default `pytest` invocation and from CI. Run them
explicitly with `pytest -m slow`.

## CI pipeline tiers

The table below describes what runs at each stage. **Currently**, only the
pre-commit and PR CI tiers are implemented. Nightly and pre-release tiers are
planned as part of epic [#726](https://github.com/generative-computing/mellea/issues/726).

| Tier | Trigger | Budget | What runs |
|------|---------|--------|-----------|
| **Pre-commit** | Every commit | < 60 s | ruff, mypy, uv-lock, codespell, markdownlint |
| **PR CI** | Every push / merge group | ~15 min | pre-commit + full pytest on Python 3.11/3.12/3.13, Ollama with `granite4:micro` and `granite4:micro-h` pulled. `CICD=1` skips qualitative tests. `slow` tests excluded by default. |
| **Nightly** | Scheduled — *planned* | ~60 min | Every test, all backends, qualitative included. Tracked in [#733](https://github.com/generative-computing/mellea/issues/733). |
| **Pre-release** | Manual — *planned* | ~90 min | Nightly suite on release candidate. Tracked in [#733](https://github.com/generative-computing/mellea/issues/733). |
| **Local dev** | Ad-hoc | varies | Any subset — see [Local dev workflow](#local-dev-workflow) |

### PR CI in detail

CI runs `quality.yml`, a reusable workflow invoked by `ci.yml` on every pull
request and merge group event. For each Python version in the matrix:

1. Pre-commit checks (ruff, mypy, uv-lock, codespell, markdownlint).
2. Ollama installed and served, `granite4:micro` and `granite4:micro-h` pulled.
3. `uv run -m pytest -v --junit-xml=... test` — no `-m` filter, using
   `pyproject.toml` `addopts` which excludes `slow` and runs coverage.
4. Job summary written with pass/fail counts.

`CICD=1` is set in the workflow environment. This triggers the conftest skip
for all `qualitative`-marked tests.

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
CICD=1 uv run pytest
```

### Test collection scope

`pyproject.toml` sets `testpaths = ["test", "docs"]`. The `test/` directory
is collected first (fail fast); `docs/examples/` is collected second via an
examples conftest that applies the opt-in `# pytest:` filter.

### GPU tests

On machines with a GPU, e2e tests with `huggingface` or `vllm` markers run
automatically when resources are sufficient — the conftest auto-detects
capabilities and skips cleanly when they are not met. See
[Backend & resource gating](#backend--resource-gating).

On HPC clusters with `EXCLUSIVE_PROCESS` GPU mode, run `test/` and
`docs/examples/` separately to avoid CUDA context conflicts. See
[test/README.md](https://github.com/generative-computing/mellea/blob/main/test/README.md)
for NVIDIA MPS setup.

## Examples as tests

Files in `docs/examples/` are not auto-collected. A file is only executed by
pytest if it has an opt-in `# pytest:` comment near the top:

```python
# pytest: e2e, ollama, qualitative
"""Greeting example — demonstrates session.instruct()."""
```

The comment lists comma-separated markers, in the same format as pytest `-m`
expressions. Files without this comment are silently ignored — they do not
appear in skip summaries or collection output.

The same classification rules and marker conventions apply. Because examples
have variable dependencies (models, hardware, API keys) and are intended to be
readable standalone, only add the `# pytest:` comment when the example should
be part of the test suite and has the necessary setup documented.

Parser: `docs/examples/conftest.py` (`_extract_markers_from_file`).

## Backend & resource gating

### Backend markers

Backend markers (`ollama`, `openai`, `huggingface`, etc.) identify which
backend a test needs and drive auto-skip logic. Apply them only to `e2e` and
`qualitative` tests — unit and integration tests do not use real backends.

See [test/MARKERS_GUIDE.md](https://github.com/generative-computing/mellea/blob/main/test/MARKERS_GUIDE.md#backend-markers)
for the full backend marker table.

### Resource predicates

Fine-grained resource gating uses predicate decorators from `test/predicates.py`.
They compose with `pytestmark` and give precise, self-documenting skip reasons.

```python
from test.predicates import require_gpu, require_ram, require_api_key

# Module-level gating — applies to every test in the file
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.huggingface,
    require_gpu(min_vram_gb=20),
]

# Function-level gating
@require_api_key("OPENAI_API_KEY")
def test_openai_chat():
    ...
```

| Predicate | Use when test needs |
|-----------|---------------------|
| `require_gpu()` | Any GPU (CUDA or MPS) |
| `require_gpu(min_vram_gb=N)` | GPU with at least N GB VRAM |
| `require_ram(min_gb=N)` | N GB+ system RAM (genuinely RAM-bound tests only) |
| `require_api_key("ENV_VAR", ...)` | One or more API credentials |
| `require_package("pkg")` | Optional Python dependency |
| `require_python((3, 11))` | Minimum Python version |

### Auto-detection

The conftest checks system capabilities once per session and caches the results:

| Capability | Detection method |
|------------|-----------------|
| Ollama | Port 11434 reachability check |
| GPU / VRAM | `torch` + `sysctl hw.memsize` (Apple Silicon) or `torch.cuda` (CUDA) |
| API keys | Environment variable presence |

Ollama tests are skipped **at collection time** (not setup time) — this
prevents fixture setup errors from backend-dependent fixtures.

### Auto-applied `unit` marker

The `conftest.pytest_collection_modifyitems` hook applies `pytest.mark.unit`
to every test that carries no explicit granularity marker (`integration`,
`e2e`, `qualitative`, or the deprecated `llm`). This means `pytest -m unit`
works without any per-file maintenance.

## See also

- [test/MARKERS_GUIDE.md](https://github.com/generative-computing/mellea/blob/main/test/MARKERS_GUIDE.md) — full marker reference with tier definitions, backend matrix, common patterns, and the `/audit-markers` skill
- [test/README.md](https://github.com/generative-computing/mellea/blob/main/test/README.md) — operational notes (model eviction, GPU CUDA conflicts, coverage)
- [Epic #726](https://github.com/generative-computing/mellea/issues/726) — Testing Infrastructure & Strategy Overhaul (parent epic, CI tier roadmap)
