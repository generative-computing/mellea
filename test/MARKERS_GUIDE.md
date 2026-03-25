# Pytest Markers Guide

## Quick Reference

```bash
# By granularity tier
pytest -m unit                          # Self-contained, no services (fast)
pytest -m integration                   # Multi-component, fixture-managed deps
pytest -m e2e                           # Real backends (ollama, APIs, GPU models)
pytest -m "e2e and not qualitative"     # Deterministic real-backend tests only

# By backend
pytest -m ollama                        # Ollama tests
pytest -m huggingface                   # HuggingFace tests
pytest -m "openai or watsonx"           # Cloud API tests

# By characteristics
pytest -m "not qualitative"             # Fast, deterministic tests (~2 min)
pytest -m qualitative                   # Non-deterministic output quality tests
pytest -m slow                          # Long-running tests (>1 min)

# Default (configured in pyproject.toml): skips slow, includes qualitative
pytest
```

## Granularity Tiers

Every test belongs to exactly one tier. The tier determines what infrastructure
the test needs and how fast/heavy it is to run.

### Unit (auto-applied)

**Entirely self-contained** — no services, no I/O, no fixtures that connect
to anything external. Pure logic testing.

- Auto-applied by conftest hook when no other granularity marker is present
- **Never write `@pytest.mark.unit` on files** — it is implicit
- Runs in milliseconds to low seconds, minimal memory
- Would pass on any machine with just Python and project deps

```python
# No markers needed — auto-applied as unit
def test_cblock_repr():
    assert str(CBlock(value="hi")) == "hi"
```

### Integration (explicit)

**Multiple components wired together**, potentially needing additional services
or fixture-managed dependencies. Backends may be mocked, stubbed, or stood up
by test fixtures. The test controls or provides its own dependencies.

- Add `@pytest.mark.integration` explicitly
- Slower than unit (fixture setup, service lifecycle), may consume more memory
- No backend markers needed — integration tests don't use real backends

```python
@pytest.mark.integration
def test_session_chains_components(mock_backend):
    session = start_session(backend=mock_backend)
    result = session.instruct("hello")
    assert mock_backend.generate.called
```

### E2E (explicit)

**Tests against real backends** — cloud APIs, local servers (ollama), or
GPU-loaded models (huggingface, vllm). No mocks on the critical path.

- Add `@pytest.mark.e2e` explicitly, always combined with backend marker(s)
- Resource/capability markers (`requires_gpu`, `requires_heavy_ram`, etc.)
  only apply to e2e and qualitative tests
- Assertions are **deterministic** — structural, type-based, or functional

```python
pytestmark = [pytest.mark.e2e, pytest.mark.ollama]

def test_structured_output(session):
    result = session.format(Person, "Make up a person")
    assert isinstance(json.loads(result.value), dict)
```

### Qualitative (explicit, per-function)

**Subset of e2e.** Same infrastructure requirements, but assertions check
**non-deterministic output content** that may vary across model versions or runs.

- Add `@pytest.mark.qualitative` per-function (not module-level)
- Module must also carry `e2e` + backend markers at module level
- Skipped in CI when `CICD=1`
- Included by default in local runs

```python
pytestmark = [pytest.mark.e2e, pytest.mark.ollama]

@pytest.mark.qualitative
def test_greeting_content(session):
    result = session.instruct("Write a greeting")
    assert "hello" in result.value.lower()
```

**Decision rule:** If swapping the model version could break the assertion
despite the system working correctly, it is `qualitative`. If the assertion
checks structure, types, or functional correctness, it is `e2e`.

### The `llm` marker (deprecated)

`llm` is a legacy marker equivalent to `e2e`. It remains registered for
backward compatibility but should not be used in new tests. Use `e2e` instead.

The conftest auto-apply hook treats `llm` the same as `e2e` — tests marked
`llm` will not receive the `unit` marker.

## Backend Markers

Backend markers identify which backend a test needs. They enable selective
test runs (`pytest -m ollama`) and drive auto-skip logic.

**Backend markers only go on e2e and qualitative tests.** Unit and integration
tests don't need real backends.

| Marker         | Backend                       | Resources                             |
| -------------- | ----------------------------- | ------------------------------------- |
| `ollama`       | Ollama (port 11434)           | Local, light (~2-4GB RAM)             |
| `openai`       | OpenAI API or compatible      | API calls (may use Ollama `/v1`)      |
| `watsonx`      | Watsonx API                   | API calls, requires credentials       |
| `huggingface`  | HuggingFace transformers      | Local, GPU, 48GB+ RAM                 |
| `vllm`         | vLLM                          | Local, GPU required, 48GB+ RAM        |
| `litellm`      | LiteLLM (wraps other backends)| Depends on underlying backend         |

### OpenAI-via-Ollama pattern

Some tests use the OpenAI client pointed at Ollama's `/v1` endpoint. Mark
these with **both** `openai` and `ollama`, but **not** `requires_api_key`:

```python
pytestmark = [pytest.mark.e2e, pytest.mark.openai, pytest.mark.ollama]
```

## Resource / Capability Markers

These markers gate tests on hardware or credentials. They only apply to
e2e and qualitative tests — unit and integration tests should never need them.
Use sparingly.

| Marker                   | Gate                                  | Auto-skip when                                    |
| ------------------------ | ------------------------------------- | ------------------------------------------------- |
| `requires_gpu`           | CUDA or MPS                           | `torch.cuda.is_available()` is False              |
| `requires_heavy_ram`     | 48GB+ system RAM                      | `psutil` reports < 48GB                           |
| `requires_gpu_isolation` | Subprocess isolation for CUDA memory  | `--isolate-heavy` not set and `CICD != 1`         |
| `requires_api_key`       | External API credentials              | Env vars missing (checked per backend)            |
| `slow`                   | Tests taking >1 minute                | Excluded by default via `pyproject.toml` addopts  |
| `qualitative`            | Non-deterministic output              | Skipped when `CICD=1`                             |

### Typical combinations

- `huggingface` → `requires_gpu` + `requires_heavy_ram` + `requires_gpu_isolation`
- `vllm` → `requires_gpu` + `requires_heavy_ram` + `requires_gpu_isolation`
- `watsonx` → `requires_api_key`
- `openai` → `requires_api_key` only when using real OpenAI API (not Ollama-compatible)

## Auto-Detection

The test suite automatically detects system capabilities and skips tests
whose requirements are not met. No configuration needed.

| Capability | How detected                  | Override flag            |
| ---------- | ----------------------------- | ------------------------ |
| Ollama     | Port 11434 check              | `--ignore-ollama-check`  |
| GPU        | `torch.cuda.is_available()`   | `--ignore-gpu-check`     |
| RAM        | `psutil.virtual_memory()`     | `--ignore-ram-check`     |
| API keys   | Environment variable check    | `--ignore-api-key-check` |
| All        | —                             | `--ignore-all-checks`    |

Use `-rs` with pytest to see skip reasons:
```bash
pytest -rs
```

## Common Marker Patterns

```python
# Unit — no markers needed (auto-applied by conftest)
def test_cblock_repr():
    assert str(CBlock(value="hi")) == "hi"

# Integration — mocked backend
@pytest.mark.integration
def test_session_with_mock(mock_backend):
    session = start_session(backend=mock_backend)
    result = session.instruct("hello")
    assert mock_backend.generate.called

# E2E — real Ollama backend, deterministic
pytestmark = [pytest.mark.e2e, pytest.mark.ollama]
def test_structured_output(session):
    result = session.format(Person, "Make up a person")
    assert isinstance(json.loads(result.value), dict)

# E2E + qualitative — real backend, non-deterministic
pytestmark = [pytest.mark.e2e, pytest.mark.ollama]
@pytest.mark.qualitative
def test_greeting_content(session):
    result = session.instruct("Write a greeting")
    assert "hello" in result.value.lower()

# Heavy GPU e2e
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.huggingface,
    pytest.mark.requires_gpu,
    pytest.mark.requires_heavy_ram,
    pytest.mark.requires_gpu_isolation,
]
```

## Example Files (`docs/examples/`)

Examples use a comment-based marker format instead of `pytestmark`:

```python
# pytest: e2e, ollama, qualitative
"""Example description..."""
```

Same classification rules apply. The comment must appear in the first few
lines before non-comment code. Parser: `docs/examples/conftest.py`
(`_extract_markers_from_file`).

## Adding Markers to New Tests

1. **Classify the test** — unit, integration, e2e, or qualitative?
2. **Add granularity marker** — integration and e2e are explicit; unit is auto-applied
3. **Add backend marker(s)** — only for e2e/qualitative
4. **Add resource markers** — only for e2e/qualitative, only when needed
5. **Verify** — `pytest --collect-only -m "your_marker"` to check

Use the `/audit-markers` skill to validate markers on existing or new test files.

## CI/CD Integration

```yaml
jobs:
  unit-tests:
    run: pytest -m unit              # Fast, no services needed

  ollama-tests:
    run: pytest -m "e2e and ollama and not qualitative"

  quality-tests:
    if: github.event_name == 'schedule'
    run: pytest -m "qualitative and ollama"
```

- `CICD=1` skips qualitative tests
- `CICD=1` enables GPU process isolation (`--isolate-heavy` behaviour)
- `slow` tests excluded by default (add `-m slow` to include)

## Related Files

- `test/conftest.py` — marker registration, auto-detection, skip logic, unit auto-apply hook
- `docs/examples/conftest.py` — example marker parser (`_extract_markers_from_file`)
- `pyproject.toml` — marker definitions and pytest configuration
- `.agents/skills/audit-markers/SKILL.md` — skill for auditing and fixing markers
