# Pytest Markers Guide

## Quick Reference

```bash
# By granularity tier
pytest -m unit                          # Self-contained, no services (fast)
pytest -m integration                   # Real SDK/library boundary or multi-component wiring
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

**Verifies that your code correctly communicates across a real boundary.**
The boundary may be a third-party SDK/library whose API contract you are
asserting against, multiple internal components wired together, or a
fixture-managed local service. What distinguishes integration from unit is
that at least one real external component — not a mock or stub — is on the
other side of the boundary being tested.

- Add `@pytest.mark.integration` explicitly
- No backend markers needed — integration tests do not use real LLM backends
- Slower than unit (fixture setup, real SDK objects), but faster than e2e

**Positive indicators:**

- Uses a real third-party SDK object to *capture and assert* on output —
  e.g. `InMemoryMetricReader`, `InMemorySpanExporter`, `LoggingHandler` —
  rather than patching the SDK away
- Asserts on the format or content of data as received by an external
  component (semantic conventions, attribute names, accumulated values)
- Wires multiple real project components together and mocks only at the
  outermost boundary
- Breaking the interface between your code and the external component
  (e.g. a changed attribute name, a missing SDK call) would cause the test
  to fail

**Negative indicators (likely unit instead):**

- All external boundaries replaced with `MagicMock`, `patch`, or `AsyncMock`
- Third-party library imported only as a type or helper, not as a real
  collaborator being asserted against
- Toggles env vars and checks booleans or config state with no real SDK
  objects instantiated

**Tie-breaker:** If you changed the contract between your code and the
external component, would this test catch it? If yes → integration. If no
→ unit.

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

### E2E (explicit)

**Tests against real backends** — cloud APIs, local servers (ollama), or
GPU-loaded models (huggingface, vllm). No mocks on the critical path.

- Add `@pytest.mark.e2e` explicitly, always combined with backend marker(s)
- Resource predicates (`require_gpu()`, `require_ram()`, etc.) only apply to
  e2e and qualitative tests — see "Resource Gating" section below
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
| `huggingface`  | HuggingFace transformers      | Local, GPU required                   |
| `vllm`         | vLLM                          | Local, GPU required                   |
| `litellm`      | LiteLLM (wraps other backends)| Depends on underlying backend         |
| `bedrock`      | AWS Bedrock                   | API calls, requires credentials       |

### OpenAI-via-Ollama pattern

Some tests use the OpenAI client pointed at Ollama's `/v1` endpoint. Mark
these with **both** `openai` and `ollama`, but do **not** add `require_api_key`:

```python
pytestmark = [pytest.mark.e2e, pytest.mark.openai, pytest.mark.ollama]
```

## Resource Gating (Predicates)

E2E and qualitative tests need gating so they skip cleanly when required
infrastructure is absent. Use **predicate decorators** from `test/predicates.py`
— they give test authors precise control over skip conditions.

```python
from test.predicates import require_gpu, require_api_key
```

| Predicate | Use when test needs |
| --------- | ------------------- |
| `require_gpu()` | Any GPU (CUDA or MPS) |
| `require_gpu(min_vram_gb=N)` | GPU with at least N GB VRAM |
| `require_ram(min_gb=N)` | N GB+ system RAM (genuinely RAM-bound tests only) |
| `require_api_key("ENV_VAR")` | Specific API credentials |
| `require_package("pkg")` | Optional dependency |
| `require_python((3, 11))` | Minimum Python version |

### Typical combinations

- `huggingface` → `require_gpu(min_vram_gb=N)` (compute N from model params)
- `vllm` → `require_gpu(min_vram_gb=N)` (compute N from model params)
- `watsonx` → `require_api_key("WATSONX_API_KEY", "WATSONX_URL", "WATSONX_PROJECT_ID")`
- `openai` → `require_api_key("OPENAI_API_KEY")` only for real OpenAI (not Ollama-compat)

### Other gating markers

These are not resource predicates but still control test selection:

| Marker         | Gate                             | Auto-skip when                                   |
| -------------- | -------------------------------- | ------------------------------------------------ |
| `slow`         | Tests taking >1 minute           | Excluded by default via `pyproject.toml` addopts |
| `qualitative`  | Non-deterministic output         | Skipped when `CICD=1`                            |

### Removed markers

`requires_gpu`, `requires_heavy_ram`, and `requires_gpu_isolation` have been
removed. Use `require_gpu(min_vram_gb=N)` from `test.predicates` instead.
`requires_api_key` is still active — see below.

## Auto-Detection

The test suite automatically detects system capabilities and skips tests
whose requirements are not met. No configuration needed.

| Capability | How detected                  |
| ---------- | ----------------------------- |
| Ollama     | Port 11434 check              |
| GPU/VRAM   | `torch` + `sysctl hw.memsize` |
| API keys   | Environment variable check    |

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

# Heavy GPU e2e (predicates for resource gating)
from test.predicates import require_gpu

pytestmark = [pytest.mark.e2e, pytest.mark.huggingface,
              require_gpu(min_vram_gb=20)]
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
4. **Add resource predicates** — only for e2e/qualitative, use `test/predicates.py`
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
- `slow` tests excluded by default (add `-m slow` to include)

## Related Files

- `test/conftest.py` — marker registration, auto-detection, skip logic, unit auto-apply hook
- `test/predicates.py` — resource gating predicates (`require_gpu`, `require_ram`, etc.)
- `docs/examples/conftest.py` — example marker parser (`_extract_markers_from_file`)
- `pyproject.toml` — marker definitions and pytest configuration
- `.agents/skills/audit-markers/SKILL.md` — skill for auditing and fixing markers
