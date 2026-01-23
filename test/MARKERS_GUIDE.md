# Pytest Markers Guide for Mellea Tests

## Overview

This guide explains the pytest marker system for categorizing and running mellea tests based on backend requirements, resource availability, and test characteristics.

## 🎯 What's Automatic vs Manual

### ✅ Automatic (No Configuration Needed)
When you run `pytest`, the system **automatically detects** and skips tests based on:
- **Ollama availability** - Checks if port 11434 is listening
- **API keys** - Checks environment variables (`OPENAI_API_KEY`, `WATSONX_API_KEY`, etc.)
- **GPU availability** - Checks for CUDA (NVIDIA) or MPS (Apple Silicon) via torch
- **System RAM** - Checks via `psutil.virtual_memory()` (if psutil installed)

**You don't need to configure anything!** Just run `pytest` and tests will automatically skip with helpful messages if requirements aren't met.

**Note:**
- GPU detection requires `torch` (included in `mellea[hf]` and `mellea[vllm]`)
- RAM detection requires `psutil` (included in dev dependencies)
- If you're not using dev dependencies, install with: `pip install psutil`

### ⚠️ Manual (Developer Adds to Test Files)
Developers must **add markers** to test files to indicate what each test needs:
```python
# Developer adds these markers once per test file
pytestmark = [pytest.mark.ollama, pytest.mark.llm]
```

**Summary:** Markers are manual (one-time setup per test file), detection is automatic (every test run).

### 🔧 Override Auto-Detection (Advanced)
Want to try running tests even when requirements aren't met? Use these pytest options:

```bash
# Try GPU tests without GPU (will use CPU, may be slow/fail)
pytest --ignore-gpu-check test/backends/test_vllm.py

# Try with less RAM than recommended
pytest --ignore-ram-check test/backends/test_huggingface.py

# Try without Ollama running
pytest --ignore-ollama-check test/backends/test_ollama.py

# Try without API keys (will fail at API call)
pytest --ignore-api-key-check test/backends/test_openai.py

# Ignore all checks at once (convenience flag)
pytest --ignore-all-checks

# Combine multiple overrides
pytest --ignore-gpu-check --ignore-ram-check -m "huggingface"
```

**Use Cases:**
- Testing with CPU when GPU tests might work (slower but functional)
- Trying with less RAM (might work for smaller models)
- Debugging test infrastructure

**Warning:** Tests will likely fail if requirements aren't actually met!

## Quick Start

```bash
# Run all tests (auto-skips based on your system)
pytest

# Run only fast unit tests (no LLM calls)
pytest -m "not llm"

# Run Ollama tests only (local, light resources)
pytest -m "ollama"

# Run tests that don't require API keys
pytest -m "not requires_api_key"

# Run infrastructure tests only (skip quality tests)
pytest -m "not qualitative"

# Run quality tests for Ollama
pytest -m "ollama and qualitative"
```

## Marker Categories

### Backend Markers

Specify which backend the test uses:

- **`@pytest.mark.ollama`**: Tests requiring Ollama backend
  - Local execution
  - Light resources (CPU, ~2-4GB RAM)
  - No API key required
  - Example: `test/backends/test_ollama.py`

- **`@pytest.mark.openai`**: Tests requiring OpenAI API
  - Requires `OPENAI_API_KEY` environment variable
  - Light resources (API calls only)
  - Incurs API costs
  - Example: `test/backends/test_vision_openai.py`

- **`@pytest.mark.watsonx`**: Tests requiring Watsonx API
  - Requires `WATSONX_API_KEY`, `WATSONX_URL`, and `WATSONX_PROJECT_ID` environment variables
  - Light resources (API calls only)
  - Incurs API costs
  - Example: `test/backends/test_watsonx.py`

- **`@pytest.mark.huggingface`**: Tests requiring HuggingFace backend
  - Local execution
  - Heavy resources (GPU recommended, 16-32GB RAM, ~8GB VRAM)
  - Downloads models (~3-8GB)
  - No API key required
  - Example: `test/backends/test_huggingface.py`

- **`@pytest.mark.vllm`**: Tests requiring vLLM backend
  - Local execution
  - Heavy resources (GPU required, 16-32GB RAM, 8GB+ VRAM)
  - Requires `VLLM_USE_V1=0` environment variable
  - Example: `test/backends/test_vllm.py`

- **`@pytest.mark.litellm`**: Tests requiring LiteLLM backend
  - Requirements depend on underlying backend
  - Example: `test/backends/test_litellm_ollama.py`

### Capability Markers

Specify resource or authentication requirements:

- **`@pytest.mark.requires_api_key`**: Tests requiring external API keys
  - Auto-skipped if required API key not found
  - Use with backend markers (openai, watsonx)

- **`@pytest.mark.requires_gpu`**: Tests requiring GPU
  - Auto-skipped if no GPU detected
  - Typically used with huggingface, vllm

- **`@pytest.mark.requires_heavy_ram`**: Tests requiring 48GB+ RAM
  - Auto-skipped if insufficient RAM detected
  - Typically used with huggingface, vllm

- **`@pytest.mark.qualitative`**: Non-deterministic quality tests
  - Tests LLM output quality rather than infrastructure
  - Skipped in CI (when `CICD=1`)
  - May be flaky due to model variability

### Composite Markers

- **`@pytest.mark.llm`**: Tests that make LLM calls
  - Requires at least Ollama to be available
  - Use to distinguish from pure unit tests

## Auto-Detection and Skipping

The test suite automatically detects your system capabilities and skips tests that cannot run:

### API Key Detection
```python
# Automatically checks for:
OPENAI_API_KEY          # For OpenAI tests
WATSONX_API_KEY         # For Watsonx tests
WATSONX_URL             # For Watsonx tests
WATSONX_PROJECT_ID      # For Watsonx tests
```

### Backend Availability Detection
```python
# Automatically detects:
- Ollama availability (checks if port 11434 is listening)
```

### Resource Detection
```python
# Automatically detects:
- GPU availability (via torch.cuda.is_available())
- GPU memory (via torch.cuda.get_device_properties())
- System RAM (via psutil.virtual_memory())
```

### Skip Messages
When a test is skipped, you'll see helpful messages (use `-rs` flag to show skip reasons):
```bash
pytest -rs

# Output:
SKIPPED [1] test/conftest.py:120: Skipping test: OPENAI_API_KEY not found in environment
SKIPPED [1] test/conftest.py:125: Skipping test: GPU not available
SKIPPED [1] test/conftest.py:130: Skipping test: Insufficient RAM (16.0GB < 32GB)
SKIPPED [1] test/conftest.py:165: Skipping test: Ollama not available (port 11434 not listening)
```

## Usage Examples

### Module-Level Markers

Apply markers to all tests in a module using `pytestmark`:

```python
# test/backends/test_ollama.py
import pytest

# All tests in this module require Ollama and make LLM calls
pytestmark = [pytest.mark.ollama, pytest.mark.llm]

def test_simple_instruct(session):
    # This test inherits ollama and llm markers
    ...
```

### Multiple Markers

Combine markers for complex requirements:

```python
# test/backends/test_huggingface.py
pytestmark = [
    pytest.mark.huggingface,
    pytest.mark.llm,
    pytest.mark.requires_gpu,
    pytest.mark.requires_heavy_ram,
]
```

### Individual Test Markers

Add markers to specific tests:

```python
@pytest.mark.qualitative
def test_output_quality(session):
    # This test checks LLM output quality
    result = session.instruct("Write a poem")
    assert "poem" in result.value.lower()
```

## Running Tests by Category

### By Backend
```bash
# Ollama only
pytest -m "ollama"

# HuggingFace only
pytest -m "huggingface"

# All API-based backends
pytest -m "openai or watsonx"
```

### By Resource Requirements
```bash
# Light tests only (no GPU, no heavy RAM)
pytest -m "not (requires_gpu or requires_heavy_ram)"

# Tests that work without API keys
pytest -m "not requires_api_key"

# GPU tests only
pytest -m "requires_gpu"
```

### By Test Type
```bash
# Infrastructure tests only (deterministic)
pytest -m "not qualitative"

# Quality tests only (non-deterministic)
pytest -m "qualitative"

# Fast unit tests (no LLM calls)
pytest -m "not llm"
```

### Complex Combinations
```bash
# Ollama infrastructure tests
pytest -m "ollama and not qualitative"

# All tests that work with just Ollama (no API keys, no GPU)
pytest -m "not (requires_api_key or requires_gpu or requires_heavy_ram)"

# Quality tests for local backends only
pytest -m "qualitative and (ollama or huggingface or vllm)"
```

## CI/CD Integration

### Current Behavior
- `CICD=1` environment variable skips all qualitative tests
- Module-level skips for heavy backends (huggingface, vllm, watsonx)

### Recommended CI Matrix
```yaml
# .github/workflows/test.yml
jobs:
  unit-tests:
    # Fast unit tests, no LLM
    run: pytest -m "not llm"
  
  ollama-tests:
    # Ollama infrastructure tests
    run: pytest -m "ollama and not qualitative"
  
  quality-tests:
    # Optional: Run quality tests on schedule
    if: github.event_name == 'schedule'
    run: pytest -m "qualitative and ollama"
```

## Adding Markers to New Tests

### Step 1: Identify Requirements
Ask yourself:
1. Which backend does this test use?
2. Does it require an API key?
3. Does it need a GPU?
4. Does it need heavy RAM (48GB+)?
5. Is it testing output quality (qualitative) or infrastructure?

### Step 2: Add Appropriate Markers

For a new Ollama test:
```python
# Use module-level marker if all tests use same backend
pytestmark = [pytest.mark.ollama, pytest.mark.llm]

@pytest.mark.qualitative  # Add if testing output quality
def test_my_new_feature(session):
    ...
```

For a new HuggingFace test:
```python
pytestmark = [
    pytest.mark.huggingface,
    pytest.mark.llm,
    pytest.mark.requires_gpu,
    pytest.mark.requires_heavy_ram,
]

@pytest.mark.qualitative
def test_my_new_feature(session):
    ...
```

### Step 3: Test Your Markers
```bash
# Verify your test is properly marked
pytest --collect-only -m "your_marker"

# Run just your test
pytest -k "test_my_new_feature"
```

## Troubleshooting

### Test Not Running
```bash
# Check which markers are applied
pytest --collect-only test/path/to/test.py

# Check why test is being skipped
pytest -v test/path/to/test.py

# Force run despite auto-skip (will likely fail if requirements not met)
pytest test/path/to/test.py --runxfail
```

### Marker Not Recognized
```bash
# List all registered markers
pytest --markers

# Check pytest.ini configuration
cat pytest.ini
```

### Auto-Skip Not Working
```bash
# Debug system capabilities
pytest --setup-show test/path/to/test.py

# Check conftest.py detection logic
# See test/conftest.py:get_system_capabilities()

# Run with verbose output to see skip reasons
pytest -v -s test/path/to/test.py
```

### Force Run Tests (Override Auto-Skip)
```bash
# Run specific test ignoring auto-skip (useful for debugging)
pytest test/backends/test_ollama.py --runxfail

# Run with specific marker, will fail if requirements not met
pytest -m "ollama" -v

# Note: Tests will fail if actual requirements (Ollama, GPU, etc.) aren't met
# This is useful for testing the test infrastructure itself
```

## Best Practices

1. **Use module-level markers** for consistent backend requirements
2. **Combine markers** to accurately describe test requirements
3. **Keep qualitative marker** for non-deterministic tests
4. **Test locally** before pushing to ensure markers work correctly
5. **Document special requirements** in test docstrings

## Related Files

- `test/conftest.py`: Auto-detection and skip logic
- `pyproject.toml`: Marker definitions and pytest configuration

## Questions?

For questions or issues with the marker system:
1. Check this guide first
2. Open an issue on GitHub with the `testing` label