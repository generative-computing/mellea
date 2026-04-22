# Mellea Test Suite

Test files must be named as `test_*.py` so that pydocstyle ignores them.

## Running Tests

```bash
# Fast tests only (~2 min) - skips qualitative and slow tests
uv run pytest -m "not qualitative"

# Default - includes qualitative tests, skips slow tests
uv run pytest

# All tests including slow tests (>5 min)
uv run pytest -m slow
```

## Environment Variables

- `CICD=1` - Enable CI mode (skips qualitative tests)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` - Helps with GPU memory fragmentation

## Ollama Model Eviction

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
and reloaded (~5-15 s overhead). Predictable memory behaviour is more important
than saving a reload, especially on constrained CI runners. Tests within a single
file share the loaded model with no overhead.

**Caveat:** eviction targets *all* loaded Ollama models, not just those loaded by
the test. If you are using Ollama interactively while the suite runs, your model
will be evicted between test modules.

## GPU Testing on CUDA Systems

### The Problem: CUDA EXCLUSIVE_PROCESS Mode

When running GPU tests on systems with `EXCLUSIVE_PROCESS` mode (common on HPC clusters), you may encounter "CUDA device busy" errors. This happens because:

1. **Parent Process Context**: The pytest parent process creates a CUDA context when running regular tests
2. **Subprocess Blocking**: Example tests run in subprocesses (via `docs/examples/conftest.py`)
3. **Exclusive Access**: In `EXCLUSIVE_PROCESS` mode, only one process can hold a CUDA context per GPU
4. **Result**: Subprocesses fail with "CUDA device busy" when the parent still holds the context

### Solution 1: NVIDIA MPS (Recommended)

**NVIDIA Multi-Process Service (MPS)** allows multiple processes to share a GPU in `EXCLUSIVE_PROCESS` mode:

```bash
# Enable MPS in your job scheduler configuration
# Consult your HPC documentation for specific syntax
```

### Solution 2: Run Smaller Test Subsets

If MPS is unavailable, break down test execution into smaller subsets to avoid GPU sharing conflicts:

```bash
# Run tests and examples separately
pytest -m huggingface test/
pytest -m huggingface docs/examples/

# Or run specific test directories
pytest test/backends/test_huggingface.py
pytest docs/examples/safety/
```

**Note**: If conflicts persist, continue breaking down into smaller subsets until tests pass. The key is reducing the number of concurrent GPU-using processes.

### Why This Matters

The test infrastructure runs examples in subprocesses (see `docs/examples/conftest.py`) to:
- Isolate example execution environments
- Capture stdout/stderr cleanly
- Prevent cross-contamination between examples

However, this creates the "Parent Trap": the parent pytest process holds a CUDA context from running regular tests, blocking subprocesses from accessing the GPU.

### Technical Details

**CUDA Context Lifecycle**:
- Created on first CUDA operation (e.g., `torch.cuda.is_available()`)
- Persists until process exit or explicit `cudaDeviceReset()`
- In `EXCLUSIVE_PROCESS` mode, blocks other processes from GPU access

**MPS Architecture**:
- Runs as a proxy service between applications and GPU driver
- Multiplexes CUDA contexts from multiple processes onto single GPU
- Transparent to applications - no code changes needed
- Requires explicit enablement via job scheduler flags

**Alternative Approaches Tried** (documented in `GPU_PARENT_TRAP_SOLUTION.md`):
- ❌ `torch.cuda.empty_cache()` - Only affects PyTorch allocator, not driver context
- ❌ `cudaDeviceReset()` in subprocesses - Parent still holds context
- ❌ Inter-example delays - Doesn't release parent context
- ❌ pynvml polling - Can't force parent to release context
- ✅ MPS - Allows GPU sharing without code changes

## Test Markers

See [`MARKERS_GUIDE.md`](MARKERS_GUIDE.md) for complete marker documentation.

Key markers for GPU testing:
- `@pytest.mark.vllm` - Requires vLLM backend (local, GPU required)
- `@pytest.mark.huggingface` - Requires HuggingFace backend (local, GPU-heavy)
- `@pytest.mark.slow` - Tests taking >5 minutes

### Resource gating (predicates)

Use predicate functions from `test/predicates.py` for resource gating:

```python
from test.predicates import require_gpu, require_ram

pytestmark = [pytest.mark.e2e, pytest.mark.huggingface, require_gpu(), require_ram(min_gb=48)]
```

| Predicate | Use when test needs |
| --------- | ------------------- |
| `require_gpu()` | Any GPU (CUDA or MPS) |
| `require_gpu(min_vram_gb=N)` | GPU with at least N GB VRAM |
| `require_ram(min_gb=N)` | N GB+ system RAM |
| `require_api_key("ENV_VAR")` | Specific API credentials |

Pass `--skip-resource-checks` to bypass `require_gpu` and `require_ram` gates — useful for running test logic on under-spec hardware or reproducing failures from higher-spec machines. API credential and Ollama checks are unaffected. On machines with no GPU at all, gated tests will run and may fail naturally.

The env var `_MELLEA_SKIP_RESOURCE_CHECKS=1` has the same effect and can be used in CI environments without modifying the pytest invocation.

> **Deprecated:** The markers `requires_gpu`, `requires_heavy_ram`, `requires_api_key`,
> and `requires_gpu_isolation` are deprecated. Existing tests using them still work
> (conftest auto-skip handles them) but new tests must use predicates. Migrate legacy
> markers to predicates when touching those files. `require_gpu_isolation()` has been
> removed — use `--group-by-backend` for backend grouping instead.

## Coverage

Coverage reports are generated in `htmlcov/` and `coverage.json`.
