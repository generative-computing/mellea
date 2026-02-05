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

- `CICD=1` - Enable CI mode (skips qualitative tests, enables aggressive memory cleanup)
- `VLLM_USE_V1=0` - Required for vLLM tests (automatically set by process isolation)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` - Helps with GPU memory fragmentation

## Heavy GPU Tests - Automatic Process Isolation

**Heavy GPU tests (HuggingFace, vLLM) automatically use process isolation when multiple test modules are detected.**

### Why Process Isolation?

Heavy GPU backends (HuggingFace, vLLM) hold GPU memory at the process level. Even with aggressive cleanup (garbage collection, CUDA cache clearing, etc.), GPU memory remains locked by the CUDA driver until the process exits. When running multiple heavy GPU test modules in sequence, this causes OOM errors.

### How It Works

The collection hook in `test/conftest.py` detects multiple modules with `requires_heavy_ram` marker and automatically:

1. Runs each module in a separate subprocess
2. Sets required environment variables (`VLLM_USE_V1=0`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`)
3. Ensures full GPU memory release between modules
4. Reports results from all modules

### Usage

```bash
# Run all heavy GPU tests with automatic isolation
uv run pytest -m requires_heavy_ram

# Run vLLM tests specifically
uv run pytest -m vllm

# Run HuggingFace tests specifically
uv run pytest -m huggingface

# Single module runs normally (no isolation needed)
uv run pytest test/backends/test_vllm.py

# Works with other pytest options
uv run pytest -m "requires_heavy_ram and not qualitative"
```

### Affected Tests

Tests marked with `@pytest.mark.requires_heavy_ram`:
- `test/backends/test_huggingface.py` - HuggingFace backend tests
- `test/backends/test_huggingface_tools.py` - HuggingFace tool calling
- `test/backends/test_vllm.py` - vLLM backend tests
- `test/backends/test_vllm_tools.py` - vLLM tool calling

### Technical Details

- **Single module**: Runs normally in the main pytest process
- **Multiple modules**: Each runs in its own subprocess with full GPU memory isolation
- **No external server needed**: Tests instantiate `LocalVLLMBackend` directly
- **Automatic detection**: Based on `@pytest.mark.vllm` marker

See `VLLM_MEMORY_SOLUTION.md` for detailed technical analysis.

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
- `@pytest.mark.vllm` - Requires vLLM backend (local, GPU required, auto-isolated)
- `@pytest.mark.huggingface` - Requires HuggingFace backend (local, GPU-heavy)
- `@pytest.mark.requires_gpu` - Requires GPU hardware
- `@pytest.mark.requires_heavy_ram` - Requires 48GB+ RAM
- `@pytest.mark.slow` - Tests taking >5 minutes

## Coverage

Coverage reports are generated in `htmlcov/` and `coverage.json`.
