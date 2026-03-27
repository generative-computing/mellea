import gc
import os
import subprocess
import sys

import pytest
import requests

from mellea.core import FancyLogger

# Try to import optional dependencies for system detection
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ============================================================================
# System Capability Detection
# ============================================================================


def _check_ollama_available():
    """Check if Ollama is available by checking if port 11434 is listening.

    Note: This only checks if Ollama is running, not which models are loaded.
    Tests may still fail if required models (e.g., granite4:micro) are not pulled.
    """
    import socket

    host_str = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    # OLLAMA_HOST may be "host:port" or just "host" (bare IP without port)
    if ":" in host_str:
        host, port = host_str.rsplit(":", 1)
        port = int(port)
    else:
        host, port = host_str, int(os.environ.get("OLLAMA_PORT", 11434))
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


_capabilities_cache: dict | None = None


def get_system_capabilities():
    """Detect system capabilities for test requirements."""
    global _capabilities_cache
    if _capabilities_cache is not None:
        return _capabilities_cache

    capabilities = {
        "has_gpu": False,
        "gpu_memory_gb": 0,
        "ram_gb": 0,
        "has_api_keys": {},
        "has_ollama": False,
    }

    # Detect GPU (CUDA for NVIDIA, MPS for Apple Silicon)
    import platform as _platform
    import subprocess as _subprocess

    _is_apple_silicon = sys.platform == "darwin" and _platform.machine() == "arm64"

    if _is_apple_silicon:
        capabilities["has_gpu"] = True
        try:
            out = _subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            total_gb = int(out.stdout.strip()) / (1024**3)
            capabilities["gpu_memory_gb"] = min(total_gb * 0.75, total_gb - 16)
        except Exception:
            pass
    elif HAS_TORCH:
        if torch.cuda.is_available():
            capabilities["has_gpu"] = True
            try:
                # Use nvidia-smi to avoid initializing CUDA in parent process.
                # torch.cuda.get_device_properties(0) creates a CUDA context,
                # which causes "Cannot re-initialize CUDA in forked subprocess"
                # when vLLM's EngineCore forks (vLLM v1 uses multiprocessing).
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                capabilities["gpu_memory_gb"] = float(result.stdout.strip()) / 1024
            except Exception:
                pass

    # Detect RAM
    if HAS_PSUTIL:
        capabilities["ram_gb"] = psutil.virtual_memory().total / (1024**3)

    # Detect API keys
    api_key_vars = {
        "openai": "OPENAI_API_KEY",
        "watsonx": ["WATSONX_API_KEY", "WATSONX_URL", "WATSONX_PROJECT_ID"],
    }

    for backend, env_vars in api_key_vars.items():
        if isinstance(env_vars, str):
            env_vars = [env_vars]
        capabilities["has_api_keys"][backend] = all(
            os.environ.get(var) for var in env_vars
        )

    # Detect Ollama availability
    capabilities["has_ollama"] = _check_ollama_available()

    _capabilities_cache = capabilities
    return capabilities


@pytest.fixture(scope="session")
def system_capabilities():
    """Fixture providing system capabilities."""
    return get_system_capabilities()


@pytest.fixture(scope="session")
def gh_run() -> int:
    return int(os.environ.get("CICD", 0))  # type: ignore


@pytest.fixture(scope="session")
def shared_vllm_backend(request):
    """Shared vLLM backend for ALL vLLM tests across all modules.

    When --group-by-backend is used, delays creation until after openai_vllm group.
    Uses IBM Granite 4 Micro as a small, fast model suitable for all vLLM tests.
    """
    # When using --group-by-backend, delay backend creation until after openai_vllm group
    if request.config.getoption("--group-by-backend", default=False):
        # Check if we're currently in the openai_vllm group
        if hasattr(pytest_runtest_setup, "_last_backend_group"):
            current_group = pytest_runtest_setup._last_backend_group
            if current_group == "openai_vllm":
                logger = FancyLogger.get_logger()
                logger.info(
                    "Backend grouping enabled: Delaying vLLM backend creation until after openai_vllm group"
                )
                yield None
                return

    try:
        import mellea.backends.model_ids as model_ids
        from mellea.backends.vllm import LocalVLLMBackend
    except ImportError:
        pytest.skip("vLLM backend not available")
        return

    try:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for vLLM tests")
            return
    except ImportError:
        pytest.skip("PyTorch not available")
        return

    logger = FancyLogger.get_logger()
    logger.info(
        "Creating shared vLLM backend (session-scoped) for all vLLM tests. "
        "This backend will be reused to avoid GPU memory fragmentation."
    )

    backend = LocalVLLMBackend(
        model_id=model_ids.IBM_GRANITE_4_MICRO_3B,
        model_options={
            "gpu_memory_utilization": 0.6,
            "max_model_len": 4096,
            "max_num_seqs": 4,
        },
    )

    logger.info("Shared vLLM backend created successfully.")
    yield backend

    logger.info("Cleaning up shared vLLM backend (end of test session)")
    cleanup_gpu_backend(backend, "shared-vllm")


# ============================================================================
# Backend Test Grouping Configuration
# ============================================================================

# Define backend groups for organized test execution
# This helps reduce GPU memory fragmentation by running all tests for a
# backend together before switching to the next backend
BACKEND_GROUPS = {
    "huggingface": {
        "marker": "huggingface",
        "description": "HuggingFace backend tests (GPU)",
    },
    "openai_vllm": {
        "marker": "openai",
        "description": "OpenAI backend tests with vLLM server (subprocess)",
    },
    "vllm": {
        "marker": "vllm",
        "description": "vLLM backend tests (GPU, shared in-process backend)",
    },
    "ollama": {
        "marker": "ollama",
        "description": "Ollama backend tests (local server)",
    },
    "api": {
        "marker": "requires_api_key",
        "description": "API-based backends (OpenAI, Watsonx, Bedrock)",
    },
}

# Execution order when --group-by-backend is used
BACKEND_GROUP_ORDER = ["huggingface", "openai_vllm", "vllm", "ollama", "api"]


# ============================================================================
# Pytest Marker Registration and CLI Options
# ============================================================================


def pytest_addoption(parser):
    """Add custom command-line options.

    Uses safe registration to avoid conflicts when both test/ and docs/
    conftest files are loaded.
    """

    # Helper to safely add option only if it doesn't exist
    def add_option_safe(option_name, **kwargs):
        try:
            parser.addoption(option_name, **kwargs)
        except ValueError:
            # Option already exists (likely from docs/examples/conftest.py)
            pass

    add_option_safe(
        "--disable-default-mellea-plugins",
        action="store_true",
        default=False,
        help="Register all acceptance plugin sets for every test",
    )
    add_option_safe(
        "--group-by-backend",
        action="store_true",
        default=False,
        help="Group tests by backend and run them together (reduces GPU memory fragmentation)",
    )


BACKEND_MARKERS: dict[str, str] = {
    "ollama": "Tests requiring Ollama backend (local, light)",
    "openai": "Tests requiring OpenAI API (requires API key)",
    "watsonx": "Tests requiring Watsonx API (requires API key)",
    "huggingface": "Tests requiring HuggingFace backend (local, heavy)",
    "vllm": "Tests requiring vLLM backend (local, GPU required)",
    "litellm": "Tests requiring LiteLLM backend",
    "bedrock": "Tests requiring AWS Bedrock backend (requires credentials)",
}
"""Single source of truth for backend marker names and descriptions.

Add new backends here — ``pytest_configure`` registers them automatically.
Keep ``pyproject.toml`` ``[tool.pytest.ini_options].markers`` in sync.
"""


def pytest_configure(config):
    """Register custom markers."""
    # Backend markers (driven by BACKEND_MARKERS registry)
    for name, desc in BACKEND_MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {desc}")

    # Capability markers
    config.addinivalue_line(
        "markers", "requires_api_key: Tests requiring external API keys"
    )
    config.addinivalue_line("markers", "qualitative: Non-deterministic quality tests")

    # Granularity markers
    config.addinivalue_line(
        "markers",
        "unit: Self-contained tests — no services, no I/O (auto-applied when no other granularity marker present)",
    )
    config.addinivalue_line(
        "markers",
        "integration: Tests needing additional services or multi-component wiring (may use fixture-managed dependencies)",
    )
    config.addinivalue_line(
        "markers",
        "e2e: Tests against real backends — cloud APIs, local servers, or GPU-loaded models",
    )

    # Composite markers (llm is deprecated — use e2e instead)
    config.addinivalue_line(
        "markers", "llm: Tests that make LLM calls (deprecated — use e2e instead)"
    )

    # Store vLLM isolation flag in config
    config._vllm_process_isolation = False


# ============================================================================
# Heavy GPU Test Process Isolation
# ============================================================================


# ============================================================================
# vLLM Backend Cleanup Helper
# ============================================================================


def cleanup_gpu_backend(backend, backend_name="unknown"):
    """Release GPU memory held by a model backend.

    Cleans up ALL GPU-resident state: model weights, KV caches, adapter
    weights, class-level caches, and accelerate dispatch hooks.

    Args:
        backend: The backend instance to clean up.
        backend_name: Name for logging.
    """
    import gc

    logger = FancyLogger.get_logger()
    logger.info(f"Cleaning up {backend_name} backend GPU memory...")

    try:
        import torch

        if torch.cuda.is_available():
            free_before, total = torch.cuda.mem_get_info()
            logger.info(
                f"  GPU before cleanup: {free_before / 1024**3:.1f}GB free "
                f"/ {total / 1024**3:.1f}GB total"
            )
        else:
            free_before = 0

        # 1. Clear the LRU cache (holds DynamicCache KV tensors on GPU)
        if hasattr(backend, "_cache") and hasattr(backend._cache, "cache"):
            for key in list(backend._cache.cache.keys()):
                value = backend._cache.cache.pop(key)
                if backend._cache.on_evict is not None:
                    try:
                        backend._cache.on_evict(value)
                    except Exception:
                        pass
            logger.info("  Cleared LRU cache")

        # 2. Clear class-level _cached_blocks (DynamicCache on GPU, shared
        #    across all instances of LocalHFBackend)
        try:
            from mellea.backends.huggingface import LocalHFBackend

            if LocalHFBackend._cached_blocks:
                for key in list(LocalHFBackend._cached_blocks.keys()):
                    dc = LocalHFBackend._cached_blocks.pop(key)
                    if hasattr(dc, "key_cache"):
                        dc.key_cache.clear()
                    if hasattr(dc, "value_cache"):
                        dc.value_cache.clear()
                    del dc
                logger.info("  Cleared class-level _cached_blocks")
        except ImportError:
            pass

        # 3. Unload PEFT adapters (hold GPU weights)
        if hasattr(backend, "_loaded_adapters"):
            backend._loaded_adapters.clear()
        if hasattr(backend, "_added_adapters"):
            backend._added_adapters.clear()

        # 4. Delete llguidance tokenizer
        if hasattr(backend, "_llguidance_tokenizer"):
            del backend._llguidance_tokenizer

        # 5. Remove accelerate dispatch hooks before moving model to CPU.
        #    Models loaded with device_map="cuda" have hooks that can
        #    prevent .cpu() from fully releasing VRAM.
        if hasattr(backend, "_model"):
            try:
                from accelerate.hooks import remove_hook_from_module

                remove_hook_from_module(backend._model, recurse=True)
                logger.info("  Removed accelerate dispatch hooks")
            except (ImportError, Exception):
                pass

            # Move model to CPU to free VRAM
            try:
                backend._model.cpu()
            except Exception:
                pass
            try:
                del backend._model
            except AttributeError:
                pass  # _model is a @property on vLLM backends (no deleter)

        # 6. Delete tokenizer
        if hasattr(backend, "_tokenizer"):
            del backend._tokenizer

        # 7. vLLM backends
        if hasattr(backend, "_underlying_model"):
            try:
                backend._underlying_model.shutdown()
            except Exception:
                pass
            del backend._underlying_model

        # 8. Force garbage collection and flush device caches
        gc.collect()
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            free_after, total = torch.cuda.mem_get_info()
            logger.info(
                f"  GPU after cleanup: {free_after / 1024**3:.1f}GB free "
                f"/ {total / 1024**3:.1f}GB total "
                f"(reclaimed {(free_after - free_before) / 1024**3:.1f}GB)"
            )
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    except ImportError:
        pass


# ============================================================================
# Test Collection Filtering
# ============================================================================


def pytest_collection_modifyitems(config, items):
    """Skip tests at collection time based on markers and optionally reorder by backend.

    This prevents fixture setup errors for tests that would be skipped anyway.
    When --group-by-backend is used, reorders tests to group by backend.
    """
    capabilities = get_system_capabilities()

    skip_ollama = pytest.mark.skip(
        reason="Ollama not available (port 11434 not listening)"
    )

    # Auto-apply 'unit' marker to tests without explicit granularity markers.
    # This enables `pytest -m unit` without per-file maintenance burden.
    _NON_UNIT = {"integration", "e2e", "qualitative", "llm"}

    for item in items:
        # Skip ollama tests if ollama not available
        if item.get_closest_marker("ollama"):
            if not capabilities["has_ollama"]:
                item.add_marker(skip_ollama)

        # Auto-apply unit marker
        if not any(item.get_closest_marker(m) for m in _NON_UNIT):
            item.add_marker(pytest.mark.unit)

    # Reorder tests by backend if requested
    if config.getoption("--group-by-backend", default=False):
        logger = FancyLogger.get_logger()
        logger.info("Grouping tests by backend (--group-by-backend enabled)")

        # Group items by backend
        grouped_items = []
        seen = set()

        for group_name in BACKEND_GROUP_ORDER:
            marker = BACKEND_GROUPS[group_name]["marker"]
            group_tests = [
                item
                for item in items
                if item.get_closest_marker(marker) and id(item) not in seen
            ]

            if group_tests:
                logger.info(
                    f"Backend group '{group_name}': {len(group_tests)} tests ({BACKEND_GROUPS[group_name]['description']})"
                )
                grouped_items.extend(group_tests)
                for item in group_tests:
                    seen.add(id(item))

        # Add tests without backend markers at the end
        unmarked = [item for item in items if id(item) not in seen]
        if unmarked:
            logger.info(f"Unmarked tests: {len(unmarked)} tests")
            grouped_items.extend(unmarked)

        # Reorder in place
        items[:] = grouped_items
        logger.info(f"Total tests reordered: {len(items)}")


# ============================================================================
# Test Skipping Logic (Runtime)
# ============================================================================


def pytest_runtest_setup(item):
    """Skip tests based on markers and system capabilities.

    Can be overridden with command-line options:
    """
    capabilities = get_system_capabilities()
    gh_run = int(os.environ.get("CICD", 0))
    config = item.config

    # Track backend group transitions when --group-by-backend is used
    if config.getoption("--group-by-backend", default=False):
        current_group = None
        for group_name, group_info in BACKEND_GROUPS.items():
            if item.get_closest_marker(group_info["marker"]):
                current_group = group_name
                break

        prev_group = getattr(pytest_runtest_setup, "_last_backend_group", None)

        if prev_group is not None and current_group != prev_group:
            logger = FancyLogger.get_logger()
            logger.info(
                f"Backend transition: {prev_group} → {current_group}. "
                "Running GPU cleanup."
            )

            # Clean up shared vLLM backend if leaving vLLM group
            if prev_group in ("vllm", "openai_vllm"):
                try:
                    shared_backend_defs = (
                        item.session._fixturemanager._arg2fixturedefs.get(
                            "shared_vllm_backend"
                        )
                    )
                    if shared_backend_defs:
                        backend_instance = shared_backend_defs[-1].cached_result[0]
                        if backend_instance is not None:
                            cleanup_gpu_backend(
                                backend_instance, "shared-vllm-transition"
                            )
                except Exception as e:
                    logger.warning(f"Failed to cleanup vLLM backend on transition: {e}")

            # General GPU flush for any transition
            try:
                import torch

                if torch.cuda.is_available():
                    gc.collect()
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass

        pytest_runtest_setup._last_backend_group = current_group

    # Skip qualitative tests in CI
    if item.get_closest_marker("qualitative") and gh_run == 1:
        pytest.skip(
            reason="Skipping qualitative test: got env variable CICD == 1. Used only in gh workflows."
        )

    # Skip tests requiring API keys if not available
    if item.get_closest_marker("requires_api_key"):
        for backend in ["openai", "watsonx"]:
            if item.get_closest_marker(backend):
                if not capabilities["has_api_keys"].get(backend):
                    pytest.skip(
                        f"Skipping test: {backend} API key not found in environment"
                    )

    if item.get_closest_marker("watsonx"):
        if not capabilities["has_api_keys"].get("watsonx"):
            pytest.skip(
                "Skipping test: Watsonx API credentials not found in environment"
            )

    if item.get_closest_marker("vllm"):
        if not capabilities["has_gpu"]:
            pytest.skip("Skipping test: vLLM requires GPU")

    # Note: Ollama tests are now skipped at collection time in pytest_collection_modifyitems
    # to prevent fixture setup errors


def memory_cleaner():
    """Lightweight memory cleanup — safety net for per-test GPU leaks."""
    yield

    gc.collect()
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


@pytest.fixture(autouse=True, scope="session")
def normalize_ollama_host():
    """Normalize OLLAMA_HOST to work with client libraries.

    If OLLAMA_HOST is set to 0.0.0.0 (server bind address), change it to
    127.0.0.1:11434 for client connections. This prevents connection errors
    when tests try to connect to Ollama.
    """
    original_host = os.environ.get("OLLAMA_HOST")

    # If OLLAMA_HOST starts with 0.0.0.0, replace with 127.0.0.1
    if original_host and original_host.startswith("0.0.0.0"):
        # Extract port if present, default to 11434
        if ":" in original_host:
            port = original_host.split(":", 1)[1]
        else:
            port = "11434"
        os.environ["OLLAMA_HOST"] = f"127.0.0.1:{port}"

    yield

    # Restore original value
    if original_host is not None:
        os.environ["OLLAMA_HOST"] = original_host
    elif "OLLAMA_HOST" in os.environ:
        del os.environ["OLLAMA_HOST"]


@pytest.fixture(autouse=True, scope="function")
def aggressive_cleanup():
    """Aggressive memory cleanup after each test to prevent OOM on CI runners."""
    memory_cleaner()


# ============================================================================
# Plugin Acceptance Sets
# ============================================================================


@pytest.fixture()
async def register_acceptance_sets(request):
    """Register all acceptance plugin sets (logging, sequential, concurrent, fandf).

    Usage: mark your test with ``@pytest.mark.plugins`` and request this fixture,
    or rely on the autouse ``auto_register_acceptance_sets`` fixture below.
    """
    plugins_disabled = request.config.getoption(
        "--disable-default-mellea-plugins", default=False
    )
    if not plugins_disabled:
        # If plugins are enabled, we don't need to re-enable them for this specific test.
        return

    from mellea.plugins.registry import _HAS_PLUGIN_FRAMEWORK

    if not _HAS_PLUGIN_FRAMEWORK:
        yield
        return

    from mellea.plugins import register
    from mellea.plugins.manager import shutdown_plugins
    from test.plugins._acceptance_sets import ALL_ACCEPTANCE_SETS

    for ps in ALL_ACCEPTANCE_SETS:
        register(ps)
    yield
    await shutdown_plugins()


@pytest.fixture(autouse=True, scope="session")
async def auto_register_acceptance_sets(request):
    """Auto-register acceptance plugin sets for all tests by default; disable when ``--disable-default-mellea-plugins`` is passed on the CLI."""
    disable_plugins = request.config.getoption(
        "--disable-default-mellea-plugins", default=False
    )
    if disable_plugins:
        yield
        return

    from mellea.plugins.registry import _HAS_PLUGIN_FRAMEWORK

    if not _HAS_PLUGIN_FRAMEWORK:
        yield
        return

    from mellea.plugins import register
    from mellea.plugins.manager import shutdown_plugins
    from test.plugins._acceptance_sets import ALL_ACCEPTANCE_SETS

    for ps in ALL_ACCEPTANCE_SETS:
        register(ps)
    yield
    await shutdown_plugins()


@pytest.fixture(autouse=True, scope="module")
def cleanup_module_fixtures():
    """Cleanup module-scoped fixtures to free memory between test modules."""
    memory_cleaner()
