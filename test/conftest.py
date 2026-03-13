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

    try:
        # Try to connect to Ollama's default port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", 11434))
        sock.close()
        return result == 0
    except Exception:
        return False


def get_system_capabilities():
    """Detect system capabilities for test requirements."""
    capabilities = {
        "has_gpu": False,
        "gpu_memory_gb": 0,
        "ram_gb": 0,
        "has_api_keys": {},
        "has_ollama": False,
    }

    # Detect GPU (CUDA for NVIDIA, MPS for Apple Silicon)
    if HAS_TORCH:
        has_cuda = torch.cuda.is_available()
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        capabilities["has_gpu"] = has_cuda or has_mps

        if has_cuda:
            try:
                capabilities["gpu_memory_gb"] = torch.cuda.get_device_properties(
                    0
                ).total_memory / (1024**3)
            except Exception:
                pass
        # Note: MPS doesn't provide easy memory query, leave at 0

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

    When --isolate-heavy is used, returns None to allow module-scoped backends.
    Uses IBM Granite 4 Micro as a small, fast model suitable for all vLLM tests.
    """
    # Check if process isolation is enabled
    use_isolation = (
        request.config.getoption("--isolate-heavy", default=False)
        or os.environ.get("CICD", "0") == "1"
    )

    if use_isolation:
        logger = FancyLogger.get_logger()
        logger.info(
            "Process isolation enabled (--isolate-heavy). "
            "Skipping shared vLLM backend - each module will create its own."
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
    cleanup_vllm_backend(backend)


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
        "needs_gpu_cleanup": True,
    },
    "vllm": {
        "marker": "vllm",
        "description": "vLLM backend tests (GPU, shared backend)",
        "needs_gpu_cleanup": True,
    },
    "ollama": {
        "marker": "ollama",
        "description": "Ollama backend tests (local server)",
        "needs_gpu_cleanup": False,
    },
    "api": {
        "marker": "requires_api_key",
        "description": "API-based backends (OpenAI, Watsonx, Bedrock)",
        "needs_gpu_cleanup": False,
    },
}

# Execution order when --group-by-backend is used
BACKEND_GROUP_ORDER = ["huggingface", "vllm", "ollama", "api"]


def aggressive_gpu_cleanup(backend_name):
    """Aggressive GPU cleanup between backend groups.

    Args:
        backend_name: Name of the backend for logging
    """
    logger = FancyLogger.get_logger()
    logger.info(f"{backend_name} group: Starting aggressive GPU cleanup...")

    import gc
    import time

    try:
        import torch

        if not torch.cuda.is_available():
            logger.info(f"{backend_name} group: No GPU, skipping cleanup")
            return

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

        # Set expandable segments for better memory management
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # Cleanup NCCL process groups
        if torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception:
                pass

        # Multi-pass cleanup between backend groups (more aggressive than per-test)
        for _ in range(5):
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(0.5)

        logger.info(f"{backend_name} group: GPU cleanup complete")
    except ImportError:
        logger.warning(f"{backend_name} group: PyTorch not available")


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
        "--ignore-gpu-check",
        action="store_true",
        default=False,
        help="Ignore GPU requirement checks (tests may fail without GPU)",
    )
    add_option_safe(
        "--ignore-ram-check",
        action="store_true",
        default=False,
        help="Ignore RAM requirement checks (tests may fail with insufficient RAM)",
    )
    add_option_safe(
        "--ignore-ollama-check",
        action="store_true",
        default=False,
        help="Ignore Ollama availability checks (tests will fail if Ollama not running)",
    )
    add_option_safe(
        "--ignore-api-key-check",
        action="store_true",
        default=False,
        help="Ignore API key checks (tests will fail without valid API keys)",
    )
    add_option_safe(
        "--ignore-all-checks",
        action="store_true",
        default=False,
        help="Ignore all requirement checks (GPU, RAM, Ollama, API keys)",
    )
    add_option_safe(
        "--disable-default-mellea-plugins",
        action="store_true",
        default=False,
        help="Register all acceptance plugin sets for every test",
    )
    add_option_safe(
        "--isolate-heavy",
        action="store_true",
        default=False,
        help="Run heavy GPU tests in isolated subprocesses (slower, but guarantees CUDA memory release)",
    )
    add_option_safe(
        "--group-by-backend",
        action="store_true",
        default=False,
        help="Group tests by backend and run them together (reduces GPU memory fragmentation)",
    )


def pytest_configure(config):
    """Register custom markers."""
    # Backend markers
    config.addinivalue_line(
        "markers", "ollama: Tests requiring Ollama backend (local, light)"
    )
    config.addinivalue_line(
        "markers", "openai: Tests requiring OpenAI API (requires API key)"
    )
    config.addinivalue_line(
        "markers", "watsonx: Tests requiring Watsonx API (requires API key)"
    )
    config.addinivalue_line(
        "markers", "huggingface: Tests requiring HuggingFace backend (local, heavy)"
    )
    config.addinivalue_line(
        "markers", "vllm: Tests requiring vLLM backend (local, GPU required)"
    )
    config.addinivalue_line("markers", "litellm: Tests requiring LiteLLM backend")

    # Capability markers
    config.addinivalue_line(
        "markers", "requires_api_key: Tests requiring external API keys"
    )
    config.addinivalue_line("markers", "requires_gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "requires_heavy_ram: Tests requiring 16GB+ RAM")
    config.addinivalue_line(
        "markers",
        "requires_gpu_isolation: Explicitly tag tests/modules that require OS-level process isolation to clear CUDA memory.",
    )
    config.addinivalue_line("markers", "qualitative: Non-deterministic quality tests")

    # Composite markers
    config.addinivalue_line(
        "markers", "llm: Tests that make LLM calls (needs at least Ollama)"
    )

    # Plugin acceptance markers
    config.addinivalue_line(
        "markers", "plugins: Acceptance tests that register all built-in plugin sets"
    )

    # Store vLLM isolation flag in config
    config._vllm_process_isolation = False


# ============================================================================
# Heavy GPU Test Process Isolation
# ============================================================================


def _run_heavy_modules_isolated(session, heavy_modules: list[str]) -> int:
    """Run heavy RAM test modules in separate processes for GPU memory isolation.

    Streams output in real-time and parses for test failures to provide
    a clear summary at the end.

    Returns exit code (0 = all passed, 1 = any failed).
    """
    print("\n" + "=" * 70)
    print("Heavy GPU Test Process Isolation Active")
    print("=" * 70)
    print(
        f"Running {len(heavy_modules)} heavy GPU test module(s) in separate processes"
    )
    print("to ensure GPU memory is fully released between modules.\n")

    # Set environment variables for vLLM
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    all_passed = True
    failed_modules = {}  # module_path -> list of failed test names

    for i, module_path in enumerate(heavy_modules, 1):
        print(f"\n[{i}/{len(heavy_modules)}] Running: {module_path}")
        print("-" * 70)

        # Build pytest command with same options as parent session
        cmd = [sys.executable, "-m", "pytest", module_path, "-v", "--no-cov"]

        # Add markers from original command if present
        config = session.config
        markexpr = config.getoption("-m", default=None)
        if markexpr:
            cmd.extend(["-m", markexpr])

        import pathlib

        repo_root = str(pathlib.Path(__file__).parent.parent.resolve())
        env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}"

        # Stream output in real-time while capturing for parsing
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,  # Line buffered for immediate output
        )

        failed_tests = []

        # Stream output line by line
        if process.stdout:
            for line in process.stdout:
                print(line, end="")  # Print immediately (streaming)

                # Parse for failures (pytest format: "test_file.py::test_name FAILED")
                if " FAILED " in line:
                    # Extract test name from pytest output
                    try:
                        parts = line.split(" FAILED ")
                        if len(parts) >= 2:
                            # Get the test identifier (the part before " FAILED ")
                            # Strip whitespace and take last token (handles indentation)
                            test_name = parts[0].strip().split()[-1]
                            failed_tests.append(test_name)
                    except Exception:
                        # If parsing fails, continue - we'll still show module failed
                        pass

        process.wait()

        if process.returncode != 0:
            all_passed = False
            failed_modules[module_path] = failed_tests
            print(f"✗ Module failed: {module_path}")
        else:
            print(f"✓ Module passed: {module_path}")

    print("\n" + "=" * 70)
    if all_passed:
        print("All heavy GPU modules passed!")
    else:
        print(f"Failed modules ({len(failed_modules)}):")
        for module, tests in failed_modules.items():
            print(f"  {module}:")
            if tests:
                for test in tests:
                    print(f"    - {test}")
            else:
                print("    (module failed but couldn't parse specific test names)")
    print("=" * 70 + "\n")

    return 0 if all_passed else 1


# ============================================================================
# vLLM Backend Cleanup Helper
# ============================================================================


def cleanup_vllm_backend(backend):
    """Best-effort cleanup of vLLM backend GPU memory.

    Note: CUDA driver holds GPU memory at process level. Only process exit
    reliably releases it. Cross-module isolation uses separate subprocesses
    (see pytest_collection_finish hook).

    Args:
        backend: The vLLM backend instance to cleanup
    """
    import gc
    import time

    import torch

    backend._underlying_model.shutdown()
    del backend._underlying_model
    del backend
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

        # Cleanup NCCL process groups to suppress warnings
        if torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception:
                # Ignore if already destroyed
                pass

        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(1)


def pytest_collection_finish(session):
    """
    Opt-in process isolation for heavy GPU tests.
    Prevents CUDA OOMs by forcing OS-level memory release between heavy modules.
    """
    # 1. Test Discovery Guard: Never isolate during discovery
    if session.config.getoption("collectonly", default=False):
        return

    # 2. Opt-in Guard: Only isolate if explicitly requested or in CI
    use_isolation = (
        session.config.getoption("--isolate-heavy", default=False)
        or os.environ.get("CICD", "0") == "1"
    )
    if not use_isolation:
        return

    # 3. Hardware Guard: Only applies to CUDA environments
    try:
        import torch

        if not torch.cuda.is_available():
            return
    except ImportError:
        return

    # Collect modules explicitly marked for GPU isolation
    heavy_items = [
        item
        for item in session.items
        if item.get_closest_marker("requires_gpu_isolation")
    ]

    # Extract unique module paths
    heavy_modules = list({str(item.path) for item in heavy_items})

    if len(heavy_modules) <= 1:
        return  # No isolation needed for a single module

    # Confirmation logging: Show which modules will be isolated
    print(f"\n[INFO] GPU Isolation enabled for {len(heavy_modules)} modules:")
    for module in heavy_modules:
        print(f"  - {module}")

    # Execute heavy modules in subprocesses
    exit_code = _run_heavy_modules_isolated(session, heavy_modules)

    # 4. Non-Destructive Execution: Remove heavy items, DO NOT exit.
    session.items = [
        item for item in session.items if str(item.path) not in heavy_modules
    ]

    # Propagate subprocess failures to the main pytest session
    if exit_code != 0:
        # Count actual test failures from the isolated modules
        # Note: We increment testsfailed by the number of modules that failed,
        # not the total number of modules. The _run_heavy_modules_isolated
        # function already tracks which modules failed.
        session.testsfailed += 1  # Mark that failures occurred
        session.exitstatus = exit_code


# ============================================================================
# Test Collection Filtering
# ============================================================================


def pytest_collection_modifyitems(config, items):
    """Skip tests at collection time based on markers and optionally reorder by backend.

    This prevents fixture setup errors for tests that would be skipped anyway.
    When --group-by-backend is used, reorders tests to group by backend.
    """
    capabilities = get_system_capabilities()

    # Check for override flags
    ignore_all = config.getoption("--ignore-all-checks", default=False)
    ignore_ollama = (
        config.getoption("--ignore-ollama-check", default=False) or ignore_all
    )

    skip_ollama = pytest.mark.skip(
        reason="Ollama not available (port 11434 not listening)"
    )

    for item in items:
        # Skip ollama tests if ollama not available
        if item.get_closest_marker("ollama") and not ignore_ollama:
            if not capabilities["has_ollama"]:
                item.add_marker(skip_ollama)

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
    - pytest --ignore-gpu-check
    - pytest --ignore-ram-check
    - pytest --ignore-ollama-check
    - pytest --ignore-api-key-check

    Also handles aggressive GPU cleanup between backend groups when
    --group-by-backend is enabled.
    """
    capabilities = get_system_capabilities()
    gh_run = int(os.environ.get("CICD", 0))
    config = item.config

    # Handle backend group cleanup when --group-by-backend is used
    if config.getoption("--group-by-backend", default=False):
        # Track the current backend group across test runs
        if not hasattr(pytest_runtest_setup, "_last_backend_group"):
            pytest_runtest_setup._last_backend_group = None

        # Determine which backend group this test belongs to
        current_group = None
        for group_name in BACKEND_GROUP_ORDER:
            marker = BACKEND_GROUPS[group_name]["marker"]
            if item.get_closest_marker(marker):
                current_group = group_name
                break

        # If we're switching to a new backend group, do aggressive cleanup
        if (
            current_group != pytest_runtest_setup._last_backend_group
            and pytest_runtest_setup._last_backend_group is not None
        ):
            prev_group = pytest_runtest_setup._last_backend_group
            if BACKEND_GROUPS[prev_group]["needs_gpu_cleanup"]:
                aggressive_gpu_cleanup(prev_group)

        pytest_runtest_setup._last_backend_group = current_group

    # Check for override flags from CLI
    ignore_all = config.getoption("--ignore-all-checks", default=False)
    ignore_gpu = config.getoption("--ignore-gpu-check", default=False) or ignore_all
    ignore_ram = config.getoption("--ignore-ram-check", default=False) or ignore_all
    ignore_api_key = (
        config.getoption("--ignore-api-key-check", default=False) or ignore_all
    )

    # Skip qualitative tests in CI
    if item.get_closest_marker("qualitative") and gh_run == 1:
        pytest.skip(
            reason="Skipping qualitative test: got env variable CICD == 1. Used only in gh workflows."
        )

    # Skip tests requiring API keys if not available (unless override)
    if item.get_closest_marker("requires_api_key") and not ignore_api_key:
        # Check specific backend markers
        for backend in ["openai", "watsonx"]:
            if item.get_closest_marker(backend):
                if not capabilities["has_api_keys"].get(backend):
                    pytest.skip(
                        f"Skipping test: {backend} API key not found in environment"
                    )

    # Skip tests requiring GPU if not available (unless override)
    if item.get_closest_marker("requires_gpu") and not ignore_gpu:
        if not capabilities["has_gpu"]:
            pytest.skip("Skipping test: GPU not available")

    # Skip tests requiring heavy RAM if insufficient (unless override)
    # NOTE: The 48GB threshold is based on empirical testing:
    #   - HuggingFace tests with granite-3.3-8b-instruct failed on 32GB M1 MacBook
    #   - Also failed on 36GB system
    #   - Set to 48GB as safe threshold for 8B model + overhead
    # TODO: Consider per-model thresholds or make configurable
    #       Can be overridden with: pytest --ignore-ram-check
    if item.get_closest_marker("requires_heavy_ram") and not ignore_ram:
        RAM_THRESHOLD_GB = 48  # Based on real-world testing
        if capabilities["ram_gb"] > 0 and capabilities["ram_gb"] < RAM_THRESHOLD_GB:
            pytest.skip(
                f"Skipping test: Insufficient RAM ({capabilities['ram_gb']:.1f}GB < {RAM_THRESHOLD_GB}GB)"
            )

    # Backend-specific skipping
    # Leaving OpenAI commented since our current OpenAI tests don't require OpenAI apikeys.
    # if item.get_closest_marker("openai") and not ignore_api_key:
    #     if not capabilities["has_api_keys"].get("openai"):
    #         pytest.skip("Skipping test: OPENAI_API_KEY not found in environment")

    if item.get_closest_marker("watsonx") and not ignore_api_key:
        if not capabilities["has_api_keys"].get("watsonx"):
            pytest.skip(
                "Skipping test: Watsonx API credentials not found in environment"
            )

    if item.get_closest_marker("vllm") and not ignore_gpu:
        if not capabilities["has_gpu"]:
            pytest.skip("Skipping test: vLLM requires GPU")

    # Note: Ollama tests are now skipped at collection time in pytest_collection_modifyitems
    # to prevent fixture setup errors


def memory_cleaner():
    """Aggressive memory cleanup function."""
    yield
    # Only run aggressive cleanup in CI where memory is constrained
    if int(os.environ.get("CICD", 0)) != 1:
        return

    # Cleanup after module
    gc.collect()
    gc.collect()
    gc.collect()

    # If torch is available, clear CUDA cache
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
