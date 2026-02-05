import gc
import os
import subprocess
import sys

import pytest

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


# ============================================================================
# Pytest Marker Registration and CLI Options
# ============================================================================


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--ignore-gpu-check",
        action="store_true",
        default=False,
        help="Ignore GPU requirement checks (tests may fail without GPU)",
    )
    parser.addoption(
        "--ignore-ram-check",
        action="store_true",
        default=False,
        help="Ignore RAM requirement checks (tests may fail with insufficient RAM)",
    )
    parser.addoption(
        "--ignore-ollama-check",
        action="store_true",
        default=False,
        help="Ignore Ollama availability checks (tests will fail if Ollama not running)",
    )
    parser.addoption(
        "--ignore-api-key-check",
        action="store_true",
        default=False,
        help="Ignore API key checks (tests will fail without valid API keys)",
    )
    parser.addoption(
        "--ignore-all-checks",
        action="store_true",
        default=False,
        help="Ignore all requirement checks (GPU, RAM, Ollama, API keys)",
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
    config.addinivalue_line("markers", "qualitative: Non-deterministic quality tests")

    # Composite markers
    config.addinivalue_line(
        "markers", "llm: Tests that make LLM calls (needs at least Ollama)"
    )

    # Store vLLM isolation flag in config
    config._vllm_process_isolation = False


# ============================================================================
# vLLM Process Isolation
# ============================================================================


def _collect_vllm_modules(session) -> list[str]:
    """Collect all test modules that have vLLM tests.

    Returns list of module paths (e.g., 'test/backends/test_vllm.py').
    """
    vllm_modules = set()

    for item in session.items:
        # Check if test has vllm marker
        if item.get_closest_marker("vllm"):
            # Get the module path
            module_path = str(item.path)
            vllm_modules.add(module_path)

    return sorted(vllm_modules)


def _run_vllm_modules_isolated(session, vllm_modules: list[str]) -> int:
    """Run vLLM test modules in separate processes for GPU memory isolation.

    Returns exit code (0 = all passed, 1 = any failed).
    """
    print("\n" + "=" * 70)
    print("vLLM Process Isolation Active")
    print("=" * 70)
    print(f"Running {len(vllm_modules)} vLLM test module(s) in separate processes")
    print("to ensure GPU memory is fully released between modules.\n")

    # Set environment variables for vLLM
    env = os.environ.copy()
    env["VLLM_USE_V1"] = "0"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    all_passed = True

    for i, module_path in enumerate(vllm_modules, 1):
        print(f"\n[{i}/{len(vllm_modules)}] Running: {module_path}")
        print("-" * 70)

        # Build pytest command with same options as parent session
        cmd = [sys.executable, "-m", "pytest", module_path, "-v"]

        # Add markers from original command if present
        config = session.config
        markexpr = config.getoption("-m", default=None)
        if markexpr:
            cmd.extend(["-m", markexpr])

        # Run in subprocess
        result = subprocess.run(cmd, env=env)

        if result.returncode != 0:
            all_passed = False
            print(f"✗ Module failed: {module_path}")
        else:
            print(f"✓ Module passed: {module_path}")

    print("\n" + "=" * 70)
    if all_passed:
        print("All vLLM modules passed!")
    else:
        print("Some vLLM modules failed.")
    print("=" * 70 + "\n")

    return 0 if all_passed else 1


def pytest_collection_finish(session):
    """After collection, check if we need vLLM process isolation.

    If vLLM tests are collected and there are multiple modules,
    run them in separate processes and exit.
    """
    # Only activate if vLLM marker is explicitly requested
    config = session.config
    markexpr = config.getoption("-m", default=None)

    # Check if vllm marker is in the expression
    if not markexpr or "vllm" not in markexpr:
        return

    # Collect vLLM modules
    vllm_modules = _collect_vllm_modules(session)

    # Only use process isolation if multiple modules
    if len(vllm_modules) <= 1:
        return

    # Run modules in isolation
    exit_code = _run_vllm_modules_isolated(session, vllm_modules)

    # Clear collected items so pytest doesn't run them again
    session.items.clear()

    # Set flag to indicate we handled vLLM tests
    config._vllm_process_isolation = True

    # Exit with appropriate code
    pytest.exit("vLLM tests completed in isolated processes", returncode=exit_code)


# ============================================================================
# Test Skipping Logic
# ============================================================================


def pytest_runtest_setup(item):
    """Skip tests based on markers and system capabilities.

    Can be overridden with command-line options:
    - pytest --ignore-gpu-check
    - pytest --ignore-ram-check
    - pytest --ignore-ollama-check
    - pytest --ignore-api-key-check
    """
    capabilities = get_system_capabilities()
    gh_run = int(os.environ.get("CICD", 0))
    config = item.config

    # Check for override flags from CLI
    ignore_all = config.getoption("--ignore-all-checks", default=False)
    ignore_gpu = config.getoption("--ignore-gpu-check", default=False) or ignore_all
    ignore_ram = config.getoption("--ignore-ram-check", default=False) or ignore_all
    ignore_ollama = (
        config.getoption("--ignore-ollama-check", default=False) or ignore_all
    )
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

    if item.get_closest_marker("ollama") and not ignore_ollama:
        if not capabilities["has_ollama"]:
            pytest.skip(
                "Skipping test: Ollama not available (port 11434 not listening)"
            )


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


@pytest.fixture(autouse=True, scope="module")
def cleanup_module_fixtures():
    """Cleanup module-scoped fixtures to free memory between test modules."""
    memory_cleaner()
