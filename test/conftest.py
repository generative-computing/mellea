import gc
import os

import pytest


@pytest.fixture(scope="session")
def gh_run() -> int:
    return int(os.environ.get("CICD", 0))  # type: ignore


def pytest_runtest_setup(item):
    """Skip qualitative tests when running in CI environment."""
    # Runs tests *not* marked with `@pytest.mark.qualitative` to run normally.
    if not item.get_closest_marker("qualitative"):
        return

    gh_run = int(os.environ.get("CICD", 0))

    if gh_run == 1:
        pytest.skip(
            reason="Skipping qualitative test: got env variable CICD == 1. Used only in gh workflows."
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


@pytest.fixture(autouse=True, scope="function")
def aggressive_cleanup():
    """Aggressive memory cleanup after each test to prevent OOM on CI runners."""
    memory_cleaner()


@pytest.fixture(autouse=True, scope="module")
def cleanup_module_fixtures():
    """Cleanup module-scoped fixtures to free memory between test modules."""
    memory_cleaner()
