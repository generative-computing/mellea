"""Reusable test predicates for resource-gated test skipping.

These return ``pytest.mark.skipif`` decorators that test authors apply directly.
Each predicate encapsulates a specific availability check so that:

- Test authors specify *exactly* what their test needs (not a vague tier).
- Skip reasons are self-documenting.
- No marker registration or conftest hook is required.

Usage::

    from test.predicates import require_gpu, require_ram, require_api_key

    @require_gpu()
    def test_cuda_basic():
        ...

    @require_gpu(min_vram_gb=24)
    def test_large_model():
        ...

    # Module-level gating (applies to all tests in the file):
    pytestmark = [pytest.mark.e2e, pytest.mark.huggingface, require_gpu(), require_ram(48)]
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys

import pytest

# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

_IS_APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"


def _apple_silicon_vram_gb() -> float:
    """Conservative usable GPU memory estimate for Apple Silicon.

    Metal's ``recommendedMaxWorkingSetSize`` is a static device property
    (~75% of total RAM) that does not account for current system load.
    We use ``min(total * 0.75, total - 16)`` to leave headroom for the OS
    and desktop applications, which typically consume 8–16 GB on a loaded
    developer machine.
    """
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=2,
        )
        total_gb = int(out.stdout.strip()) / (1024**3)
        return min(total_gb * 0.75, total_gb - 16)
    except Exception:
        return 0.0


def _gpu_available() -> bool:
    if _IS_APPLE_SILICON:
        return True
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _gpu_vram_gb() -> float:
    """Return usable GPU VRAM in GB, or 0 if unavailable.

    On Apple Silicon: uses a conservative heuristic based on total unified
    memory rather than Metal's static ``recommendedMaxWorkingSetSize``.
    On CUDA: reports device 0 total memory via torch.
    """
    if _IS_APPLE_SILICON:
        return _apple_silicon_vram_gb()
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except (ImportError, RuntimeError, AttributeError):
        pass
    return 0.0


def require_gpu(*, min_vram_gb: int | None = None):
    """Skip unless a GPU is available, optionally with minimum VRAM.

    Args:
        min_vram_gb: Minimum VRAM in GB.  When ``None``, any GPU suffices.
    """
    if not _gpu_available():
        return pytest.mark.skipif(True, reason="No GPU available (CUDA/MPS)")

    if min_vram_gb is not None:
        vram = _gpu_vram_gb()
        if vram < min_vram_gb:
            return pytest.mark.skipif(
                True,
                reason=f"Insufficient VRAM: {vram:.0f} GB < {min_vram_gb} GB required",
            )

    return pytest.mark.skipif(False, reason="")


# ---------------------------------------------------------------------------
# System RAM
# ---------------------------------------------------------------------------


def _system_ram_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        return 0.0


def require_ram(min_gb: int):
    """Skip unless the system has at least *min_gb* GB of RAM."""
    ram = _system_ram_gb()
    if ram > 0 and ram < min_gb:
        return pytest.mark.skipif(
            True, reason=f"Insufficient RAM: {ram:.0f} GB < {min_gb} GB required"
        )
    return pytest.mark.skipif(False, reason="")


# ---------------------------------------------------------------------------
# GPU process isolation
# ---------------------------------------------------------------------------


def require_gpu_isolation():
    """Skip unless GPU process isolation is enabled.

    Isolation is active when ``--isolate-heavy`` is passed or ``CICD=1``.
    Tests marked with this predicate will be run in separate subprocesses
    to prevent CUDA OOM from cross-test memory leaks.
    """
    isolate = os.environ.get("CICD", "0") == "1"
    # Note: --isolate-heavy is a pytest CLI flag checked at collection time
    # by conftest.py.  At import time we can only check the env var.
    return pytest.mark.skipif(
        not (isolate or _gpu_available()),
        reason="GPU isolation requires CICD=1 or --isolate-heavy with a GPU",
    )


# ---------------------------------------------------------------------------
# API keys / credentials
# ---------------------------------------------------------------------------


def require_api_key(*env_vars: str):
    """Skip unless all specified environment variables are set.

    Usage::

        @require_api_key("OPENAI_API_KEY")
        def test_openai_chat(): ...

        @require_api_key("WATSONX_API_KEY", "WATSONX_URL", "WATSONX_PROJECT_ID")
        def test_watsonx_generate(): ...
    """
    missing = [v for v in env_vars if not os.environ.get(v)]
    if missing:
        return pytest.mark.skipif(
            True, reason=f"Missing environment variables: {', '.join(missing)}"
        )
    return pytest.mark.skipif(False, reason="")


# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------


def require_package(package: str):
    """Skip unless *package* is importable.

    For simple cases, ``pytest.importorskip(package)`` at module level is
    equivalent and more idiomatic.  This predicate is useful when you want
    a decorator rather than a module-level call::

        @require_package("cpex.framework")
        def test_plugin_registration(): ...
    """
    try:
        __import__(package)
        available = True
    except ImportError:
        available = False

    return pytest.mark.skipif(not available, reason=f"{package} not installed")


# ---------------------------------------------------------------------------
# Python version
# ---------------------------------------------------------------------------


def require_python(min_version: tuple[int, ...]):
    """Skip unless running on at least the given Python version.

    Usage::

        @require_python((3, 11))
        async def test_asyncio_timeout(): ...
    """
    version_str = ".".join(str(v) for v in min_version)
    return pytest.mark.skipif(
        sys.version_info < min_version, reason=f"Requires Python {version_str}+"
    )
