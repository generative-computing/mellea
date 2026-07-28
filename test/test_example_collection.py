# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for docs/examples/ collection hooks.

These hooks have regressed twice (#794, #796). This test ensures:
- Support files (__init__.py, helpers.py, conftest.py) are never collected
- Real examples with markers ARE collected
- No example is collected twice (duplicate guard)
"""

import importlib.util
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE_CONFTEST_PATH = REPO_ROOT / "docs" / "examples" / "conftest.py"
# Keep this fixture capability-gated but free of skip, slow, and qualitative gates.
DIRECT_EXAMPLE = "docs/examples/tutorial/simple_email.py"
DIRECT_NODEID = f"{DIRECT_EXAMPLE}::simple_email.py"


def _load_example_conftest():
    spec = importlib.util.spec_from_file_location(
        "example_collection_conftest", EXAMPLE_CONFTEST_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


example_conftest = _load_example_conftest()


class _Config:
    def __init__(self, *enabled: str):
        self.enabled = set(enabled)

    def getoption(self, name: str, default: bool = False) -> bool:
        return True if name in self.enabled else default


def _collect_example_nodeids(
    *paths: str, capability_options: tuple[str, ...] = ("--ignore-all-checks",)
) -> list[str]:
    result = subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            *paths,
            "--collect-only",
            "-q",
            # This subprocess only discovers tests; nested coverage reports add
            # noise and can overwrite the parent run's outputs.
            "--no-cov",
            # Collection is otherwise narrowed by whatever this host provides
            # (Ollama, GPU, credentials), which would make the floor below a
            # property of the machine rather than of the hooks (#1346).
            *capability_options,
            # Pin the rootdir so node IDs stay repo-relative even when an
            # outer run exports PYTEST_ADDOPTS with its own --rootdir.
            "--rootdir=.",
            # vLLM examples are currently also skip-marked, so this is
            # defensive: if that changes, `pytest_collection_finish` can execute
            # multiple vLLM examples even under `--collect-only`. An explicit
            # `-m` replaces the configured `not slow`, so preserve that too.
            "-m",
            "not slow and not vllm",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, (
        f"example collection failed (exit {result.returncode}):\n"
        f"{result.stdout}\n{result.stderr}"
    )

    return [line for line in result.stdout.splitlines() if "::" in line]


def test_example_collection_sanity():
    """Verify example collection excludes support files and avoids duplicates."""
    collected = _collect_example_nodeids("docs/examples/")

    # Support files must never appear as collected tests
    for item in collected:
        filename = item.split("::")[0].rsplit("/", 1)[-1]
        assert filename != "__init__.py", f"__init__.py collected as test: {item}"
        assert filename != "helpers.py", f"helpers.py collected as test: {item}"
        assert filename != "conftest.py", f"conftest.py collected as test: {item}"

    # Sanity floor — we have ~79 examples today; 50 is a safe lower bound
    assert len(collected) >= 50, (
        f"Only {len(collected)} examples collected — expected at least 50. "
        "Collection hooks may be broken."
    )

    # No duplicates — each test ID should appear exactly once
    seen = set()
    for item in collected:
        assert item not in seen, f"Duplicate collection detected: {item}"
        seen.add(item)


def test_ignore_all_checks_applies_to_direct_example():
    """Verify directly specified examples receive config through the module hook."""
    expected = [DIRECT_NODEID]
    assert _collect_example_nodeids(DIRECT_EXAMPLE) == expected
    assert (
        _collect_example_nodeids(
            DIRECT_EXAMPLE, capability_options=("--ignore-ollama-check",)
        )
        == expected
    )


def test_ignore_all_avoids_detection_in_direct_hook(monkeypatch):
    """Verify the direct-file hook does not probe capabilities under the aggregate flag."""

    def fail_capability_detection():
        raise RuntimeError("capability detection should be bypassed")

    expected = object()
    monkeypatch.setattr(
        example_conftest, "get_system_capabilities", fail_capability_detection
    )
    monkeypatch.setattr(
        example_conftest.ExampleModule, "from_parent", lambda *args, **kwargs: expected
    )

    class DirectParent:
        config = _Config("--ignore-all-checks")

    assert (
        example_conftest.pytest_pycollect_makemodule(
            REPO_ROOT / DIRECT_EXAMPLE, DirectParent()
        )
        is expected
    )


def test_collection_capability_gates(monkeypatch):
    """Verify capability gates distinguish available from unavailable hosts."""
    capabilities = {
        "has_gpu": True,
        "has_ollama": True,
        "has_api_keys": {"watsonx": "set", "openai": "set"},
    }
    monkeypatch.setattr(
        example_conftest, "get_system_capabilities", lambda: capabilities
    )
    config = _Config()

    for markers in (["huggingface"], ["vllm"], ["ollama"], ["watsonx"], ["openai"]):
        assert example_conftest._should_skip_collection(markers, config) == (
            False,
            None,
        )

    capabilities.update(has_gpu=False, has_ollama=False, has_api_keys={})
    for markers, reason_fragment in (
        (["huggingface"], "GPU"),
        (["vllm"], "GPU"),
        (["ollama"], "Ollama"),
        (["watsonx"], "Watsonx"),
        (["openai"], "OpenAI"),
    ):
        should_skip, reason = example_conftest._should_skip_collection(markers, config)
        assert should_skip
        assert reason_fragment in reason


@pytest.mark.parametrize(
    ("option", "markers"),
    [
        ("--ignore-gpu-check", ["huggingface"]),
        ("skip_resource_checks", ["vllm"]),
        ("--ignore-ollama-check", ["ollama"]),
        ("--ignore-api-key-check", ["watsonx"]),
        ("--ignore-api-key-check", ["openai"]),
    ],
)
def test_collection_capability_overrides(option, markers, monkeypatch):
    """Verify individual runtime overrides also apply during collection."""
    monkeypatch.setattr(
        example_conftest,
        "get_system_capabilities",
        lambda: {"has_gpu": False, "has_ollama": False, "has_api_keys": {}},
    )

    assert example_conftest._should_skip_collection(markers, _Config(option)) == (
        False,
        None,
    )


def test_ignore_all_preserves_non_capability_gates(monkeypatch):
    """Verify the aggregate override avoids detection but preserves explicit gates."""

    def fail_capability_detection():
        raise AssertionError("capability detection should be bypassed")

    monkeypatch.setattr(
        example_conftest, "get_system_capabilities", fail_capability_detection
    )
    monkeypatch.setenv("CICD", "1")
    monkeypatch.setenv("SKIP_SLOW", "1")
    config = _Config("--ignore-all-checks")

    assert example_conftest._should_skip_collection(["ollama"], config) == (False, None)
    for markers in (["skip_always"], ["qualitative"], ["skip"], ["slow"]):
        should_skip, reason = example_conftest._should_skip_collection(markers, config)
        assert should_skip
        assert reason
