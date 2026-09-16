# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for docs/examples/ collection hooks.

These hooks have regressed twice (#794, #796). This test ensures:
- Support files (__init__.py, helpers.py, conftest.py) are never collected
- Real examples with markers ARE collected
- No example is collected twice (duplicate guard)
- Every notebook declares its requirements in its own notebook metadata (#89)
"""

import importlib.util
import json
import pathlib
import subprocess
import sys
import tomllib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE_CONFTEST_PATH = REPO_ROOT / "docs" / "examples" / "conftest.py"
NOTEBOOK_DIR = REPO_ROOT / "docs" / "examples" / "notebooks"
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
    capability_probes = []

    def record_capability_detection():
        capability_probes.append(True)
        return {
            "has_gpu": True,
            "has_ollama": True,
            "has_api_keys": {"watsonx": "set", "openai": "set"},
        }

    expected = object()
    monkeypatch.setattr(
        example_conftest, "get_system_capabilities", record_capability_detection
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
    assert capability_probes == []


def test_direct_hook_applies_capability_gate(monkeypatch):
    """Verify the direct-file hook probes and skips without an override."""
    capability_probes = []

    def record_capability_detection():
        capability_probes.append(True)
        return {"has_gpu": False, "has_ollama": False, "has_api_keys": {}}

    expected = object()
    monkeypatch.setattr(
        example_conftest, "get_system_capabilities", record_capability_detection
    )
    monkeypatch.setattr(
        example_conftest.SkippedFile, "from_parent", lambda *args, **kwargs: expected
    )
    monkeypatch.setattr(
        example_conftest.ExampleModule,
        "from_parent",
        lambda *args, **kwargs: pytest.fail("capability-gated example was collected"),
    )

    class DirectParent:
        config = _Config()

    assert (
        example_conftest.pytest_pycollect_makemodule(
            REPO_ROOT / DIRECT_EXAMPLE, DirectParent()
        )
        is expected
    )
    assert capability_probes == [True]


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
    ("option", "markers", "still_gated"),
    [
        (
            "--ignore-gpu-check",
            ["huggingface"],
            ((["ollama"], "Ollama"), (["watsonx"], "Watsonx")),
        ),
        (
            "skip_resource_checks",
            ["vllm"],
            ((["ollama"], "Ollama"), (["openai"], "OpenAI")),
        ),
        (
            "--ignore-ollama-check",
            ["ollama"],
            ((["huggingface"], "GPU"), (["watsonx"], "Watsonx")),
        ),
        (
            "--ignore-api-key-check",
            ["watsonx"],
            ((["huggingface"], "GPU"), (["ollama"], "Ollama")),
        ),
        (
            "--ignore-api-key-check",
            ["openai"],
            ((["vllm"], "GPU"), (["ollama"], "Ollama")),
        ),
    ],
)
def test_collection_capability_overrides(option, markers, still_gated, monkeypatch):
    """Verify individual runtime overrides apply only to their capability gate."""
    monkeypatch.setattr(
        example_conftest,
        "get_system_capabilities",
        lambda: {"has_gpu": False, "has_ollama": False, "has_api_keys": {}},
    )
    config = _Config(option)

    assert example_conftest._should_skip_collection(markers, config) == (False, None)
    for gated_markers, reason_fragment in still_gated:
        should_skip, reason = example_conftest._should_skip_collection(
            gated_markers, config
        )
        assert should_skip
        assert reason_fragment in reason


def test_ignore_all_preserves_non_capability_gates(monkeypatch):
    """Verify the aggregate override avoids detection but preserves explicit gates."""
    capability_probes = []

    def record_capability_detection():
        capability_probes.append(True)
        return {
            "has_gpu": True,
            "has_ollama": True,
            "has_api_keys": {"watsonx": "set", "openai": "set"},
        }

    monkeypatch.setattr(
        example_conftest, "get_system_capabilities", record_capability_detection
    )
    monkeypatch.setenv("CICD", "1")
    monkeypatch.setenv("SKIP_SLOW", "1")
    config = _Config("--ignore-all-checks")

    assert example_conftest._should_skip_collection(["ollama"], config) == (False, None)
    for markers in (["skip_always"], ["qualitative"], ["skip"], ["slow"]):
        should_skip, reason = example_conftest._should_skip_collection(markers, config)
        assert should_skip
        assert reason
    assert capability_probes == []


def _write_notebook(path: pathlib.Path, mellea_metadata: dict | None) -> pathlib.Path:
    """Write a minimal, cell-free notebook carrying `mellea_metadata` (or none)."""
    metadata = {} if mellea_metadata is None else {"mellea": mellea_metadata}
    path.write_text(
        json.dumps(
            {"cells": [], "metadata": metadata, "nbformat": 4, "nbformat_minor": 4}
        ),
        encoding="utf-8",
    )
    return path


def test_every_notebook_declares_requirements():
    """Verify every notebook on disk opts in via its own `mellea` metadata."""
    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    assert notebooks, f"no notebooks found under {NOTEBOOK_DIR}"

    missing = [
        path.name
        for path in notebooks
        if example_conftest._notebook_requirements(path) is None
    ]
    assert not missing, (
        f"notebooks with no `mellea.markers` in their notebook metadata: {missing}. "
        "Add a `mellea` block to the notebook's top-level metadata so it is tested, "
        "or delete the notebook. See docs/examples/notebooks/README.md."
    )


def test_notebook_metadata_uses_registered_markers():
    """Verify notebook-declared marker names are declared in pyproject.toml."""
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        declared_markers = {
            line.split(":", 1)[0]
            for line in tomllib.load(f)["tool"]["pytest"]["ini_options"]["markers"]
        }

    for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        entry = example_conftest._notebook_requirements(path)
        assert entry, f"{path.name} declares no markers"
        unknown = set(entry["markers"]) - declared_markers
        assert not unknown, f"{path.name} uses undeclared markers: {sorted(unknown)}"


@pytest.mark.parametrize(
    "metadata",
    [
        None,  # no `mellea` block at all
        {"packages": ["mcp"]},  # block present but no markers
        {"markers": []},  # markers present but empty
    ],
    ids=["no-block", "no-markers-key", "empty-markers"],
)
def test_notebook_without_markers_is_skipped(tmp_path, metadata):
    """Verify a notebook that does not opt in is never executed, even with checks off."""
    notebook = _write_notebook(tmp_path / "undeclared.ipynb", metadata)
    should_skip, reason = example_conftest._should_skip_notebook(
        notebook, _Config("--ignore-all-checks")
    )
    assert should_skip
    assert "mellea" in reason


def test_malformed_notebook_is_skipped(tmp_path):
    """Verify unparseable JSON is skipped rather than executed without gates."""
    notebook = tmp_path / "broken.ipynb"
    notebook.write_text("{not valid json", encoding="utf-8")
    should_skip, reason = example_conftest._should_skip_notebook(
        notebook, _Config("--ignore-all-checks")
    )
    assert should_skip
    assert "mellea" in reason


def test_notebook_skipped_when_package_missing(tmp_path):
    """Verify the notebook's `packages` requirement gates it."""
    notebook = _write_notebook(
        tmp_path / "needs_package.ipynb",
        {"markers": ["e2e", "ollama"], "packages": ["definitely_not_installed_pkg"]},
    )
    should_skip, reason = example_conftest._should_skip_notebook(
        notebook, _Config("--ignore-all-checks")
    )
    assert should_skip
    assert "definitely_not_installed_pkg" in reason


def test_notebooks_not_collected_without_nbmake():
    """Verify notebooks stay out of a plain pytest run (they need --nbmake)."""
    result = subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            "docs/examples/notebooks",
            "--collect-only",
            "-q",
            "--no-cov",
            "--rootdir=.",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
    )

    # Exit 5 == "no tests collected", which is the expected outcome here.
    assert result.returncode in (0, 5), (
        f"unexpected exit {result.returncode}:\n{result.stdout}\n{result.stderr}"
    )
    assert ".ipynb::" not in result.stdout, (
        f"notebooks collected without --nbmake:\n{result.stdout}"
    )


def _collect_notebook_names(marker_expr: str) -> set[str]:
    """Collect notebooks with --nbmake under `marker_expr`, returning bare filenames."""
    result = subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            "docs/examples/notebooks",
            # --nbmake is what makes pytest collect .ipynb at all.
            "--nbmake",
            "--collect-only",
            "-q",
            "--no-cov",
            # Otherwise the split would depend on whether this host has Ollama.
            "--ignore-all-checks",
            "--rootdir=.",
            "-m",
            marker_expr,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
    )

    # Exit 5 == "no tests collected", legitimate when a marker selects nothing.
    assert result.returncode in (0, 5), (
        f"notebook collection failed (exit {result.returncode}) for -m {marker_expr!r}:"
        f"\n{result.stdout}\n{result.stderr}"
    )

    return {
        line.split("::")[0].rsplit("/", 1)[-1]
        for line in result.stdout.splitlines()
        if ".ipynb::" in line
    }


def _expected_notebook_names(*, slow: bool) -> set[str]:
    """Notebooks that a fast (or slow) run should collect on this host.

    Package-gated notebooks drop out before markers matter, so honour that here too —
    otherwise this would assert docling is installed rather than that markers work.
    """
    expected = set()
    for path in NOTEBOOK_DIR.glob("*.ipynb"):
        entry = example_conftest._notebook_requirements(path)
        assert entry, f"{path.name} declares no markers"
        if example_conftest._missing_packages(entry.get("packages", [])):
            continue
        if ("slow" in entry["markers"]) == slow:
            expected.add(path.name)
    return expected


def test_nbmake_collection_splits_fast_and_slow():
    """Verify declared markers reach pytest's `-m` selection.

    This is the gate that keeps the heavyweight notebooks out of PR CI: they are marked
    `slow` in their own metadata, and the `-m "not slow"` in addopts deselects them.
    """
    fast = _collect_notebook_names("not slow")
    slow = _collect_notebook_names("slow")

    assert fast, "no notebooks collected for a fast run — marker wiring may be broken"
    assert fast == _expected_notebook_names(slow=False)
    assert slow == _expected_notebook_names(slow=True)
    assert not fast & slow, (
        f"notebooks in both fast and slow runs: {sorted(fast & slow)}"
    )
