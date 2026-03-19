"""Dependency isolation tests.

Verify that:
1. Core mellea modules work with only base dependencies (no extras).
2. Each optional dependency group only imports packages from
   core dependencies + that group's declared optional dependencies.
3. Every optional-dependency group in pyproject.toml has a test mapping.

Each test spawns a fresh subprocess to get a clean sys.modules snapshot.
"""

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

# Core modules that must work with only the base dependencies declared in
# pyproject.toml [project.dependencies] — no extras installed.
#
# Notably excluded (these eagerly import optional-extra packages):
#   - mellea.backends.huggingface / vllm / litellm / watsonx (backend extras)
#   - mellea.telemetry (telemetry extra)
#   - mellea.stdlib.tools (sandbox extra — __init__ imports interpreter)
#   - mellea.stdlib.components.docs.richdocument (docling extra)
#   - mellea.formatters.granite.retrievers (granite_retriever — __init__ imports elasticsearch + numpy)
#   - mellea.plugins.hooks (hooks extra)
#   - cli.serve.app (server extra)
#   - cli.m (imports cli.serve.app)
CORE_MODULES: list[str] = [
    # Top-level package
    "mellea",
    # Core abstractions
    "mellea.core",
    # Backends (core-only — openai, ollama, bedrock, tools, adapters, etc.)
    "mellea.backends",
    "mellea.backends.backend",
    "mellea.backends.bedrock",
    "mellea.backends.cache",
    "mellea.backends.dummy",
    "mellea.backends.model_ids",
    "mellea.backends.model_options",
    "mellea.backends.ollama",
    "mellea.backends.openai",
    "mellea.backends.tools",
    "mellea.backends.utils",
    "mellea.backends.adapters",
    # Formatters (core-only — no retrievers/__init__ which pulls elasticsearch)
    "mellea.formatters",
    "mellea.formatters.chat_formatter",
    "mellea.formatters.template_formatter",
    "mellea.formatters.granite",
    # Helpers
    "mellea.helpers",
    # Plugin system (core infra, not the hooks extra)
    "mellea.plugins",
    # Standard library (core components, sessions, sampling)
    "mellea.stdlib",
    "mellea.stdlib.components",
    "mellea.stdlib.components.docs",
    "mellea.stdlib.context",
    "mellea.stdlib.functional",
    "mellea.stdlib.session",
    "mellea.stdlib.sampling",
    # CLI (excluding serve/app and m which depend on server extra)
    "cli",
    "cli.alora.commands",
    "cli.decompose",
    "cli.eval.commands",
]

# Map each pyproject optional-dependency group to the mellea modules it covers.
GROUP_MODULES: dict[str, list[str]] = {
    "hf": ["mellea.backends.huggingface"],
    "vllm": ["mellea.backends.vllm"],
    "litellm": ["mellea.backends.litellm"],
    "watsonx": ["mellea.backends.watsonx"],
    "tools": ["mellea.backends.tools"],
    "telemetry": ["mellea.telemetry"],
    "docling": ["mellea.stdlib.components.docs.richdocument"],
    "granite_retriever": ["mellea.formatters.granite.retrievers.elasticsearch"],
    "server": ["cli.serve.app"],
    "sandbox": ["mellea.stdlib.tools.interpreter"],
    "hooks": ["mellea.plugins"],
}

# Aggregate/meta groups that just combine other groups — no modules of their own.
# These don't need isolation tests; they're tested via their constituent groups.
META_GROUPS: set[str] = {"all", "backends"}

# Optional-dependency groups whose packages are imported opportunistically by
# core code via `try/except ImportError` guards.  The code works without them,
# but when they're installed they *will* appear in sys.modules.  Every test
# allows these so we don't flag guarded imports as violations.
#
# Currently only "hooks": mellea.plugins.{manager,registry,base} guard-import cpex.
GUARDED_GROUPS: list[str] = ["hooks"]

CHECKER_SCRIPT = Path(__file__).parent / "_check_dep_isolation.py"

PYPROJECT_PATH = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"

# Maximum allowed wall-clock time for `import mellea` in a fresh interpreter.
# Current baseline is ~140-565ms depending on hardware. 750ms catches
# heavy-dep regressions (torch ~2s, transformers ~800ms+) without flaking.
IMPORT_TIME_LIMIT_MS = 750


def _read_pyproject_groups() -> set[str]:
    """Return all optional-dependency group names from pyproject.toml."""
    with open(PYPROJECT_PATH, "rb") as f:
        pyproject = tomllib.load(f)
    return set(pyproject.get("project", {}).get("optional-dependencies", {}).keys())


def _run_checker(group: str, modules: list[str]) -> subprocess.CompletedProcess[str]:
    """Spawn the checker script in a fresh subprocess.

    Runs ``_check_dep_isolation.py`` with the given group and modules,
    automatically adding ``--allow-group`` flags for each entry in
    GUARDED_GROUPS (skipping the group being tested to avoid redundancy).

    Return codes from the checker:
        0 — all imports are within declared dependencies.
        1 — undeclared dependency violations found (details in stdout).
        2 — one or more target modules could not be imported (details in stderr).
    """
    allow_flags: list[str] = []
    for g in GUARDED_GROUPS:
        if g != group:  # Don't redundantly allow the group being tested
            allow_flags.extend(["--allow-group", g])
    return subprocess.run(
        [sys.executable, str(CHECKER_SCRIPT), *allow_flags, group, *modules],
        capture_output=True,
        text=True,
        timeout=120,
    )


def _find_untested_groups(
    group_modules: dict[str, list[str]], meta_groups: set[str]
) -> set[str]:
    """Return pyproject optional-dependency groups that lack a test mapping.

    Compares the groups declared in pyproject.toml against those covered by
    ``group_modules`` and ``meta_groups``, returning any that are missing.
    """
    pyproject_groups = _read_pyproject_groups()
    tested_groups = set(group_modules.keys()) | meta_groups
    return pyproject_groups - tested_groups


# ---------------------------------------------------------------------------
# Core import tests
# ---------------------------------------------------------------------------


def test_core_modules_only_use_declared_dependencies() -> None:
    """Core modules must import successfully and only use declared base dependencies."""
    result = _run_checker("core", CORE_MODULES)

    if result.returncode == 2:
        pytest.fail(
            f"Core module import failed (no extras should be needed):\n{result.stderr.strip()}"
        )

    if result.returncode != 0:
        violations = result.stdout.strip()
        pytest.fail(
            f"Core modules pull in undeclared packages "
            f"(these should be added to [project.dependencies] "
            f"or the import should be made lazy):\n{violations}"
        )


# ---------------------------------------------------------------------------
# Per-group isolation tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("group", sorted(GROUP_MODULES.keys()))
def test_dependency_isolation(group: str) -> None:
    """Each optional group should only import its declared dependencies."""
    modules = GROUP_MODULES[group]
    result = _run_checker(group, modules)

    # Import errors mean the group's deps aren't installed — skip
    if result.returncode == 2:
        pytest.skip(
            f"Could not import modules for group '{group}': {result.stderr.strip()}"
        )

    if result.returncode != 0:
        violations = result.stdout.strip()
        pytest.fail(f"Undeclared dependency imports in group '{group}':\n{violations}")


# ---------------------------------------------------------------------------
# Guard: all pyproject groups must be mapped
# ---------------------------------------------------------------------------


def test_all_groups_have_isolation_tests() -> None:
    """Every optional-dependency group in pyproject.toml must have a test mapping.

    If this test fails, a new group was added to [project.optional-dependencies]
    without a corresponding entry in GROUP_MODULES above. To fix:
    1. Add the group -> module list mapping to GROUP_MODULES.
    2. If it's a meta/aggregate group (like 'all' or 'backends'), add it to META_GROUPS instead.
    """
    untested = _find_untested_groups(GROUP_MODULES, META_GROUPS)

    if untested:
        pytest.fail(
            f"New optional-dependency group(s) in pyproject.toml missing isolation tests: "
            f"{sorted(untested)}. Add them to GROUP_MODULES in test_dependency_isolation.py "
            f"(or to META_GROUPS if they are aggregate groups)."
        )


def test_isolation_detects_undeclared_import() -> None:
    """Verify the checker flags a module imported under the wrong group.

    Imports mellea.backends.watsonx under the hf group — watsonx's dep
    (ibm-watsonx-ai) is not declared in hf and has no transitive overlap
    via extras chains, so the checker must report violations.
    """
    result = _run_checker("hf", ["mellea.backends.watsonx"])

    if result.returncode == 2:
        pytest.skip(f"watsonx extras not installed: {result.stderr.strip()}")

    assert result.returncode == 1, (
        f"Expected checker to flag undeclared deps (exit 1) but got exit {result.returncode}. "
        f"stdout: {result.stdout.strip()}"
    )
    assert "VIOLATIONS" in result.stdout


def test_guard_detects_missing_group() -> None:
    """Verify the guard actually catches unmapped groups.

    Simulates a new pyproject group by removing 'hooks' from the tested set.
    The guard logic should flag it as untested.
    """
    incomplete_modules = {k: v for k, v in GROUP_MODULES.items() if k != "hooks"}
    untested = _find_untested_groups(incomplete_modules, META_GROUPS)
    assert "hooks" in untested, (
        "Guard failed to detect that 'hooks' was missing from GROUP_MODULES"
    )


# ---------------------------------------------------------------------------
# Import time budget
# ---------------------------------------------------------------------------


# TODO: Test is marked as qualitative to prevent false regressions. Once we are confident,
#       we can have this run on nightlies / github actions.
@pytest.mark.qualitative
def test_import_mellea_time() -> None:
    """``import mellea`` must complete within the time budget.

    Uses a single fresh subprocess rather than averaging multiple runs,
    because the OS page cache warms after the first invocation and would
    make subsequent runs artificially fast. The threshold provides enough
    headroom over the baseline to absorb normal single-run variance.
    """
    timing_script = (
        "import json, time; "
        "s = time.perf_counter(); "
        "import mellea; "
        "print(json.dumps({'ms': (time.perf_counter() - s) * 1000}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", timing_script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"Import crashed: {result.stderr}"
    elapsed_ms = json.loads(result.stdout)["ms"]

    assert elapsed_ms < IMPORT_TIME_LIMIT_MS, (
        f"import mellea took {elapsed_ms:.0f}ms (limit: {IMPORT_TIME_LIMIT_MS}ms). "
        f"A heavy dependency may have been added to the import chain."
    )
