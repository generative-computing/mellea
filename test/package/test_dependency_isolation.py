"""Tests that verify each optional dependency group installs and imports correctly in isolation.

Each test uses ``uv run --isolated`` to create a throwaway environment with only the
specified extra installed, then runs a generated Python script to verify that:
1. Expected imports succeed
2. Optional imports that need other extras fail or degrade gracefully
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,  # Very slow on the first run.
    pytest.mark.timeout(600),
]

# Skip entire module if uv is not available
UV_BIN = shutil.which("uv")
if UV_BIN is None:
    pytest.skip("uv not found on PATH", allow_module_level=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IS_MACOS = sys.platform == "darwin"

# ---------------------------------------------------------------------------
# Import statements keyed by extra name ("core" for base dependencies)
# ---------------------------------------------------------------------------

# NOTE: There's some risk that we hit overlap situations (ie if a group imports another group).
#       In those cases, special handling is needed (see telemetry).
#       There's also some risk that a group could unintentionally fully import the dependencies of
#       another group (without explicitly listing it). That will lead to false positives here on
#       the should_fail side. This happens with vllm (which imports all the parts of hf) so we just
#       exclude the hf import statements from that test.
IMPORTS: dict[str, list[str]] = {
    "core": [
        "import mellea",
        "from mellea.backends import Backend, ModelIdentifier, ModelOption",
        "from mellea.backends.ollama import OllamaModelBackend",
        "from mellea.backends.openai import OpenAIBackend",
        "from mellea.core import Backend",
    ],
    "hf": ["from mellea.backends.huggingface import LocalHFBackend"],
    "vllm": ["from mellea.backends.vllm import LocalVLLMBackend"],
    "litellm": ["from mellea.backends.litellm import LiteLLMBackend"],
    "watsonx": ["from mellea.backends.watsonx import WatsonxAIBackend"],
    "tools": ["import langchain_core", "import smolagents"],
    "telemetry": [
        "from opentelemetry import trace"  # We directly import a non-Mellea class/package here since the telemetry classes are defined regardless of if telemetry is installed.
    ],
    "docling": ["from mellea.stdlib.components.docs.richdocument import RichDocument"],
    "granite_retriever": [
        "import sentence_transformers",
        "import pyarrow",
        "import elasticsearch",
    ],
    "server": ["import uvicorn", "import fastapi"],
    "sandbox": ["import llm_sandbox"],
    "hooks": [
        "import cpex"  # We directly import a non-Mellea class/package here since the hooks classes are defined regardless of if cpex is installed.
    ],
}

# Flag checks keyed by extra name: list of (module, attr, expected_with, expected_without)
# Used to check that these flags are set correctly when the dependency group is/isn't installed.
FLAG_CHECKS: dict[str, list[tuple[str, str, bool]]] = {
    "telemetry": [
        ("mellea.telemetry.tracing", "_OTEL_AVAILABLE", True),
        ("mellea.telemetry.logging", "_OTEL_AVAILABLE", True),
        ("mellea.telemetry.metrics", "_OTEL_AVAILABLE", True),
        ("mellea.plugins.base", "_HAS_PLUGIN_FRAMEWORK", True),
    ],
    "hooks": [("mellea.plugins.base", "_HAS_PLUGIN_FRAMEWORK", True)],
}

# There are a few special tests for backends.
BACKEND_EXTRAS = {"hf", "vllm", "litellm", "watsonx"}

# Meta-groups that compose other extras (no dedicated isolation test needed).
META_GROUPS = {"backends", "all"}

# Extras that have a corresponding test_<name> function in this module.
# Used to determine if we are missing tests for a given optional-dependency group.
TESTED_EXTRAS = {
    "hf",
    "vllm",
    "litellm",
    "watsonx",
    "tools",
    "telemetry",
    "docling",
    "granite_retriever",
    "server",
    "sandbox",
    "hooks",
    "backends",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_optional_dependency_groups() -> set[str]:
    """Parse ``pyproject.toml`` and return the set of optional-dependency group names.

    Returns:
        set[str]: The group names defined in ``pyproject.toml`` (e.g. ``{"hf", "vllm", ...}``).
    """
    pyproject = PROJECT_ROOT / "pyproject.toml"
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)
    return set(data.get("project", {}).get("optional-dependencies", {}).keys())


def _backend_fail_imports(exclude: set[str] | str = "") -> list[str]:
    """Return backend import statements that should fail, excluding the given extras.

    Collects import statements from all ``BACKEND_EXTRAS`` except those in
    ``exclude``, and also excludes ``vllm`` on macOS since it is unsupported there.

    Args:
        exclude (set[str] | str): Backend extras to omit (their imports are
            expected to succeed). Pass a set of extra names, a single name, or
            an empty string to include all backends.

    Returns:
        list[str]: Import statements that should raise ``ImportError`` in the
            isolated environment.
    """
    if isinstance(exclude, str):
        exclude = {exclude} if exclude else set()
    extras = BACKEND_EXTRAS - exclude
    if IS_MACOS:
        extras -= {"vllm"}
    return [stmt for name in sorted(extras) for stmt in IMPORTS[name]]


def _build_check_script(
    should_succeed: list[str] = [],
    should_fail: list[str] = [],
    flag_checks: list[tuple[str, str, bool]] = [],
) -> str:
    """Build a self-contained Python script that tests imports and prints failures.

    The generated script attempts each import statement literally (no ``exec``),
    checks module-level flags via ``importlib``, and raises ``SystemExit`` with
    a newline-delimited failure summary if anything is wrong.

    Args:
        should_succeed (list[str]): Import statements that must execute without error.
        should_fail (list[str]): Import statements that must raise ``ImportError``.
        flag_checks (list[tuple[str, str, bool]]): Tuples of
            ``(module_path, attribute_name, expected_value)`` for boolean flag
            assertions (e.g. ``("mellea.telemetry.tracing", "_OTEL_AVAILABLE", True)``).

    Returns:
        str: A complete Python script suitable for ``python -c``.
    """
    lines = ["import importlib", "failures = []", ""]

    for stmt in should_succeed:
        lines.append("try:")
        lines.append(f"    {stmt}")
        lines.append("except Exception as e:")
        lines.append(f"    failures.append(f'SHOULD SUCCEED: {stmt}: {{e}}')")
        lines.append("")

    for stmt in should_fail:
        lines.append("try:")
        lines.append(f"    {stmt}")
        lines.append(f"    failures.append('SHOULD FAIL but succeeded: {stmt}')")
        lines.append("except ImportError:")
        lines.append("    pass")
        lines.append("except Exception:")
        lines.append("    pass  # non-ImportError counts as unavailable")
        lines.append("")

    for module, attr, expected in flag_checks:
        lines.append("try:")
        lines.append(f"    _mod = importlib.import_module('{module}')")
        lines.append(f"    _val = getattr(_mod, '{attr}')")
        lines.append(f"    if _val != {expected!r}:")
        lines.append(
            f"        failures.append(f'FLAG {module}.{attr}: expected={expected!r}, got={{_val!r}}')"
        )
        lines.append("except Exception as e:")
        lines.append(f"    failures.append(f'FLAG {module}.{attr}: {{e}}')")
        lines.append("")

    lines.append("if failures:")
    lines.append("    raise SystemExit('\\n'.join(failures))")

    return "\n".join(lines)


def _run_check(
    extra: str | None = None,
    should_succeed: list[str] = [],
    should_fail: list[str] = [],
    flag_checks: list[tuple[str, str, bool]] = [],
) -> None:
    """Build and run an import check script in an isolated uv environment.

    Creates a throwaway environment via ``uv run --isolated --no-project``,
    installs mellea (with the given extra if provided), and executes the
    generated checker script. Fails the current pytest test if any check
    doesn't pass.

    Args:
        extra (str | None): The optional-dependency extra to install
            (e.g. ``"hf"``). Pass ``None`` to install core mellea only.
        should_succeed (list[str]): Import statements that must execute without error.
        should_fail (list[str]): Import statements that must raise ``ImportError``.
        flag_checks (list[tuple[str, str, bool]]): Tuples of
            ``(module_path, attribute_name, expected_value)`` for boolean flag
            assertions.

    Raises:
        ValueError: If any import statement appears in both ``should_succeed``
            and ``should_fail``.
    """
    overlap = set(should_succeed) & set(should_fail)
    if overlap:
        raise ValueError(
            f"Same imports in both should_succeed and should_fail: {overlap}"
        )

    script = _build_check_script(should_succeed, should_fail, flag_checks)
    install_spec = f".[{extra}]" if extra else "."
    assert UV_BIN is not None  # guaranteed by module-level skip
    cmd = [
        UV_BIN,
        "run",
        "--isolated",
        "--no-project",
        "--with",
        install_spec,
        "--",
        "python",
        "-c",
        script,
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=300, cwd=str(PROJECT_ROOT)
    )
    if result.returncode != 0:
        # The script prints failures to stderr via SystemExit; uv may also print to stderr
        output = (result.stdout.strip() + "\n" + result.stderr.strip()).strip()
        pytest.fail(f"Import isolation check failures:\n{output}")


def _inverted_flag_checks(extra: str) -> list[tuple[str, str, bool]]:
    """Return flag checks for the given extra with inverted expected values.

    Used in core-only testing to assert that feature flags are ``False`` when
    the extra is not installed.

    Args:
        extra (str): Key into ``FLAG_CHECKS`` (e.g. ``"telemetry"``). If the
            key is absent, an empty list is returned.

    Returns:
        list[tuple[str, str, bool]]: Flag check tuples with each expected value
            negated.
    """
    return [(mod, attr, not val) for mod, attr, val in FLAG_CHECKS.get(extra, [])]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_core_only() -> None:
    """Core mellea with no extras: basic imports work, optional backends fail."""
    _run_check(
        should_succeed=IMPORTS["core"],
        should_fail=_backend_fail_imports(exclude=""),
        flag_checks=[
            *_inverted_flag_checks("telemetry"),
            *_inverted_flag_checks("hooks"),
        ],
    )


def test_hf() -> None:
    """mellea[hf]: HuggingFace backend imports succeed, others fail."""
    _run_check(
        extra="hf",
        should_succeed=[*IMPORTS["core"], *IMPORTS["hf"]],
        should_fail=_backend_fail_imports(exclude="hf"),
    )


@pytest.mark.skipif(IS_MACOS, reason="vllm is not supported on macOS")
def test_vllm() -> None:
    """mellea[vllm]: vLLM backend imports succeed, others fail."""
    _run_check(
        extra="vllm",
        should_succeed=[*IMPORTS["core"], *IMPORTS["vllm"]],
        should_fail=_backend_fail_imports(exclude={"vllm", "hf"}),
    )


def test_litellm() -> None:
    """mellea[litellm]: LiteLLM backend imports succeed, others fail."""
    _run_check(
        extra="litellm",
        should_succeed=[*IMPORTS["core"], *IMPORTS["litellm"]],
        should_fail=_backend_fail_imports(exclude="litellm"),
    )


def test_watsonx() -> None:
    """mellea[watsonx]: Watsonx backend imports succeed, others fail."""
    _run_check(
        extra="watsonx",
        should_succeed=[*IMPORTS["core"], *IMPORTS["watsonx"]],
        should_fail=_backend_fail_imports(exclude="watsonx"),
    )


def test_tools() -> None:
    """mellea[tools]: langchain_core and smolagents are available."""
    _run_check(extra="tools", should_succeed=[*IMPORTS["core"], *IMPORTS["tools"]])


def test_telemetry() -> None:
    """mellea[telemetry]: OTEL available, plugin framework available (telemetry includes hooks)."""
    _run_check(
        extra="telemetry",
        should_succeed=[*IMPORTS["core"], *IMPORTS["telemetry"], *IMPORTS["hooks"]],
        flag_checks=[*FLAG_CHECKS["telemetry"], *FLAG_CHECKS["hooks"]],
    )


def test_docling() -> None:
    """mellea[docling]: RichDocument component is importable."""
    _run_check(extra="docling", should_succeed=[*IMPORTS["core"], *IMPORTS["docling"]])


def test_granite_retriever() -> None:
    """mellea[granite_retriever]: sentence_transformers, pyarrow, elasticsearch available."""
    _run_check(
        extra="granite_retriever",
        should_succeed=[*IMPORTS["core"], *IMPORTS["granite_retriever"]],
    )


def test_server() -> None:
    """mellea[server]: uvicorn and fastapi are available."""
    _run_check(extra="server", should_succeed=[*IMPORTS["core"], *IMPORTS["server"]])


def test_sandbox() -> None:
    """mellea[sandbox]: llm_sandbox is available."""
    _run_check(extra="sandbox", should_succeed=[*IMPORTS["core"], *IMPORTS["sandbox"]])


def test_hooks() -> None:
    """mellea[hooks]: plugin framework is available."""
    _run_check(
        extra="hooks",
        should_succeed=[*IMPORTS["core"], *IMPORTS["hooks"]],
        flag_checks=FLAG_CHECKS["hooks"],
    )


def test_backends_meta() -> None:
    """mellea[backends]: all backend imports work."""
    extras = BACKEND_EXTRAS - ({"vllm"} if IS_MACOS else set())
    should_succeed = [*IMPORTS["core"]]
    for name in sorted(extras):
        should_succeed.extend(IMPORTS[name])
    _run_check(extra="backends", should_succeed=should_succeed)


def test_core_rejects_optional_backend() -> None:
    """Core-only env correctly fails when we claim an optional backend should succeed."""
    with pytest.raises(pytest.fail.Exception, match="SHOULD SUCCEED"):
        # We expect to see "SHOULD SUCCEED" since the hf import statements fail but
        # are falsely expected to succeed here.
        _run_check(should_succeed=IMPORTS["hf"])


# ---------------------------------------------------------------------------
# Coverage guard
# ---------------------------------------------------------------------------


def test_all_extras_have_tests() -> None:
    """Every optional dependency group in pyproject.toml must have a test (except meta-groups)."""
    defined = _parse_optional_dependency_groups()
    untested = defined - TESTED_EXTRAS - META_GROUPS
    unexpected = TESTED_EXTRAS - defined
    errors = []
    if untested:
        errors.append(
            f"Optional dependency groups without isolation tests: {sorted(untested)}\n"
            "  Add a test_<name> function and include the extra in TESTED_EXTRAS."
        )
    if unexpected:
        errors.append(
            f"TESTED_EXTRAS references groups not in pyproject.toml: {sorted(unexpected)}\n"
            "  Remove them from TESTED_EXTRAS or add the group to pyproject.toml."
        )
    if errors:
        pytest.fail("\n\n".join(errors))


# ---------------------------------------------------------------------------
# Checker script self-tests (run in the current env, no uv isolation needed)
# ---------------------------------------------------------------------------


def _run_script_raw(script: str) -> subprocess.CompletedProcess:
    """Run a checker script with the current Python interpreter and return the result.

    Unlike ``_run_check``, this does **not** use ``uv run --isolated`` — it
    executes directly with ``sys.executable``, making it suitable for fast
    self-tests of the checker script logic.

    Args:
        script (str): A complete Python script (as returned by ``_build_check_script``).

    Returns:
        subprocess.CompletedProcess: The completed process with ``stdout``,
            ``stderr``, and ``returncode``.
    """
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )


def test_checker_detects_should_succeed_failure() -> None:
    """Checker script exits non-zero when a should_succeed import fails."""
    script = _build_check_script(should_succeed=["import no_such_module_xyz"])
    result = _run_script_raw(script)
    assert result.returncode != 0
    assert "SHOULD SUCCEED" in result.stderr
    assert "no_such_module_xyz" in result.stderr


def test_checker_detects_should_fail_that_succeeds() -> None:
    """Checker script exits non-zero when a should_fail import actually works."""
    script = _build_check_script(should_fail=["import json"])
    result = _run_script_raw(script)
    assert result.returncode != 0
    assert "SHOULD FAIL but succeeded" in result.stderr
    assert "import json" in result.stderr


def test_checker_detects_wrong_flag_value() -> None:
    """Checker script exits non-zero when a flag has the wrong value."""
    # sys.maxsize is an int, definitely not True
    script = _build_check_script(flag_checks=[("sys", "maxsize", True)])
    result = _run_script_raw(script)
    assert result.returncode != 0
    assert "FLAG sys.maxsize" in result.stderr


def test_checker_passes_when_all_correct() -> None:
    """Checker script exits zero when all checks pass."""
    script = _build_check_script(
        should_succeed=["import json", "from os.path import join"],
        should_fail=["import no_such_module_xyz"],
        flag_checks=[("sys", "maxsize", sys.maxsize)],  # type: ignore[list-item]
    )
    result = _run_script_raw(script)
    assert result.returncode == 0, f"Unexpected failure:\n{result.stderr}"


if __name__ == "__main__":
    pytest.main([__file__])
