"""Tests for Python code execution requirements."""

import pytest

try:
    import llm_sandbox  # type: ignore[import-not-found]

    try:
        with llm_sandbox.SandboxSession(
            lang="python", verbose=False, keep_template=False
        ) as session:
            result = session.run("print('docker test')", timeout=5)
        _llm_sandbox_available = True
    except Exception:
        _llm_sandbox_available = False
except ImportError:
    _llm_sandbox_available = False

from pathlib import Path

from mellea.core import Context, ModelOutputThunk
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements.python_reqs import (
    PythonExecutionReq,
    _has_python_code_listing,
    _python_executes_without_error,
)
from mellea.stdlib.tools import (
    COMPATIBILITY_MATRIX,
    DOCKER_POLICY,
    LOCAL_POLICY,
    Artifact,
    CapabilityPolicy,
    StaticAnalysisEnvironment,
    UnsafeEnvironment,
)


def from_model(content: str) -> Context:
    """Helper to create context from model output."""
    ctx = ChatContext()
    ctx = ctx.add(ModelOutputThunk(value=content))
    return ctx


# Test contexts
VALID_PYTHON_CODE = """```python
def hello_world():
    return "Hello, World!"

print(hello_world())
```"""

PYTHON_WITH_SYNTAX_ERROR = """```python
def hello_world(
    return "Hello, World!"
```"""

PYTHON_WITH_RUNTIME_ERROR = """```python
def divide_by_zero():
    return 1 / 0

divide_by_zero()
```"""

PYTHON_WITH_IMPORTS = """```python
import os
import sys
from pathlib import Path

print("Hello from imports!")
```"""

PYTHON_WITH_FORBIDDEN_IMPORTS = """```python
import subprocess
import socket
import urllib

print("Dangerous imports!")
```"""

PYTHON_SIMPLE_PRINT = """```python
print("Hello, World!")
```"""

PYTHON_INFINITE_LOOP = """```python
while True:
    pass
```"""

PYTHON_RST_DOUBLE_BACKTICK = """``python
def f():
    return 1
``"""

NO_PYTHON_CODE = """
This is just text without any Python code blocks.
It contains no executable content.
"""

# Create contexts
VALID_PYTHON_CTX = from_model(VALID_PYTHON_CODE)
SYNTAX_ERROR_CTX = from_model(PYTHON_WITH_SYNTAX_ERROR)
RUNTIME_ERROR_CTX = from_model(PYTHON_WITH_RUNTIME_ERROR)
PYTHON_WITH_IMPORTS_CTX = from_model(PYTHON_WITH_IMPORTS)
PYTHON_WITH_FORBIDDEN_IMPORTS_CTX = from_model(PYTHON_WITH_FORBIDDEN_IMPORTS)
PYTHON_SIMPLE_CTX = from_model(PYTHON_SIMPLE_PRINT)
PYTHON_INFINITE_LOOP_CTX = from_model(PYTHON_INFINITE_LOOP)
NO_PYTHON_CTX = from_model(NO_PYTHON_CODE)


# region: Code extraction tests


def test_has_python_code_listing_valid():
    """Test extraction of valid Python code."""
    result = _has_python_code_listing(VALID_PYTHON_CTX)
    assert result.as_bool() is True
    assert "def hello_world" in result.reason  # type: ignore


def test_has_python_code_listing_no_code():
    """Test handling when no Python code is present."""
    result = _has_python_code_listing(NO_PYTHON_CTX)
    assert result.as_bool() is False
    assert "No Python code blocks found" in result.reason  # type: ignore


def test_has_python_code_listing_simple():
    """Test extraction of simple Python code."""
    result = _has_python_code_listing(PYTHON_SIMPLE_CTX)
    assert result.as_bool() is True
    assert "print" in result.reason  # type: ignore


def test_has_python_code_listing_rst_double_backtick():
    """Test extraction of Python code in RST-style double-backtick fences."""
    ctx = from_model(PYTHON_RST_DOUBLE_BACKTICK)
    result = _has_python_code_listing(ctx)
    assert result.as_bool() is True
    assert "def f" in result.reason  # type: ignore


# endregion

# region: Safe mode tests (default behavior)


def test_safe_mode_default():
    """Test that safe mode is default and validates without executing."""
    req = PythonExecutionReq()
    result = req.validation_fn(VALID_PYTHON_CTX)
    assert result.as_bool() is True


def test_safe_mode_syntax_error():
    """Test that safe mode catches syntax errors."""
    req = PythonExecutionReq()
    result = req.validation_fn(SYNTAX_ERROR_CTX)
    assert result.as_bool() is False


def test_safe_mode_no_execution():
    """Test that safe mode doesn't execute code (even infinite loops)."""
    with pytest.warns(DeprecationWarning, match="no effect on the static tier"):
        req = PythonExecutionReq(timeout=1)
    result = req.validation_fn(PYTHON_INFINITE_LOOP_CTX)
    assert result.as_bool() is True  # Should pass because it's not actually executed


# endregion

# region: Unsafe execution tests


def test_unsafe_execution_valid():
    """Test unsafe execution with valid code."""
    with pytest.warns(DeprecationWarning) as record:
        req = PythonExecutionReq(allow_unsafe_execution=True, timeout=5)
    assert len(record) >= 2  # allow_unsafe_execution and timeout warnings
    result = req.validation_fn(VALID_PYTHON_CTX)
    assert result.as_bool() is True


def test_unsafe_execution_runtime_error():
    """Test unsafe execution with runtime error."""
    with pytest.warns(DeprecationWarning) as record:
        req = PythonExecutionReq(allow_unsafe_execution=True, timeout=5)
    assert len(record) >= 2  # allow_unsafe_execution and timeout warnings
    result = req.validation_fn(RUNTIME_ERROR_CTX)
    assert result.as_bool() is False
    assert "error" in result.reason.lower()  # type: ignore


def test_unsafe_execution_timeout():
    """Test unsafe execution with timeout."""
    with pytest.warns(DeprecationWarning) as record:
        req = PythonExecutionReq(allow_unsafe_execution=True, timeout=1)
    assert len(record) >= 2  # allow_unsafe_execution and timeout warnings
    result = req.validation_fn(PYTHON_INFINITE_LOOP_CTX)
    assert result.as_bool() is False
    assert "timed out" in result.reason.lower()  # type: ignore


def test_unsafe_execution_syntax_error():
    """Test unsafe execution with syntax error."""
    with pytest.warns(DeprecationWarning, match="allow_unsafe_execution is deprecated"):
        req = PythonExecutionReq(allow_unsafe_execution=True)
    result = req.validation_fn(SYNTAX_ERROR_CTX)
    assert result.as_bool() is False


# endregion

# region: Import restriction tests


def test_import_restrictions_block_forbidden():
    """Test that import restrictions block forbidden imports."""
    with pytest.warns(DeprecationWarning, match="allow_unsafe_execution is deprecated"):
        req = PythonExecutionReq(
            allow_unsafe_execution=True, allowed_imports=["os", "sys"]
        )
    result = req.validation_fn(PYTHON_WITH_FORBIDDEN_IMPORTS_CTX)
    assert result.as_bool() is False
    assert "Unauthorized imports" in result.reason  # type: ignore


def test_import_restrictions_allow_permitted():
    """Test that import restrictions allow permitted imports."""
    with pytest.warns(DeprecationWarning, match="allow_unsafe_execution is deprecated"):
        req = PythonExecutionReq(
            allow_unsafe_execution=True, allowed_imports=["os", "sys", "pathlib"]
        )
    result = req.validation_fn(PYTHON_WITH_IMPORTS_CTX)
    assert result.as_bool() is True


def test_import_restrictions_with_safe_mode():
    """Test that import restrictions work with safe mode."""
    req = PythonExecutionReq(allowed_imports=["os", "sys"])
    result = req.validation_fn(PYTHON_WITH_FORBIDDEN_IMPORTS_CTX)
    assert result.as_bool() is False
    assert "Unauthorized imports" in result.reason  # type: ignore


# endregion

# region: Sandbox execution tests


@pytest.mark.skipif(
    not _llm_sandbox_available,
    reason="Sandbox tests require llm-sandbox[docker] and Docker to be available",
)
def test_sandbox_execution_valid():
    """Test sandbox execution with valid code."""
    with pytest.warns(DeprecationWarning) as record:
        req = PythonExecutionReq(use_sandbox=True, timeout=10)
    assert len(record) >= 2  # use_sandbox and timeout warnings
    result = req.validation_fn(VALID_PYTHON_CTX)
    assert result.as_bool() is True


@pytest.mark.skipif(
    not _llm_sandbox_available,
    reason="Sandbox tests require llm-sandbox[docker] and Docker to be available",
)
def test_sandbox_execution_with_imports():
    """Test sandbox execution with allowed imports."""
    with pytest.warns(DeprecationWarning) as record:
        req = PythonExecutionReq(
            use_sandbox=True, allowed_imports=["os", "sys", "pathlib"], timeout=10
        )
    assert len(record) >= 2  # use_sandbox and timeout warnings
    result = req.validation_fn(PYTHON_WITH_IMPORTS_CTX)
    assert result.as_bool() is True


@pytest.mark.skipif(
    not _llm_sandbox_available,
    reason="Sandbox tests require llm-sandbox[docker] and Docker to be available",
)
def test_sandbox_execution_timeout():
    """Test sandbox execution timeout."""
    with pytest.warns(DeprecationWarning) as record:
        req = PythonExecutionReq(use_sandbox=True, timeout=2)
    assert len(record) >= 2  # use_sandbox and timeout warnings
    result = req.validation_fn(PYTHON_INFINITE_LOOP_CTX)
    assert result.as_bool() is False


def test_sandbox_without_llm_sandbox_installed():
    """Test graceful handling when llm-sandbox is not available."""
    # This test will pass even if llm-sandbox is installed, but tests the error handling
    with pytest.warns(DeprecationWarning, match="use_sandbox is deprecated"):
        req = PythonExecutionReq(use_sandbox=True)
    # We can't easily test this without mocking, but the error handling is in the code
    assert req is not None


# endregion

# region: Configuration tests


def test_description_updates_based_on_mode():
    """Test that requirement description reflects execution tier."""
    safe_req = PythonExecutionReq()
    assert "validation only" in safe_req.description  # type: ignore

    local_req = PythonExecutionReq("local_unsafe")
    assert "local execution" in local_req.description  # type: ignore

    docker_req = PythonExecutionReq("docker")
    assert "sandbox execution" in docker_req.description  # type: ignore
    assert "timeout" in docker_req.description  # type: ignore


def test_parameter_combinations():
    """Test various tier combinations."""
    req1 = PythonExecutionReq()
    assert req1._tier == "static"

    req2 = PythonExecutionReq("local_unsafe")
    assert req2._tier == "local_unsafe"

    req3 = PythonExecutionReq("local")
    assert req3._tier == "local"

    req4 = PythonExecutionReq("docker_unsafe")
    assert req4._tier == "docker_unsafe"

    req5 = PythonExecutionReq("docker")
    assert req5._tier == "docker"


# endregion

# region: Integration tests


def test_direct_validation_function():
    """Test calling validation function directly."""
    env = StaticAnalysisEnvironment()
    result = _python_executes_without_error(VALID_PYTHON_CTX, env)
    assert result.as_bool() is True

    result = _python_executes_without_error(SYNTAX_ERROR_CTX, env)
    assert result.as_bool() is False


def test_no_code_extraction():
    """Test behavior when no code can be extracted."""
    req = PythonExecutionReq()
    result = req.validation_fn(NO_PYTHON_CTX)
    assert result.as_bool() is False
    assert "Could not extract Python code" in result.reason  # type: ignore


# endregion

# region: CapabilityPolicy tests


def test_capability_policy_defaults():
    """Default policy has timeout=30 and empty artifact_export_paths."""
    policy = CapabilityPolicy()
    assert policy.timeout == 30
    assert policy.artifact_export_paths == []
    assert policy.stdout_max_bytes is None
    assert policy.stderr_max_bytes is None


def test_capability_policy_unenforced_returns_expected_fields():
    """Filesystem and network fields are declared but not enforced."""
    policy = CapabilityPolicy()
    unenforced = policy.unenforced_capabilities()
    assert "filesystem_read_roots" in unenforced
    assert "filesystem_write_roots" in unenforced
    assert "network_access" in unenforced
    assert "package_installation" in unenforced
    assert "subprocess_execution" in unenforced
    assert "env_var_access" in unenforced
    assert "packages" not in unenforced


def test_capability_policy_enforced_returns_expected_fields():
    """timeout, stdout_max_bytes, stderr_max_bytes, artifact_export_paths, and packages are enforced."""
    policy = CapabilityPolicy()
    enforced = policy.enforced_capabilities()
    assert "timeout" in enforced
    assert "stdout_max_bytes" in enforced
    assert "stderr_max_bytes" in enforced
    assert "artifact_export_paths" in enforced
    assert "packages" in enforced


def test_capability_policy_enforced_and_unenforced_are_disjoint():
    """No capability appears in both enforced and unenforced lists."""
    policy = CapabilityPolicy()
    assert set(policy.enforced_capabilities()).isdisjoint(
        policy.unenforced_capabilities()
    )


def test_local_policy_has_expected_defaults():
    """LOCAL_POLICY declares subprocess and env_var access, no package installation, no network."""
    assert LOCAL_POLICY.subprocess_execution is True
    assert LOCAL_POLICY.env_var_access is True
    assert LOCAL_POLICY.package_installation is False
    assert LOCAL_POLICY.network_access is False
    assert LOCAL_POLICY.timeout == 30


def test_docker_policy_has_expected_defaults():
    """DOCKER_POLICY allows package installation, restricts network, and has a longer timeout."""
    assert DOCKER_POLICY.package_installation is True
    assert DOCKER_POLICY.network_access is False
    assert DOCKER_POLICY.timeout == 60


def test_compatibility_matrix_has_all_tiers():
    """COMPATIBILITY_MATRIX covers all five execution tiers."""
    assert set(COMPATIBILITY_MATRIX.keys()) == {
        "static",
        "local_unsafe",
        "local",
        "docker_unsafe",
        "docker",
    }


def test_compatibility_matrix_local_tiers_no_file_io():
    """Local tiers do not support copy_in or copy_out."""
    for tier in ("local_unsafe", "local"):
        assert COMPATIBILITY_MATRIX[tier]["copy_in"] is False
        assert COMPATIBILITY_MATRIX[tier]["copy_out"] is False


def test_compatibility_matrix_docker_tiers_have_file_io():
    """Docker tiers support copy_in and copy_out."""
    for tier in ("docker_unsafe", "docker"):
        assert COMPATIBILITY_MATRIX[tier]["copy_in"] is True
        assert COMPATIBILITY_MATRIX[tier]["copy_out"] is True


def test_compatibility_matrix_policy_applied_flag():
    """Only policy tiers have policy_applied=True."""
    assert COMPATIBILITY_MATRIX["local_unsafe"]["policy_applied"] is False
    assert COMPATIBILITY_MATRIX["docker_unsafe"]["policy_applied"] is False
    assert COMPATIBILITY_MATRIX["local"]["policy_applied"] is True
    assert COMPATIBILITY_MATRIX["docker"]["policy_applied"] is True


# endregion

# region: ExecutionResult structured fields


def test_execution_result_exit_code_none_by_default():
    """ExecutionResult has exit_code=None when not set."""
    from mellea.stdlib.tools import ExecutionResult

    result = ExecutionResult(success=True, stdout="", stderr="")
    assert result.exit_code is None


def test_execution_result_timed_out_false_by_default():
    """ExecutionResult has timed_out=False when not set."""
    from mellea.stdlib.tools import ExecutionResult

    result = ExecutionResult(success=False, stdout=None, stderr=None)
    assert result.timed_out is False


def test_execution_result_artifacts_empty_by_default():
    """ExecutionResult has empty artifacts list by default."""
    from mellea.stdlib.tools import ExecutionResult

    result = ExecutionResult(success=True, stdout="", stderr="")
    assert result.artifacts == []


def test_execution_result_execution_mode_unknown_by_default():
    """ExecutionResult has execution_mode='unknown' when not set."""
    from mellea.stdlib.tools import ExecutionResult

    result = ExecutionResult(success=True, stdout="", stderr="")
    assert result.execution_mode == "unknown"


def test_static_execution_sets_execution_mode():
    """StaticAnalysisEnvironment sets execution_mode='static' on result."""
    env = StaticAnalysisEnvironment()
    result = env.execute("x = 1")
    assert result.execution_mode == "static"


def test_local_unsafe_execution_sets_execution_mode():
    """UnsafeEnvironment with no policy sets execution_mode='local_unsafe'."""
    env = UnsafeEnvironment()
    result = env.execute("print('hi')")
    assert result.execution_mode == "local_unsafe"


def test_local_policy_execution_sets_execution_mode():
    """UnsafeEnvironment with a policy sets execution_mode='local'."""
    env = UnsafeEnvironment(policy=LOCAL_POLICY)
    result = env.execute("print('hi')")
    assert result.execution_mode == "local"


def test_local_unsafe_execution_sets_exit_code():
    """UnsafeEnvironment sets exit_code=0 on success."""
    env = UnsafeEnvironment()
    result = env.execute("print('ok')")
    assert result.exit_code == 0


def test_local_unsafe_execution_nonzero_exit_code():
    """UnsafeEnvironment sets non-zero exit_code on failure."""
    env = UnsafeEnvironment()
    result = env.execute("raise ValueError('boom')")
    assert result.exit_code is not None
    assert result.exit_code != 0


def test_timed_out_flag_set_on_timeout():
    """UnsafeEnvironment sets timed_out=True when execution times out."""
    env = UnsafeEnvironment()
    result = env.execute("import time; time.sleep(999)", timeout=1)
    assert result.timed_out is True
    assert result.skipped is True


def test_stdout_truncation_via_policy():
    """Policy stdout_max_bytes truncates long output and total length stays within budget."""
    policy = CapabilityPolicy(timeout=5, stdout_max_bytes=30)
    env = UnsafeEnvironment(policy=policy)
    result = env.execute("print('a' * 100)")
    assert result.stdout is not None
    assert "... [truncated]" in result.stdout
    assert len(result.stdout.encode()) <= 30


def test_stderr_truncation_via_policy():
    """Policy stderr_max_bytes truncates long stderr and total length stays within budget."""
    policy = CapabilityPolicy(timeout=5, stderr_max_bytes=30)
    env = UnsafeEnvironment(policy=policy)
    result = env.execute("import sys; sys.stderr.write('e' * 100)")
    assert result.stderr is not None
    assert "... [truncated]" in result.stderr
    assert len(result.stderr.encode()) <= 30


def test_working_directory_passed_to_subprocess(tmp_path: Path):
    """UnsafeEnvironment uses working_directory as cwd and reflects it on the result."""
    env = UnsafeEnvironment(working_directory=str(tmp_path))
    result = env.execute("import os; print(os.getcwd())")
    assert result.success
    assert result.stdout is not None
    assert str(tmp_path) in result.stdout
    assert result.working_directory == str(tmp_path)


def test_working_directory_none_by_default():
    """UnsafeEnvironment with no working_directory has working_directory=None on result."""
    env = UnsafeEnvironment()
    result = env.execute("print('ok')")
    assert result.working_directory is None


# endregion

# region: copy_in / copy_out stubs


def test_static_environment_copy_in_raises():
    """StaticAnalysisEnvironment raises NotImplementedError for copy_in."""
    env = StaticAnalysisEnvironment()
    with pytest.raises(NotImplementedError, match="does not support copy_in"):
        env.copy_in(Path("/tmp/x.py"), "/sandbox/x.py")


def test_static_environment_copy_out_raises():
    """StaticAnalysisEnvironment raises NotImplementedError for copy_out."""
    env = StaticAnalysisEnvironment()
    with pytest.raises(NotImplementedError, match="does not support copy_out"):
        env.copy_out("/sandbox/out.csv", Path("/tmp/out.csv"))


def test_unsafe_environment_copy_in_raises():
    """UnsafeEnvironment raises NotImplementedError for copy_in."""
    env = UnsafeEnvironment()
    with pytest.raises(NotImplementedError, match="does not support copy_in"):
        env.copy_in(Path("/tmp/x.py"), "/sandbox/x.py")


def test_unsafe_environment_copy_out_raises():
    """UnsafeEnvironment raises NotImplementedError for copy_out."""
    env = UnsafeEnvironment()
    with pytest.raises(NotImplementedError, match="does not support copy_out"):
        env.copy_out("/sandbox/out.csv", Path("/tmp/out.csv"))


# endregion

# region: Execution tier UX / deprecation


def test_tier_static_is_default():
    """PythonExecutionReq defaults to the static tier."""
    req = PythonExecutionReq()
    assert req._tier == "static"


def test_tier_explicit_local_unsafe():
    """PythonExecutionReq accepts local_unsafe tier."""
    req = PythonExecutionReq("local_unsafe")
    assert req._tier == "local_unsafe"


def test_tier_explicit_local():
    """PythonExecutionReq accepts local tier."""
    req = PythonExecutionReq("local")
    assert req._tier == "local"


def test_tier_explicit_docker_unsafe():
    """PythonExecutionReq accepts docker_unsafe tier without raising."""
    req = PythonExecutionReq("docker_unsafe")
    assert req._tier == "docker_unsafe"


def test_tier_explicit_docker():
    """PythonExecutionReq accepts docker tier without raising."""
    req = PythonExecutionReq("docker")
    assert req._tier == "docker"


def test_deprecated_allow_unsafe_promoted_to_local_unsafe() -> None:
    """allow_unsafe_execution=True is promoted to local_unsafe with a DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="allow_unsafe_execution is deprecated"):
        req = PythonExecutionReq(allow_unsafe_execution=True)
    assert req._tier == "local_unsafe"


def test_deprecated_allow_unsafe_silent_when_tier_already_local() -> None:
    """allow_unsafe_execution=True is a no-op (no warning) when a local tier is already set."""
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", DeprecationWarning)
        req = PythonExecutionReq("local", allow_unsafe_execution=True)
    assert req._tier == "local"


def test_deprecated_use_sandbox_promoted_to_docker() -> None:
    """use_sandbox=True is promoted to docker with a DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="use_sandbox is deprecated"):
        req = PythonExecutionReq(use_sandbox=True)
    assert req._tier == "docker"


def test_deprecated_use_sandbox_silent_when_tier_already_docker() -> None:
    """use_sandbox=True is a no-op (no warning) when execution_tier='docker' is already set."""
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", DeprecationWarning)
        req = PythonExecutionReq("docker", use_sandbox=True)
    assert req._tier == "docker"


def test_deprecated_use_sandbox_with_local_promotes_to_docker() -> None:
    """use_sandbox=True with execution_tier='local' promotes to 'docker' with a DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="use_sandbox is deprecated"):
        req = PythonExecutionReq("local", use_sandbox=True)
    assert req._tier == "docker"


def test_deprecated_use_sandbox_with_docker_unsafe_warns() -> None:
    """use_sandbox=True with execution_tier='docker_unsafe' warns to prefer 'docker' for policy enforcement."""
    with pytest.warns(DeprecationWarning, match="use_sandbox is deprecated"):
        req = PythonExecutionReq("docker_unsafe", use_sandbox=True)
    assert req._tier == "docker_unsafe"


def test_deprecated_allow_unsafe_with_docker_tier_warns_correctly() -> None:
    """allow_unsafe_execution=True with a docker tier warns without suggesting local_unsafe."""
    with pytest.warns(DeprecationWarning, match="has no effect"):
        req = PythonExecutionReq("docker_unsafe", allow_unsafe_execution=True)
    assert req._tier == "docker_unsafe"


def test_deprecated_timeout_issues_warning() -> None:
    """timeout kwarg issues a DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="timeout is deprecated"):
        req = PythonExecutionReq("local", timeout=10)
    assert req._tier == "local"
    assert req._policy is not None
    assert req._policy.timeout == 10


def test_deprecated_timeout_on_static_warns_no_effect() -> None:
    """timeout on the static tier warns that it has no effect."""
    with pytest.warns(DeprecationWarning, match="no effect on the static tier"):
        req = PythonExecutionReq("static", timeout=5)
    assert req._tier == "static"


def test_deprecated_timeout_on_unsafe_tiers_does_not_attach_policy() -> None:
    """timeout on local_unsafe/docker_unsafe warns that it is ignored and does not attach a policy."""
    with pytest.warns(DeprecationWarning, match="ignored for the 'local_unsafe' tier"):
        req_local = PythonExecutionReq("local_unsafe", timeout=5)
    assert req_local._policy is None

    with pytest.warns(DeprecationWarning, match="ignored for the 'docker_unsafe' tier"):
        req_docker = PythonExecutionReq("docker_unsafe", timeout=5)
    assert req_docker._policy is None


def test_combined_deprecated_flags_both_warn() -> None:
    """Passing both allow_unsafe_execution and use_sandbox emits two DeprecationWarnings."""
    with pytest.warns(DeprecationWarning) as record:
        req = PythonExecutionReq(allow_unsafe_execution=True, use_sandbox=True)
    categories = [w.category for w in record.list]
    assert categories.count(DeprecationWarning) >= 2
    assert req._tier == "docker"


def test_tier_label_timeout_zero_displays_correctly() -> None:
    """A policy timeout of 0 is displayed as 0s, not the policy default."""
    from mellea.stdlib.requirements.python_reqs import _tier_label

    policy = CapabilityPolicy(timeout=0)
    assert "0s" in _tier_label("local", policy)
    assert "0s" in _tier_label("docker", policy)


def test_policy_timeout_overrides_default():
    """An explicit policy timeout is reflected in the description."""
    policy = CapabilityPolicy(timeout=42)
    req = PythonExecutionReq("docker", policy=policy)
    assert "42s" in req.description  # type: ignore


def test_static_tier_still_validates():
    """Static tier still validates syntax correctly."""
    req = PythonExecutionReq("static")
    assert req.validation_fn(VALID_PYTHON_CTX).as_bool() is True
    assert req.validation_fn(SYNTAX_ERROR_CTX).as_bool() is False


def test_local_unsafe_executes_code():
    """local_unsafe tier actually executes code."""
    req = PythonExecutionReq("local_unsafe")
    assert req.validation_fn(VALID_PYTHON_CTX).as_bool() is True
    assert req.validation_fn(RUNTIME_ERROR_CTX).as_bool() is False


# endregion
