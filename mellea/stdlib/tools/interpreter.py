"""Code interpreter tool and execution environments for agentic workflows.

Provides ``ExecutionResult`` (capturing stdout, stderr, exit code, artifacts, and
optional static analysis output) and three concrete ``ExecutionEnvironment``
implementations:

- ``StaticAnalysisEnvironment`` — parse and import-check only, no execution.
- ``UnsafeEnvironment`` — subprocess execution in the current Python environment.
- ``LLMSandboxEnvironment`` — Docker-isolated execution via ``llm-sandbox``, with
  ``copy_in`` / ``copy_out`` support via ``docker cp``.

Use :func:`make_execution_environment` to select an environment by tier name
(``"local_unsafe"``, ``"local"``, ``"docker_unsafe"``, ``"docker"``) rather than
constructing classes directly.  The top-level :func:`code_interpreter` and
:func:`local_code_interpreter` functions are ready to be wrapped as ``MelleaTool``
instances for ReACT or other agentic loops.
"""

import ast
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ...core import MelleaLogger
from .execution_policy import (
    DOCKER_POLICY,
    LOCAL_POLICY,
    Artifact,
    CapabilityPolicy,
    ExecutionTier,
)

if TYPE_CHECKING:
    pass

logger = MelleaLogger.get_logger()

_TRUNCATION_MARKER = "... [truncated]"


def _truncate(text: str | None, max_bytes: int | None) -> str | None:
    """Truncate text to at most max_bytes encoded bytes, appending a marker.

    Args:
        text (str | None): The text to truncate.
        max_bytes (int | None): Maximum byte length.  ``None`` disables truncation.

    Returns:
        str | None: Original text, truncated text with marker, or ``None``.
    """
    if text is None or max_bytes is None:
        return text
    encoded = text.encode()
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode(errors="ignore") + _TRUNCATION_MARKER


@dataclass
class ExecutionResult:
    """Result of code execution.

    Code execution can be aborted prior to spinning up an interpreter (e.g., if
    prohibited imports are used).  In these cases, ``success`` is ``False`` and
    ``skipped`` is ``True``.

    If code is executed, ``success`` is ``True`` iff the exit code is 0, and
    ``stdout`` / ``stderr`` are non-``None``.

    Args:
        success (bool): `True` if execution succeeded (exit code 0 or
            static-analysis passed); `False` otherwise.
        stdout (str | None): Captured standard output, or `None` if
            execution was skipped.
        stderr (str | None): Captured standard error, or `None` if
            execution was skipped.
        skipped (bool): `True` when execution was not attempted.
        skip_message (str | None): Explanation of why execution was skipped.
        analysis_result (Any | None): Optional payload from static-analysis
            environments.
        exit_code (int | None): Raw process exit code, or ``None`` if not
            available (skipped or static analysis).
        timed_out (bool): ``True`` when execution was killed due to timeout.
        artifacts (list[Artifact]): Files exported from the execution environment
            after execution.
        execution_mode (str): Tier name used for this execution
            (``"local_unsafe"``, ``"local"``, ``"docker_unsafe"``, ``"docker"``,
            ``"static"``, or ``"unknown"``).
        working_directory (str | None): Working directory inside the environment,
            or ``None`` if not applicable.
    """

    success: bool
    stdout: str | None
    stderr: str | None
    skipped: bool = False
    skip_message: str | None = None
    analysis_result: Any | None = None
    exit_code: int | None = None
    timed_out: bool = False
    artifacts: list[Artifact] = field(default_factory=list)
    execution_mode: str = "unknown"
    working_directory: str | None = None

    def to_validationresult_reason(self) -> str:
        """Map an ExecutionResult to a ValidationResult reason string.

        Returns:
            The skip message if the execution was skipped, stdout on success,
            or stderr on failure.
        """
        assert self.skip_message is not None or (
            self.stderr is not None and self.stdout is not None
        ), (
            "Every ExecutionResult should have either a skip_message or a stdout/stderr stream."
        )
        if self.skip_message:
            reason = self.skip_message
        else:
            if self.success:
                assert self.stdout is not None
                reason = self.stdout
            else:
                assert self.stderr is not None
                reason = self.stderr
        return reason


class ExecutionEnvironment(ABC):
    """Abstract environment for executing Python code.

    Args:
        allowed_imports (list[str] | None): Allowlist of top-level module names
            that generated code may import.  ``None`` disables the import check.
        policy (CapabilityPolicy | None): Capability policy for this environment.
            ``None`` means no policy is applied (unsafe tiers).
    """

    def __init__(
        self,
        allowed_imports: list[str] | None = None,
        policy: CapabilityPolicy | None = None,
    ):
        """Initialize with an optional import allowlist and capability policy."""
        self.allowed_imports = allowed_imports
        self.policy = policy

    @abstractmethod
    def execute(self, code: str, timeout: int | None = None) -> ExecutionResult:
        """Execute the given code and return the result.

        Args:
            code (str): The Python source code to execute.
            timeout (int | None): Maximum seconds to allow the code to run.
                When ``None``, the environment's policy timeout is used, or a
                built-in default if no policy is set.

        Returns:
            ExecutionResult: Execution outcome including stdout, stderr, and
            success flag.
        """

    def _resolve_timeout(self, timeout: int | None, default: int = 30) -> int:
        if timeout is not None:
            return timeout
        if self.policy is not None:
            return self.policy.timeout
        return default

    def copy_in(self, host_path: Path, container_path: str) -> None:
        """Copy a file from the host into the execution environment.

        Args:
            host_path (Path): Absolute path on the host filesystem.
            container_path (str): Destination path inside the environment.

        Raises:
            NotImplementedError: If this environment does not support file I/O.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support copy_in. "
            "Use LLMSandboxEnvironment (docker or docker_unsafe tier) for file I/O."
        )

    def copy_out(self, container_path: str, host_path: Path) -> None:
        """Copy a file from the execution environment to the host.

        Args:
            container_path (str): Source path inside the environment.
            host_path (Path): Destination path on the host filesystem.

        Raises:
            NotImplementedError: If this environment does not support file I/O.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support copy_out. "
            "Use LLMSandboxEnvironment (docker or docker_unsafe tier) for file I/O."
        )


class StaticAnalysisEnvironment(ExecutionEnvironment):
    """Safe environment that validates but does not execute code."""

    def execute(self, code: str, timeout: int | None = None) -> ExecutionResult:
        """Validate code syntax and imports without executing.

        Args:
            code (str): The Python source code to validate.
            timeout (int | None): Ignored for static analysis; present for interface
                compatibility.

        Returns:
            ExecutionResult: Result with `skipped=True` and the parsed AST in
            `analysis_result` on success, or a syntax-error description on
            failure.
        """
        try:
            parse_tree = ast.parse(code)
        except SyntaxError as e:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message="Parse failed.",
                analysis_result=e,
                execution_mode="static",
            )

        if self.allowed_imports:
            unauthorized = _get_unauthorized_imports(code, self.allowed_imports)
            if unauthorized:
                return ExecutionResult(
                    success=False,
                    stdout=None,
                    stderr=None,
                    skipped=True,
                    skip_message=f"Unauthorized imports detected: {', '.join(unauthorized)}",
                    execution_mode="static",
                )

        return ExecutionResult(
            success=True,
            stdout=None,
            stderr=None,
            skipped=True,
            skip_message="Code parses successful; the parse result is in the analysis_result field of the ExecutionResult object. The static analysis execution environment does not execute code. To execute code, use one of the other execution environments.",
            analysis_result=parse_tree,
            execution_mode="static",
        )


class UnsafeEnvironment(ExecutionEnvironment):
    """Environment that executes code directly via subprocess.

    No container isolation.  Use ``policy`` to declare (but not enforce)
    capabilities; ``timeout`` and stdout/stderr truncation from ``policy``
    are actively enforced.
    """

    def execute(self, code: str, timeout: int | None = None) -> ExecutionResult:
        """Execute code with subprocess after checking imports.

        Args:
            code (str): The Python source code to execute.
            timeout (int | None): Maximum seconds before the subprocess is killed.
                Falls back to ``policy.timeout`` if set, then to 30 s.

        Returns:
            ExecutionResult: Execution outcome with captured stdout/stderr and
            success flag, or a skipped result if imports are unauthorized.
        """
        if self.allowed_imports:
            unauthorized = _get_unauthorized_imports(code, self.allowed_imports)
            if unauthorized:
                return ExecutionResult(
                    success=False,
                    stdout=None,
                    stderr=None,
                    skipped=True,
                    skip_message=f"Unauthorized imports detected: {', '.join(unauthorized)}",
                    execution_mode=self._mode(),
                )

        resolved_timeout = self._resolve_timeout(timeout)
        return self._execute_subprocess(code, resolved_timeout)

    def _mode(self) -> str:
        return "local" if self.policy is not None else "local_unsafe"

    def _execute_subprocess(self, code: str, timeout: int) -> ExecutionResult:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            stdout = _truncate(
                result.stdout.strip(),
                self.policy.stdout_max_bytes if self.policy else None,
            )
            stderr = _truncate(
                result.stderr.strip(),
                self.policy.stderr_max_bytes if self.policy else None,
            )
            return ExecutionResult(
                success=result.returncode == 0,
                stdout=stdout,
                stderr=stderr,
                exit_code=result.returncode,
                execution_mode=self._mode(),
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message="Execution timed out.",
                timed_out=True,
                execution_mode=self._mode(),
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message=f"Exception encountered in Mellea process (*not* the code interpreter process) when trying to run code_interpreter: {e!s}",
                execution_mode=self._mode(),
            )
        finally:
            try:
                Path(temp_file).unlink()
            except Exception:
                pass


class LLMSandboxEnvironment(ExecutionEnvironment):
    """Docker-isolated execution environment via ``llm-sandbox``.

    Supports :meth:`copy_in` and :meth:`copy_out` via ``docker cp``.  Both
    methods require the environment to be used as a context manager so that a
    single container session persists across calls.

    When used without a context manager, :meth:`execute` opens and closes a
    fresh container per call (one-shot mode), which is sufficient when file I/O
    is not needed.

    Args:
        allowed_imports (list[str] | None): Allowlist of importable top-level
            modules.  ``None`` allows any import.
        policy (CapabilityPolicy | None): Capability policy.  ``None`` means
            no policy is applied (``docker_unsafe`` tier).
    """

    def __init__(
        self,
        allowed_imports: list[str] | None = None,
        policy: CapabilityPolicy | None = None,
    ):
        """Initialize with an optional import allowlist and capability policy."""
        super().__init__(allowed_imports=allowed_imports, policy=policy)
        self._session: Any = None  # SandboxSession when open via context manager
        self._container_id: str | None = None  # set after session opens

    def _mode(self) -> str:
        return "docker" if self.policy is not None else "docker_unsafe"

    def __enter__(self) -> "LLMSandboxEnvironment":
        """Open the Docker session for use across multiple calls."""
        try:
            from llm_sandbox import SandboxSession
        except ImportError:
            raise ImportError(
                "llm-sandbox not installed. Install with: pip install 'mellea[sandbox]'"
            )
        self._session = SandboxSession(
            lang="python", verbose=False, keep_template=False
        )
        self._session.open()
        self._container_id = getattr(self._session, "container_id", None) or getattr(
            self._session, "container", None
        )
        return self

    def __exit__(self, *_: object) -> None:
        """Close the Docker session."""
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None
            self._container_id = None

    def copy_in(self, host_path: Path, container_path: str) -> None:
        """Copy a file from the host into the running Docker container via ``docker cp``.

        Args:
            host_path (Path): Absolute path on the host filesystem.
            container_path (str): Destination path inside the container.

        Raises:
            RuntimeError: If the environment is not open as a context manager.
            RuntimeError: If the container ID cannot be determined.
            subprocess.CalledProcessError: If ``docker cp`` fails.
        """
        container_id = self._require_container_id()
        subprocess.run(
            ["docker", "cp", str(host_path), f"{container_id}:{container_path}"],
            check=True,
            capture_output=True,
        )

    def copy_out(self, container_path: str, host_path: Path) -> None:
        """Copy a file from the running Docker container to the host via ``docker cp``.

        Args:
            container_path (str): Source path inside the container.
            host_path (Path): Destination path on the host filesystem.

        Raises:
            RuntimeError: If the environment is not open as a context manager.
            RuntimeError: If the container ID cannot be determined.
            subprocess.CalledProcessError: If ``docker cp`` fails.
        """
        container_id = self._require_container_id()
        subprocess.run(
            ["docker", "cp", f"{container_id}:{container_path}", str(host_path)],
            check=True,
            capture_output=True,
        )

    def _require_container_id(self) -> str:
        if self._session is None:
            raise RuntimeError(
                "LLMSandboxEnvironment must be used as a context manager to call copy_in/copy_out."
            )
        if not self._container_id:
            raise RuntimeError(
                "Could not determine Docker container ID from llm-sandbox session. "
                "The llm-sandbox version may not expose container_id."
            )
        return self._container_id

    def execute(self, code: str, timeout: int | None = None) -> ExecutionResult:
        """Execute code in a Docker container.

        When used as a context manager, reuses the open session.  Otherwise opens
        a fresh container, runs the code, and closes it immediately.

        Args:
            code (str): The Python source code to execute.
            timeout (int | None): Maximum seconds to allow the sandboxed process to
                run.  Falls back to ``policy.timeout`` if set, then to 60 s.

        Returns:
            ExecutionResult: Execution outcome with stdout/stderr and success flag,
            or a skipped result on import violation or sandbox error.
        """
        if self.allowed_imports:
            unauthorized = _get_unauthorized_imports(code, self.allowed_imports)
            if unauthorized:
                return ExecutionResult(
                    success=False,
                    stdout=None,
                    stderr=None,
                    skipped=True,
                    skip_message=f"Unauthorized imports detected: {', '.join(unauthorized)}",
                    execution_mode=self._mode(),
                )

        resolved_timeout = self._resolve_timeout(timeout, default=60)

        try:
            from llm_sandbox import SandboxSession
        except ImportError:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message="llm-sandbox not installed. Install with: pip install 'mellea[sandbox]'",
                execution_mode=self._mode(),
            )

        try:
            if self._session is not None:
                result = self._session.run(code, timeout=resolved_timeout)
            else:
                with SandboxSession(
                    lang="python", verbose=False, keep_template=False
                ) as session:
                    result = session.run(code, timeout=resolved_timeout)

            stdout = _truncate(
                result.stdout.strip(),
                self.policy.stdout_max_bytes if self.policy else None,
            )
            stderr = _truncate(
                result.stderr.strip(),
                self.policy.stderr_max_bytes if self.policy else None,
            )

            artifacts: list[Artifact] = []
            if (
                self.policy
                and self.policy.artifact_export_paths
                and self._session is not None
            ):
                for container_path in self.policy.artifact_export_paths:
                    host_tmp = Path(tempfile.mkdtemp()) / container_path.name
                    try:
                        self.copy_out(str(container_path), host_tmp)
                        artifacts.append(
                            Artifact(
                                path=host_tmp,
                                size_bytes=host_tmp.stat().st_size
                                if host_tmp.exists()
                                else None,
                            )
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to export artifact %s: %s", container_path, e
                        )

            return ExecutionResult(
                success=result.exit_code == 0,
                stdout=stdout,
                stderr=stderr,
                exit_code=result.exit_code,
                artifacts=artifacts,
                execution_mode=self._mode(),
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message=f"Sandbox execution error: {e!s}",
                execution_mode=self._mode(),
            )


def _get_unauthorized_imports(code: str, allowed_imports: list[str]) -> list[str]:
    """Get list of unauthorized imports used in code."""
    unauthorized: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return unauthorized

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base_module = alias.name.split(".")[0]
                if (
                    base_module not in allowed_imports
                    and base_module not in unauthorized
                ):
                    unauthorized.append(base_module)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                base_module = node.module.split(".")[0]
                if (
                    base_module not in allowed_imports
                    and base_module not in unauthorized
                ):
                    unauthorized.append(base_module)
    return unauthorized


def _check_allowed_imports(code: str, allowed_imports: list[str]) -> bool:
    """Check if code only uses allowed imports."""
    return len(_get_unauthorized_imports(code, allowed_imports)) == 0


def make_execution_environment(
    tier: ExecutionTier,
    policy: CapabilityPolicy | None = None,
    allowed_imports: list[str] | None = None,
) -> ExecutionEnvironment:
    """Create an :class:`ExecutionEnvironment` for the given tier.

    The ``policy`` argument overrides the tier's default policy.  For unsafe
    tiers (``"local_unsafe"``, ``"docker_unsafe"``) the policy defaults to
    ``None`` — pass an explicit policy to add declaration without changing the
    tier.

    Args:
        tier (ExecutionTier): One of ``"local_unsafe"``, ``"local"``,
            ``"docker_unsafe"``, or ``"docker"``.
        policy (CapabilityPolicy | None): Override the tier's default policy.
            ``None`` uses the tier default (``LOCAL_POLICY`` / ``DOCKER_POLICY``
            for policy tiers; ``None`` for unsafe tiers).
        allowed_imports (list[str] | None): Allowlist of importable top-level
            modules.  ``None`` allows any import.

    Returns:
        ExecutionEnvironment: Configured environment instance.
    """
    resolved_policy: CapabilityPolicy | None
    match tier:
        case "local_unsafe":
            resolved_policy = policy  # None by default — no policy
            return UnsafeEnvironment(
                allowed_imports=allowed_imports, policy=resolved_policy
            )
        case "local":
            resolved_policy = policy if policy is not None else LOCAL_POLICY
            return UnsafeEnvironment(
                allowed_imports=allowed_imports, policy=resolved_policy
            )
        case "docker_unsafe":
            resolved_policy = policy  # None by default — no policy
            return LLMSandboxEnvironment(
                allowed_imports=allowed_imports, policy=resolved_policy
            )
        case "docker":
            resolved_policy = policy if policy is not None else DOCKER_POLICY
            return LLMSandboxEnvironment(
                allowed_imports=allowed_imports, policy=resolved_policy
            )
        case _:
            raise ValueError(
                f"Unknown execution tier {tier!r}. "
                "Valid tiers: 'local_unsafe', 'local', 'docker_unsafe', 'docker'."
            )


def code_interpreter(code: str) -> ExecutionResult:
    """Execute Python code in a Docker sandbox (docker_unsafe tier).

    Args:
        code: The Python code to execute.

    Returns:
        An `ExecutionResult` with stdout, stderr, and a success flag.
    """
    env = make_execution_environment("docker_unsafe")
    return env.execute(code)


def local_code_interpreter(code: str) -> ExecutionResult:
    """Execute Python code in the current process environment (local_unsafe tier).

    Args:
        code: The Python code to execute.

    Returns:
        An `ExecutionResult` with stdout, stderr, and a success flag.
    """
    env = make_execution_environment("local_unsafe")
    return env.execute(code)
