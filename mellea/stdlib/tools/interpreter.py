"""Code interpreter tool and execution environments for agentic workflows.

Provides `ExecutionResult` (capturing stdout, stderr, exit code, artifacts, and
optional static analysis output) and three concrete `ExecutionEnvironment`
implementations:

- `StaticAnalysisEnvironment` — parse and import-check only, no execution.
- `UnsafeEnvironment` — subprocess execution in the current Python environment.
- `LLMSandboxEnvironment` — Docker-isolated execution via `llm-sandbox`, with
  `copy_in` / `copy_out` support via `docker cp`.

Use `make_execution_environment` to select an environment by tier name
(`"local_unsafe"`, `"local"`, `"docker_unsafe"`, `"docker"`) rather than
constructing classes directly.  The top-level `code_interpreter` and
`local_code_interpreter` functions are ready to be wrapped as `MelleaTool`
instances for ReACT or other agentic loops.
"""

from __future__ import annotations

import ast
import mimetypes
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import warnings
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...backends.tools import MelleaTool

from ...core import MelleaLogger
from .execution_policy import (
    DOCKER_POLICY,
    LOCAL_POLICY,
    Artifact,
    CapabilityPolicy,
    ExecutionTier,
)

logger = MelleaLogger.get_logger()

# Resolved once at import time so every _install_packages() call avoids a PATH scan.
_UV: str | None = shutil.which("uv")

_TRUNCATION_MARKER = "... [truncated]"


def _truncate(text: str | None, max_bytes: int | None) -> str | None:
    """Truncate text to at most max_bytes encoded bytes, appending a marker.

    Args:
        text (str | None): The text to truncate.
        max_bytes (int | None): Maximum byte length.  `None` disables truncation.

    Returns:
        str | None: Original text, truncated text with marker, or `None`.
    """
    if text is None or max_bytes is None:
        return text
    encoded = text.encode()
    if len(encoded) <= max_bytes:
        return text
    marker = _TRUNCATION_MARKER.encode()
    keep = max(0, max_bytes - len(marker))
    if keep == 0:
        return encoded[:max_bytes].decode(errors="ignore")
    return encoded[:keep].decode(errors="ignore") + _TRUNCATION_MARKER


@dataclass
class ExecutionResult:
    """Result of code execution.

    Code execution can be aborted prior to spinning up an interpreter (e.g., if
    prohibited imports are used).  In these cases, `success` is `False` and
    `skipped` is `True`.

    If code is executed, `success` is `True` iff the exit code is 0, and
    `stdout` / `stderr` are non-`None`.

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
        exit_code (int | None): Raw process exit code, or `None` if not
            available (skipped or static analysis).
        timed_out (bool): `True` when execution was killed due to timeout.
        artifacts (list[Artifact]): Files exported from the execution environment
            after execution.
        execution_mode (str): Tier name used for this execution
            (`"local_unsafe"`, `"local"`, `"docker_unsafe"`, `"docker"`,
            `"static"`, or `"unknown"`).
        working_directory (str | None): The working directory used for execution,
            or `None` if the default was used or not applicable.
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
            that generated code may import.  `None` disables the import check.
        policy (CapabilityPolicy | None): Capability policy for this environment.
            `None` means no policy is applied (unsafe tiers).
        working_directory (str | None): Directory to use as cwd during execution.
            `None` means use the process default.  Only honoured by environments
            that spawn subprocesses (`UnsafeEnvironment`); ignored otherwise.
    """

    def __init__(
        self,
        allowed_imports: list[str] | None = None,
        policy: CapabilityPolicy | None = None,
        working_directory: str | None = None,
    ):
        """Initialize with an optional import allowlist, capability policy, and working directory."""
        self.allowed_imports = allowed_imports
        self.policy = policy
        self.working_directory = working_directory

    @abstractmethod
    def execute(self, code: str, timeout: int | None = None) -> ExecutionResult:
        """Execute the given code and return the result.

        Args:
            code (str): The Python source code to execute.
            timeout (int | None): Maximum seconds to allow the code to run.
                When `None`, the environment's policy timeout is used, or a
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

    No container isolation.  Use `policy` to declare (but not enforce)
    capabilities; `timeout` and stdout/stderr truncation from `policy`
    are actively enforced.

    Args:
        allowed_imports (list[str] | None): Allowlist of top-level module names
            that generated code may import.  `None` disables the import check.
        policy (CapabilityPolicy | None): Capability policy for this environment.
            `None` means no policy is applied.
        working_directory (str | None): Directory to use as cwd during execution.
            `None` means use the process default.
        installed_packages (set[str] | None): Shared set to persist the install
            cache across multiple `execute()` calls.  `None` creates a fresh set.
        failed_packages (set[str] | None): Shared set of package names whose
            installation has already failed.  Packages in this set are skipped on
            subsequent calls; clear the set to allow a retry.  `None` creates a
            fresh set.
        tier (str | None): Tier name reported in `ExecutionResult.execution_mode`.
            `None` infers the tier from policy presence (`"local"` when a policy
            is set, `"local_unsafe"` otherwise).  Prefer passing an explicit value
            rather than relying on inference; `make_execution_environment` always
            supplies one.
    """

    def __init__(
        self,
        allowed_imports: list[str] | None = None,
        policy: CapabilityPolicy | None = None,
        working_directory: str | None = None,
        installed_packages: set[str] | None = None,
        failed_packages: set[str] | None = None,
        tier: str | None = None,
    ) -> None:
        """Initialize the unsafe subprocess environment with optional install caches and tier override."""
        super().__init__(
            allowed_imports=allowed_imports,
            policy=policy,
            working_directory=working_directory,
        )
        self._installed_packages: set[str] = (
            installed_packages if installed_packages is not None else set()
        )
        self._failed_packages: set[str] = (
            failed_packages if failed_packages is not None else set()
        )
        self._tier: str | None = tier

    def execute(self, code: str, timeout: int | None = None) -> ExecutionResult:
        """Execute code with subprocess after checking imports.

        Args:
            code (str): The Python source code to execute.
            timeout (int | None): Maximum seconds before the subprocess is killed.
                Falls back to `policy.timeout` if set, then to 30 s.

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

        if self.policy and self.policy.packages:
            pending = [
                p
                for p in self.policy.packages
                if p not in self._installed_packages and p not in self._failed_packages
            ]
            if pending:
                if _install_packages(pending):
                    self._installed_packages.update(pending)
                else:
                    self._failed_packages.update(pending)
                    return ExecutionResult(
                        success=False,
                        stdout=None,
                        stderr=None,
                        skipped=True,
                        skip_message=f"Package installation failed: {', '.join(pending)}",
                        execution_mode=self._mode(),
                    )

        resolved_timeout = self._resolve_timeout(timeout)
        result = self._execute_subprocess(code, resolved_timeout)

        if self.working_directory and result.success:
            result.artifacts = _scan_artifacts(Path(self.working_directory))

        return result

    def _mode(self) -> str:
        if self._tier is not None:
            return self._tier
        return "local" if self.policy is not None else "local_unsafe"

    def _execute_subprocess(self, code: str, timeout: int) -> ExecutionResult:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_file = f.name
        os.chmod(
            temp_file, 0o600
        )  # defense-in-depth: NamedTemporaryFile is 0o600 on most POSIX, but not guaranteed

        try:
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.working_directory,
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
                working_directory=self.working_directory,
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
                working_directory=self.working_directory,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message=f"Exception encountered in Mellea process (*not* the code interpreter process) when trying to run code_interpreter: {e!s}",
                execution_mode=self._mode(),
                working_directory=self.working_directory,
            )
        finally:
            try:
                Path(temp_file).unlink()
            except Exception:
                pass


class LLMSandboxEnvironment(ExecutionEnvironment):
    """Docker-isolated execution environment via `llm-sandbox`.

    Supports `copy_in` and `copy_out` via `docker cp`.  Both methods require
    the environment to be used as a context manager so that a single container
    session persists across calls.

    When used without a context manager, `execute` opens and closes a fresh
    container per call (one-shot mode), which is sufficient when file I/O is
    not needed.

    Args:
        allowed_imports (list[str] | None): Allowlist of importable top-level
            modules.  `None` allows any import.
        policy (CapabilityPolicy | None): Capability policy.  `None` means
            no policy is applied (`docker_unsafe` tier).
        working_directory (str | None): Ignored for Docker tiers; present for
            interface compatibility with `ExecutionEnvironment`.
        installed_packages (set[str] | None): Shared set to persist the install
            cache across multiple `execute()` calls.  `None` creates a fresh set.
        failed_packages (set[str] | None): Shared set of package names whose
            installation has already failed.  Packages in this set are skipped on
            subsequent calls; clear the set to allow a retry.  `None` creates a
            fresh set.
        tier (str | None): Tier name reported in `ExecutionResult.execution_mode`.
            `None` infers the tier from policy presence (`"docker"` when a policy
            is set, `"docker_unsafe"` otherwise).  Prefer passing an explicit value;
            `make_execution_environment` always supplies one.
    """

    def __init__(
        self,
        allowed_imports: list[str] | None = None,
        policy: CapabilityPolicy | None = None,
        working_directory: str | None = None,
        installed_packages: set[str] | None = None,
        failed_packages: set[str] | None = None,
        tier: str | None = None,
    ):
        """Initialize the Docker sandbox environment with optional install caches and tier override."""
        super().__init__(
            allowed_imports=allowed_imports,
            policy=policy,
            working_directory=working_directory,
        )
        self._session: Any = None  # SandboxSession when open via context manager
        self._container_id: str | None = None  # set after session opens
        self._installed_packages: set[str] = (
            installed_packages if installed_packages is not None else set()
        )
        self._failed_packages: set[str] = (
            failed_packages if failed_packages is not None else set()
        )
        self._tier: str | None = tier

    def _mode(self) -> str:
        if self._tier is not None:
            return self._tier
        return "docker" if self.policy is not None else "docker_unsafe"

    def __enter__(self) -> LLMSandboxEnvironment:
        """Open the Docker session for use across multiple calls."""
        try:
            from llm_sandbox import SandboxSession
        except ImportError:
            raise ImportError(
                "llm-sandbox not installed. Install with: uv add 'mellea[sandbox]'"
            )
        self._session = SandboxSession(
            lang="python", verbose=False, keep_template=False
        )
        self._session.open()
        _cid = getattr(self._session, "container_id", None)
        if _cid is None:
            _fallback = getattr(self._session, "container", None)
            _cid = (
                getattr(_fallback, "short_id", None) if _fallback is not None else None
            )
        self._container_id = _cid
        return self

    def __exit__(self, *_: object) -> None:
        """Close the Docker session."""
        if self._session is not None:
            try:
                self._session.close()
            except Exception as e:
                logger.warning("Failed to close sandbox session: %s", e)
            self._session = None
            self._container_id = None

    def copy_in(self, host_path: Path, container_path: str) -> None:
        """Copy a file from the host into the running Docker container via `docker cp`.

        Args:
            host_path (Path): Absolute path on the host filesystem.
            container_path (str): Destination path inside the container.

        Raises:
            RuntimeError: If the environment is not open as a context manager.
            RuntimeError: If the container ID cannot be determined.
            subprocess.CalledProcessError: If `docker cp` fails.
        """
        container_id = self._require_container_id()
        subprocess.run(
            ["docker", "cp", str(host_path), f"{container_id}:{container_path}"],
            check=True,
            capture_output=True,
        )

    def copy_out(self, container_path: str, host_path: Path) -> None:
        """Copy a file from the running Docker container to the host via `docker cp`.

        Args:
            container_path (str): Source path inside the container.
            host_path (Path): Destination path on the host filesystem.

        Raises:
            RuntimeError: If the environment is not open as a context manager.
            RuntimeError: If the container ID cannot be determined.
            subprocess.CalledProcessError: If `docker cp` fails.
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

    def _install_packages_in_session(self, session: Any) -> bool:
        """Install any pending packages inside the container via pip.

        Only installs packages not already in `_installed_packages`.  Updates
        the cache on success so repeated `execute()` calls don't reinstall.

        Args:
            session: An open `SandboxSession` instance to run pip inside.

        Returns:
            bool: `True` if all pending packages installed successfully (or there
            were none to install), `False` if installation failed.
        """
        if not (self.policy and self.policy.packages):
            return True
        pending = [
            p
            for p in self.policy.packages
            if p not in self._installed_packages and p not in self._failed_packages
        ]
        if not pending:
            return True
        # session.run() executes Python code, so invoke pip via subprocess inside
        # the container rather than as a bare shell command.  Use repr() to
        # embed each package specifier safely — this prevents code injection via
        # crafted package names containing quotes or other Python metacharacters.
        pkg_repr_list = repr(
            pending
        )  # e.g. ['numpy', 'pandas'] → "['numpy', 'pandas']"
        pip_code = (
            f"import subprocess, sys\n"
            f"r = subprocess.run([sys.executable, '-m', 'pip', 'install', *{pkg_repr_list}],"
            f" capture_output=True)\n"
            f"print(r.stdout.decode(errors='replace'))\n"
            f"raise SystemExit(r.returncode)\n"
        )
        try:
            result = session.run(pip_code)
            if result.exit_code == 0:
                self._installed_packages.update(pending)
                return True
            else:
                self._failed_packages.update(pending)
                logger.warning(
                    "pip install failed inside container for %s: %s",
                    pending,
                    result.stderr.strip() if result.stderr else "",
                )
                return False
        except Exception as e:
            self._failed_packages.update(pending)
            logger.warning(
                "Unexpected error installing packages %s in container: %s", pending, e
            )
            return False

    @staticmethod
    def _cleanup_export_dirs(dirs: list[Path]) -> None:
        for d in dirs:
            shutil.rmtree(d, True)

    def execute(self, code: str, timeout: int | None = None) -> ExecutionResult:
        """Execute code in a Docker container.

        When used as a context manager, reuses the open session.  Otherwise opens
        a fresh container, runs the code, and closes it immediately.

        Args:
            code (str): The Python source code to execute.
            timeout (int | None): Maximum seconds to allow the sandboxed process to
                run.  Falls back to `policy.timeout` if set, then to 60 s.

        Returns:
            ExecutionResult: Execution outcome with stdout/stderr and success flag,
            or a skipped result on import violation or sandbox error.
        """
        if self._session is None and self.policy and self.policy.artifact_export_paths:
            warnings.warn(
                "artifact_export_paths is set but this LLMSandboxEnvironment will only "
                "export artifacts when used as a context manager ('with env:'). "
                "In one-shot mode (env.execute(...)) artifacts are not exported.",
                RuntimeWarning,
                stacklevel=2,
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
                    execution_mode=self._mode(),
                )

        resolved_timeout = self._resolve_timeout(timeout, default=DOCKER_POLICY.timeout)

        try:
            from llm_sandbox import SandboxSession
        except ImportError:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message="llm-sandbox not installed. Install with: uv add 'mellea[sandbox]'",
                execution_mode=self._mode(),
            )

        try:
            if self._session is not None:
                # Persistent session: container is reused across calls, so the
                # install cache correctly reflects what is already in the container.
                if not self._install_packages_in_session(self._session):
                    return ExecutionResult(
                        success=False,
                        stdout=None,
                        stderr=None,
                        skipped=True,
                        skip_message=f"Package installation failed: {', '.join(self._failed_packages)}",
                        execution_mode=self._mode(),
                    )
                result = self._session.run(code, timeout=resolved_timeout)
            else:
                # One-shot mode: a fresh container is created for every execute()
                # call.  We must NOT consult the shared _installed_packages cache
                # here — packages installed in a previous container are gone.  Use
                # a throwaway local cache so _install_packages_in_session only
                # skips duplicates within this single call.
                with SandboxSession(
                    lang="python", verbose=False, keep_template=False
                ) as session:
                    saved_cache = self._installed_packages
                    self._installed_packages = set()
                    install_ok = True
                    try:
                        install_ok = self._install_packages_in_session(session)
                    finally:
                        self._installed_packages = saved_cache
                    if not install_ok:
                        return ExecutionResult(
                            success=False,
                            stdout=None,
                            stderr=None,
                            skipped=True,
                            skip_message=f"Package installation failed: {', '.join(self._failed_packages)}",
                            execution_mode=self._mode(),
                        )
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
            export_dirs: list[Path] = []
            if (
                self.policy
                and self.policy.artifact_export_paths
                and self._session is not None
            ):
                for container_path in self.policy.artifact_export_paths:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        staging = Path(tmp_dir) / container_path.name
                        try:
                            self.copy_out(str(container_path), staging)
                            export_dir = Path(tempfile.mkdtemp())
                            os.chmod(export_dir, 0o700)
                            export_dirs.append(export_dir)
                            dest = export_dir / container_path.name
                            shutil.copy2(staging, dest)
                            artifacts.append(
                                Artifact(
                                    path=dest,
                                    size_bytes=dest.stat().st_size
                                    if dest.exists()
                                    else None,
                                )
                            )
                        except Exception as e:
                            logger.warning(
                                "Failed to export artifact %s: %s", container_path, e
                            )

            execution_result = ExecutionResult(
                success=result.exit_code == 0,
                stdout=stdout,
                stderr=stderr,
                exit_code=result.exit_code,
                artifacts=artifacts,
                execution_mode=self._mode(),
            )
            if export_dirs:
                weakref.finalize(
                    execution_result, self._cleanup_export_dirs, export_dirs
                )
            return execution_result
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


_LOCAL_TIERS: frozenset[str] = frozenset({"local_unsafe", "local"})


def make_execution_environment(
    tier: ExecutionTier,
    policy: CapabilityPolicy | None = None,
    allowed_imports: list[str] | None = None,
    working_directory: str | None = None,
    _install_cache: set[str] | None = None,
    _failed_cache: set[str] | None = None,
) -> ExecutionEnvironment:
    """Create an :class:`ExecutionEnvironment` for the given tier.

    The `policy` argument overrides the tier's default policy.  For unsafe
    tiers (`"local_unsafe"`, `"docker_unsafe"`) the policy defaults to
    `None` — pass an explicit policy to add declaration without changing the
    tier.

    Args:
        tier (ExecutionTier): One of `"static"`, `"local_unsafe"`, `"local"`,
            `"docker_unsafe"`, or `"docker"`.
        policy (CapabilityPolicy | None): Override the tier's default policy.
            `None` uses the tier default (`LOCAL_POLICY` / `DOCKER_POLICY`
            for policy tiers; `None` for unsafe tiers).
        allowed_imports (list[str] | None): Allowlist of importable top-level
            modules.  `None` allows any import.
        working_directory (str | None): Directory to use as cwd during execution.
            Only honoured by `UnsafeEnvironment` (local tiers); ignored for
            Docker and static tiers.
        _install_cache (set[str] | None): Shared set of already-installed package
            names.  When provided, the environment will not reinstall packages
            already present in the set, and will add newly installed packages to
            it.  Pass the same set across multiple `make_execution_environment`
            calls to avoid redundant installs within one tool lifetime.
        _failed_cache (set[str] | None): Shared set of package names whose
            installation has already failed.  Packages in this set are skipped
            on subsequent calls; clear the set to allow a retry.  Pass the same
            set as `_install_cache` to persist failure state across calls.

    Returns:
        ExecutionEnvironment: Configured environment instance.

    Raises:
        ValueError: If `tier` is not one of the recognised execution tier strings.
    """
    resolved_policy: CapabilityPolicy | None
    match tier:
        case "static":
            return StaticAnalysisEnvironment(allowed_imports=allowed_imports)
        case "local_unsafe":
            resolved_policy = policy  # None by default — no policy
            return UnsafeEnvironment(
                allowed_imports=allowed_imports,
                policy=resolved_policy,
                working_directory=working_directory,
                installed_packages=_install_cache,
                failed_packages=_failed_cache,
                tier="local_unsafe",
            )
        case "local":
            resolved_policy = policy if policy is not None else LOCAL_POLICY
            return UnsafeEnvironment(
                allowed_imports=allowed_imports,
                policy=resolved_policy,
                working_directory=working_directory,
                installed_packages=_install_cache,
                failed_packages=_failed_cache,
                tier="local",
            )
        case "docker_unsafe":
            resolved_policy = policy  # None by default — no policy
            return LLMSandboxEnvironment(
                allowed_imports=allowed_imports,
                policy=resolved_policy,
                working_directory=working_directory,
                installed_packages=_install_cache,
                failed_packages=_failed_cache,
                tier="docker_unsafe",
            )
        case "docker":
            resolved_policy = policy if policy is not None else DOCKER_POLICY
            return LLMSandboxEnvironment(
                allowed_imports=allowed_imports,
                policy=resolved_policy,
                working_directory=working_directory,
                installed_packages=_install_cache,
                failed_packages=_failed_cache,
                tier="docker",
            )
        case _:
            raise ValueError(
                f"Unknown execution tier {tier!r}. "
                "Valid tiers: 'static', 'local_unsafe', 'local', 'docker_unsafe', 'docker'."
            )


_INSTALL_TIMEOUT_SECONDS = 120


def _validate_package_names(packages: list[str]) -> None:
    """Raise ValueError for any obviously invalid package specifier.

    Rejects empty strings and flag-style arguments (leading `-`) that would
    be passed directly to pip/uv.  Does not attempt full PEP 508 validation.

    Args:
        packages (list[str]): Package specifiers to validate.

    Raises:
        ValueError: If any specifier is empty or starts with `-`.
    """
    for spec in packages:
        if not spec or not spec.strip():
            raise ValueError(
                f"Invalid package specifier {spec!r}: specifiers must be non-empty strings."
            )
        if spec.lstrip().startswith("-"):
            raise ValueError(
                f"Invalid package specifier {spec!r}: flag-style arguments (starting with '-') "
                "are not allowed.  Pass pip flags via a CapabilityPolicy or configure your "
                "package index separately."
            )


def _install_packages(packages: list[str]) -> bool:
    """Install packages before execution.

    Prefers `uv pip install` when uv is on PATH (typical in uv-managed
    venvs where `python -m pip` is unavailable); falls back to
    `sys.executable -m pip` otherwise.  Logs failures but does not abort.

    Args:
        packages (list[str]): Package specifiers to install.

    Returns:
        bool: `True` if installation succeeded, `False` if it failed.
    """
    # Pass --python so uv installs into the same interpreter the subprocess will use,
    # not whatever venv uv happens to resolve in CI or nested environments.
    cmd = (
        [_UV, "pip", "install", "--python", sys.executable, *packages]
        if _UV
        else [sys.executable, "-m", "pip", "install", *packages]
    )
    try:
        subprocess.run(
            cmd, capture_output=True, check=True, timeout=_INSTALL_TIMEOUT_SECONDS
        )
        return True
    except subprocess.TimeoutExpired:
        logger.warning(
            "Package install timed out after %ds for %s",
            _INSTALL_TIMEOUT_SECONDS,
            packages,
        )
        return False
    except subprocess.CalledProcessError as e:
        logger.warning(
            "Package install failed for %s: %s",
            packages,
            e.stderr.decode(errors="replace"),
        )
        return False
    except Exception as e:
        logger.warning("Unexpected error installing packages %s: %s", packages, e)
        return False


def _scan_artifacts(directory: Path) -> list[Artifact]:
    """Scan a directory recursively and return an Artifact for each regular file found.

    Walks the full subtree so files written to subdirectories (e.g.
    `./output/fig.png`) are included.  Entries are sorted for stable ordering.

    Args:
        directory (Path): Root directory to scan for output files.

    Returns:
        list[Artifact]: One entry per regular file, with size_bytes and inferred
        content_type.
    """
    artifacts: list[Artifact] = []
    try:
        entries = sorted(directory.rglob("*"))
    except Exception as e:
        logger.warning("Failed to scan artifact directory %s: %s", directory, e)
        return artifacts
    for path in entries:
        try:
            st = path.stat()
            if stat.S_ISREG(st.st_mode):
                content_type, _ = mimetypes.guess_type(str(path))
                artifacts.append(
                    Artifact(
                        path=path, size_bytes=st.st_size, content_type=content_type
                    )
                )
        except Exception as e:
            logger.warning("Failed to stat artifact %s: %s", path, e)
    return artifacts


_MATPLOTLIB_AGG_PREAMBLE = "import matplotlib; matplotlib.use('Agg')\n"


def _needs_matplotlib_preamble(code: str) -> bool:
    """Return True if code imports matplotlib (so Agg backend should be pre-configured)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] == "matplotlib" for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] == "matplotlib":
                return True
    return False


_DOCKER_TIERS: frozenset[str] = frozenset({"docker", "docker_unsafe"})


def python_tool(
    tier: ExecutionTier | None = None,
    packages: list[str] | None = None,
    artifact_dir: Path | None = None,
    policy: CapabilityPolicy | None = None,
    allowed_imports: list[str] | None = None,
    name: str = "python",
    suppress_agg: bool = False,
) -> MelleaTool:
    """Create a configurable Python execution tool that returns structured artifacts.

    The returned `MelleaTool` wraps a callable with signature
    `run_python(code: str) -> ExecutionResult`.  It can be passed directly to
    `ModelOption.TOOLS` in agentic ReACT loops.

    For **local tiers** (`"local_unsafe"`, `"local"`), files written to
    `artifact_dir` (or to the per-call tempdir when `artifact_dir` is `None`)
    are surfaced as `Artifact` objects on the returned `ExecutionResult`.  Only
    files produced by a **successful** execution (exit code 0) are included.

    Note:
        **Docker tiers** (`"docker"`, `"docker_unsafe"`) do not support
        artifact scanning.  `artifact_dir` is ignored for these tiers and
        `result.artifacts` is always `[]`.  A `RuntimeWarning` is emitted if
        `artifact_dir` is passed with a docker tier.

        **Local tiers are always unenforced.** ``CapabilityPolicy`` boolean
        restrictions (``network_access``, ``subprocess_execution``, etc.)
        are declarative-only on ``"local"`` and ``"local_unsafe"`` — this
        applies to both explicitly passed policies *and* the default
        ``LOCAL_POLICY`` used by ``tier="local"``.  Use a docker tier for
        real process isolation.

    When the executed code imports `matplotlib`, `matplotlib.use('Agg')` is
    injected automatically as a preamble so plots are written to files rather
    than attempting interactive display.  Pass `suppress_agg=True` to disable
    this injection (e.g. when the code sets its own backend explicitly).

    Args:
        tier (ExecutionTier | None): Execution tier — one of `"static"`,
            `"local_unsafe"`, `"local"`, `"docker_unsafe"`, or `"docker"`.
            Defaults to `None`, which resolves to `"local_unsafe"` but
            **emits a** :class:`UserWarning` **at construction time** to
            surface the implicit trust decision.  Pass
            `tier="local_unsafe"` explicitly to suppress the warning when
            unsafe local execution is intentional.
        packages (list[str] | None): Python packages to pre-install via
            `pip install` before the first execution.  Ignored for the
            `"static"` tier.  `None` or `[]` means no installs.
            Each specifier must be a non-empty string and must not begin
            with `-` (flag-style arguments are rejected); PEP 508 specifiers
            such as `pkg @ git+https://...` are accepted.  Strings are
            passed directly to pip/uv — callers are responsible for trusting
            their content as if invoking `pip install` themselves.
            Not thread-safe: the shared install/failed caches are mutated
            without a lock, so concurrent `run_python` calls on the same
            tool instance may race on first install.
        artifact_dir (Path | None): Directory where the executed code should
            write output files.  A per-call tempdir is used when `None`;
            that tempdir is kept alive as long as the returned
            `ExecutionResult` holds artifacts, and cleaned up immediately
            when no artifacts are produced.  Ignored for docker tiers.
        policy (CapabilityPolicy | None): Override the tier's default
            `CapabilityPolicy`.  When `packages` is also provided, those
            packages are merged into this policy.
        allowed_imports (list[str] | None): Allowlist of importable top-level
            modules.  `None` disables the import check.
        name (str): Tool name exposed to the model.  Defaults to `"python"`.
        suppress_agg (bool): When `True`, skip the automatic
            `matplotlib.use('Agg')` preamble injection.  Use this when the
            executed code sets its own matplotlib backend.  Defaults to
            `False`.

    Returns:
        MelleaTool: A configured tool ready for use in `ModelOption.TOOLS`.

    Raises:
        ImportError: If `MelleaTool` cannot be imported (should not happen in
            a normal mellea installation).
        ValueError: If any entry in `packages` is empty or begins with `-`.

    Example:
        ```python
        from mellea.stdlib.tools import python_tool

        tool = python_tool(tier="local_unsafe", packages=["matplotlib", "numpy"])  # explicit unsafe local execution
        result = tool.run(code="import numpy as np; print(np.sqrt(4))")
        print(result.stdout)   # "2.0"
        print(result.artifacts)  # files written during execution
        ```
    """
    from ...backends.tools import MelleaTool

    resolved_tier: ExecutionTier
    if tier is None:
        warnings.warn(
            "python_tool() was called without an explicit 'tier' argument and defaulted "
            "to 'local_unsafe', which executes model-generated code as an unrestricted "
            "subprocess with no container isolation or resource limits. "
            "Pass tier='local_unsafe' explicitly to acknowledge this trust decision, "
            "or use tier='local', tier='docker_unsafe', or tier='docker' for a more "
            "constrained environment.",
            UserWarning,
            stacklevel=2,
        )
        resolved_tier = "local_unsafe"
    else:
        resolved_tier = tier

    effective_policy: CapabilityPolicy | None = policy

    # Warn once at construction time when the caller passed a policy on a local tier
    # whose boolean restriction fields are unenforced — do this before any replace()
    # so the warning fires exactly once and points at the python_tool() call site.
    if policy is not None and resolved_tier in _LOCAL_TIERS:
        restricted = []
        if not policy.network_access:
            restricted.append("network_access=False")
        if not policy.package_installation:
            restricted.append("package_installation=False")
        if not policy.subprocess_execution:
            restricted.append("subprocess_execution=False")
        if not policy.env_var_access:
            restricted.append("env_var_access=False")
        if restricted:
            warnings.warn(
                f"A CapabilityPolicy with unenforced restrictions was passed to a "
                f"{resolved_tier!r} execution environment. The following fields are "
                "declared but will not restrict actual execution on local tiers — no "
                f"subprocess isolation is applied: {', '.join(restricted)}. "
                "These values are informational only. "
                "Use tier='docker' or tier='docker_unsafe' for real process isolation.",
                UserWarning,
                stacklevel=2,
            )

    if packages:  # None and [] both mean "no installs requested"
        _validate_package_names(packages)
        if effective_policy is None:
            effective_policy = CapabilityPolicy(
                package_installation=True, packages=list(packages)
            )
        else:
            # Merge, deduplicate (preserving order), and ensure the flag is set.
            # Suppress the __post_init__ warning that fires on replace() — the
            # user already saw it (if applicable) from the original construction.
            seen: set[str] = set()
            merged: list[str] = []
            for p in list(effective_policy.packages) + list(packages):
                if p not in seen:
                    seen.add(p)
                    merged.append(p)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                effective_policy = replace(
                    effective_policy, package_installation=True, packages=merged
                )

    # Shared caches — persist across all run_python() calls on this tool instance
    # so packages are only installed once per tool lifetime, not per call.
    # Clear _failed_cache to force a retry of previously failed installs.
    _install_cache: set[str] = set()
    _failed_cache: set[str] = set()

    if resolved_tier in _DOCKER_TIERS and artifact_dir is not None:
        warnings.warn(
            f"artifact_dir is ignored for the {resolved_tier!r} tier — "
            "LLMSandboxEnvironment does not scan the container filesystem for "
            "artifacts.  result.artifacts will always be [].  "
            "Use a local tier to surface output files as structured artifacts.",
            RuntimeWarning,
            stacklevel=2,
        )

    def run_python(code: str) -> ExecutionResult:
        """Execute Python code and return the result with any output artifacts."""
        patched_code = (
            (_MATPLOTLIB_AGG_PREAMBLE if _needs_matplotlib_preamble(code) else "")
            if not suppress_agg
            else ""
        ) + code

        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            env = make_execution_environment(
                resolved_tier,
                policy=effective_policy,
                allowed_imports=allowed_imports,
                working_directory=str(artifact_dir),
                _install_cache=_install_cache,
                _failed_cache=_failed_cache,
            )
            return env.execute(patched_code)

        # No caller-supplied artifact_dir.  Create a per-call tempdir; clean it
        # up immediately when no artifacts were produced, or attach a finalizer
        # so it is removed when the result (and therefore its artifacts) is GC'd.
        tmp_dir = Path(tempfile.mkdtemp())
        os.chmod(
            tmp_dir, 0o700
        )  # defense-in-depth: mkdtemp is already 0o700, but explicit is safer if stdlib behavior ever changes
        try:
            env = make_execution_environment(
                resolved_tier,
                policy=effective_policy,
                allowed_imports=allowed_imports,
                working_directory=str(tmp_dir),
                _install_cache=_install_cache,
                _failed_cache=_failed_cache,
            )
            result = env.execute(patched_code)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        if not result.artifacts:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            weakref.finalize(result, shutil.rmtree, tmp_dir, True)
        return result

    return MelleaTool.from_callable(run_python, name=name)


def code_interpreter(code: str) -> ExecutionResult:
    """Execute Python code in a Docker sandbox (docker_unsafe tier).

    Deprecated:
        Use `python_tool` instead:
        ```python
        from mellea.stdlib.tools import python_tool
        result = python_tool(tier="docker_unsafe").run(code=code)
        ```

    Args:
        code: The Python code to execute.

    Returns:
        An `ExecutionResult` with stdout, stderr, and a success flag.
    """
    warnings.warn(
        "code_interpreter() is deprecated and will be removed in a future release. "
        "Use python_tool(tier='docker_unsafe') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    env = make_execution_environment("docker_unsafe")
    return env.execute(code)


def local_code_interpreter(code: str) -> ExecutionResult:
    """Execute Python code in the current process environment (local_unsafe tier).

    Deprecated:
        Use `python_tool` instead:
        ```python
        from mellea.stdlib.tools import python_tool
        result = python_tool(tier="local_unsafe").run(code=code)
        ```

    Args:
        code: The Python code to execute.

    Returns:
        An `ExecutionResult` with stdout, stderr, and a success flag.
    """
    warnings.warn(
        "local_code_interpreter() is deprecated and will be removed in a future release. "
        "Use python_tool(tier='local_unsafe') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    env = make_execution_environment("local_unsafe")
    return env.execute(code, timeout=60)
