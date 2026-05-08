"""Bash shell command execution tool and execution environments for agentic workflows.

Provides ``BashEnvironment`` (abstract base for bash execution) and three concrete
implementations: ``StaticBashEnvironment`` (parse and safety-check only, no execution),
``UnsafeBashEnvironment`` (subprocess execution in the current shell), and
``LLMSandboxBashEnvironment`` (Docker-isolated execution via ``llm-sandbox``). All
environments enforce a conservative safety denylist (sudo, rm -rf, git push --force,
system paths, interactive shells). The top-level ``bash_executor`` and
``local_bash_executor`` functions are ready to be wrapped as ``MelleaTool`` instances
for ReACT or other agentic loops.
"""

import shlex
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...core import MelleaLogger
from .interpreter import ExecutionResult

logger = MelleaLogger.get_logger()

# Commands that are always dangerous
DANGEROUS_COMMANDS = {
    # Privilege escalation
    "sudo",
    "su",
    "doas",
    # Interactive shells
    "bash",
    "sh",
    "zsh",
    "ksh",
    "tcsh",
    # User/group/password changes
    "passwd",
    "visudo",
    "chsh",
    "chfn",
    "useradd",
    "userdel",
    "usermod",
    "groupadd",
    "groupdel",
    "groupmod",
}

# System paths that are dangerous to write to
DANGEROUS_PATHS = {
    "/",
    "/bin",
    "/sbin",
    "/usr",
    "/lib",
    "/lib64",
    "/etc",
    "/sys",
    "/proc",
    "/boot",
    "/root",
    "/var/log",
    "/var/www",
}

# Dangerous flags that make commands unsafe
DANGEROUS_FLAGS = {"-rf", "-r", "--recursive", "--force", "-f", "--force-all"}

# Maximum output size (10KB per stream)
MAX_OUTPUT_SIZE = 10 * 1024


def _is_dangerous_command(argv: list[str]) -> tuple[bool, str]:
    """Check if a command is dangerous based on argv analysis.

    Args:
        argv: Tokenized command arguments.

    Returns:
        A tuple of (is_dangerous, reason_message).
    """
    if not argv:
        return False, ""

    cmd = argv[0].split("/")[-1]  # Get basename of command

    # Check for dangerous commands
    if cmd in DANGEROUS_COMMANDS:
        if cmd in ("bash", "sh", "zsh", "ksh", "tcsh"):
            if any(arg in ("-i", "--interactive", "-l", "-login") for arg in argv):
                return True, f"Interactive shell '{cmd}' is not allowed"
        else:
            return True, f"Command '{cmd}' is not allowed"

    # Check for dangerous git operations
    if cmd == "git":
        # Check for --force flag on any git operation
        if any("--force" in arg or arg == "-f" for arg in argv):
            return True, "git commands with --force or -f flag are not allowed"
        # Check for destructive git operations: push, reset --hard, clean
        has_destructive_op = False
        if "push" in argv and any("--force" in arg or arg == "-f" for arg in argv):
            has_destructive_op = True
        if "reset" in argv and "--hard" in argv:
            has_destructive_op = True
        if "clean" in argv and any(("-f" in arg or "-d" in arg) for arg in argv):
            has_destructive_op = True
        if has_destructive_op:
            return True, "Destructive git operation is not allowed"

    # Check for dangerous rm patterns
    if cmd == "rm":
        if "-r" in argv or "-rf" in argv or "--recursive" in argv:
            return True, "rm with -r or -rf flag is not allowed"

    # Check for dangerous flags in other commands
    for flag in DANGEROUS_FLAGS:
        if flag in argv:
            if cmd in ("cp", "mv", "rm", "git", "make", "apt", "yum"):
                return True, f"Command '{cmd}' with '{flag}' flag is not allowed"

    return False, ""


def _check_dangerous_paths(argv: list[str]) -> tuple[bool, str]:
    """Check if command targets dangerous filesystem paths.

    Args:
        argv: Tokenized command arguments.

    Returns:
        A tuple of (has_dangerous_paths, reason_message).
    """
    write_commands = {"rm", "touch", "cp", "mv", "mkdir", "mkfifo", "mknod", "tee"}
    if not argv or argv[0].split("/")[-1] not in write_commands:
        return False, ""

    # Scan all arguments for path-like strings
    for arg in argv[1:]:
        # Skip flags and values for flags
        if arg.startswith("-"):
            continue

        # Check for absolute paths pointing to dangerous locations
        if arg.startswith("/"):
            # For absolute paths, check directly without resolving (non-existent paths)
            for danger_path in DANGEROUS_PATHS:
                if arg == danger_path or arg.startswith(danger_path + "/"):
                    return True, f"Writing to '{arg}' is not allowed"
        else:
            # For relative paths, try to resolve
            try:
                path = Path(arg).expanduser()
                # If it starts with ~, expand and check
                if "~" in arg:
                    expanded = path.expanduser()
                    path_str = str(expanded)
                    for danger_path in DANGEROUS_PATHS:
                        if path_str == danger_path or path_str.startswith(
                            danger_path + "/"
                        ):
                            return True, f"Writing to '{path_str}' is not allowed"
            except Exception:
                # If we can't resolve, skip
                pass

    return False, ""


def _check_working_dir_restriction(
    argv: list[str], working_dir: str | None
) -> tuple[bool, str]:
    """Check if command respects working directory restriction.

    Args:
        argv: Tokenized command arguments.
        working_dir: Allowed working directory, or None for no restriction.

    Returns:
        A tuple of (violates_restriction, reason_message).
    """
    if not working_dir:
        return False, ""

    write_commands = {"rm", "touch", "cp", "mv", "mkdir", "mkfifo", "mknod", "tee"}
    if not argv or argv[0].split("/")[-1] not in write_commands:
        return False, ""

    try:
        allowed_path_str = str(Path(working_dir).expanduser().resolve())
        # Ensure the allowed path ends with / for prefix matching
        if not allowed_path_str.endswith("/"):
            allowed_path_str_prefix = allowed_path_str + "/"
        else:
            allowed_path_str_prefix = allowed_path_str

        for arg in argv[1:]:
            if arg.startswith("-"):
                continue

            # Try to resolve all paths (both absolute and relative)
            try:
                resolved_path = str(Path(arg).expanduser().resolve())
                # Check if path is allowed: in working_dir, /tmp, or /private/tmp (macOS)
                is_in_tmp = resolved_path.startswith(("/tmp", "/private/tmp"))
                is_in_working_dir = (
                    resolved_path == allowed_path_str
                    or resolved_path.startswith(allowed_path_str_prefix)
                )
                if not (is_in_tmp or is_in_working_dir):
                    return (
                        True,
                        f"Path '{arg}' is outside allowed directory '{working_dir}'",
                    )
            except Exception:
                # If we can't resolve, skip (might be a flag value)
                pass
    except Exception:
        pass

    return False, ""


def _truncate_output(output: str, max_size: int = MAX_OUTPUT_SIZE) -> tuple[str, bool]:
    """Truncate output if it exceeds max size.

    Args:
        output: The output string to potentially truncate.
        max_size: Maximum allowed size in bytes.

    Returns:
        A tuple of (truncated_output, was_truncated).
    """
    if len(output) > max_size:
        return output[:max_size] + "\n[OUTPUT TRUNCATED]", True
    return output, False


class BashEnvironment(ABC):
    """Abstract environment for executing bash commands.

    Args:
        allowed_paths (list[str] | None): Additional paths to allow writing to
            (beyond default safe paths: current dir, /tmp, home). ``None`` uses
            default safety checks only.
        working_dir (str | None): If specified, restrict file operations to this
            directory and /tmp. Useful for sandboxing agent tasks.
        timeout (int): Maximum number of seconds to allow command execution.

    """

    def __init__(
        self,
        allowed_paths: list[str] | None = None,
        working_dir: str | None = None,
        timeout: int = 60,
    ):
        """Initialize BashEnvironment with optional path allowlist and timeout."""
        self.allowed_paths = allowed_paths or []
        self.working_dir = working_dir
        self.timeout = timeout

    @abstractmethod
    def execute(self, command: str) -> ExecutionResult:
        """Execute the given bash command and return the result.

        Args:
            command (str): The bash command to execute.

        Returns:
            ExecutionResult: Execution outcome including stdout, stderr, and
            success flag.
        """


class StaticBashEnvironment(BashEnvironment):
    """Safe environment that validates but does not execute bash commands."""

    def execute(self, command: str) -> ExecutionResult:
        """Parse and validate command without executing.

        Args:
            command (str): The bash command to validate.

        Returns:
            ExecutionResult: Result with ``skipped=True`` and parsed argv in
            ``analysis_result`` on success, or a safety-check failure on rejection.
        """
        # Parse command into argv
        try:
            argv = shlex.split(command)
        except ValueError as e:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message=f"Failed to parse command: {e!s}",
            )

        if not argv:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message="Empty command",
            )

        # Check for dangerous commands
        is_dangerous, reason = _is_dangerous_command(argv)
        if is_dangerous:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message=reason,
            )

        # Check for dangerous paths
        has_dangerous, reason = _check_dangerous_paths(argv)
        if has_dangerous:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message=reason,
            )

        # Check working directory restriction
        violates_restriction, reason = _check_working_dir_restriction(
            argv, self.working_dir
        )
        if violates_restriction:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message=reason,
            )

        return ExecutionResult(
            success=True,
            stdout=None,
            stderr=None,
            skipped=True,
            skip_message="Command passes safety checks; static analysis environment does not execute commands. To execute, use UnsafeBashEnvironment or LLMSandboxBashEnvironment.",
            analysis_result=argv,
        )


class UnsafeBashEnvironment(BashEnvironment):
    """Unsafe environment that executes bash commands directly with subprocess."""

    def execute(self, command: str) -> ExecutionResult:
        """Execute bash command after safety checks.

        Args:
            command (str): The bash command to execute.

        Returns:
            ExecutionResult: Execution outcome with captured stdout/stderr and
            success flag, or a skipped result if safety checks fail.
        """
        # Parse and validate
        try:
            argv = shlex.split(command)
        except ValueError as e:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message=f"Failed to parse command: {e!s}",
            )

        if not argv:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message="Empty command",
            )

        # Check for dangerous commands
        is_dangerous, reason = _is_dangerous_command(argv)
        if is_dangerous:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message=reason,
            )

        # Check for dangerous paths
        has_dangerous, reason = _check_dangerous_paths(argv)
        if has_dangerous:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message=reason,
            )

        # Check working directory restriction
        violates_restriction, reason = _check_working_dir_restriction(
            argv, self.working_dir
        )
        if violates_restriction:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message=reason,
            )

        # Execute command
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.working_dir,
            )

            stdout, stdout_truncated = _truncate_output(result.stdout.strip())
            stderr, stderr_truncated = _truncate_output(result.stderr.strip())

            # Append truncation warnings if needed
            if stdout_truncated:
                stdout += "\n[Output truncated - stdout exceeded 10KB]"
            if stderr_truncated:
                stderr += "\n[Output truncated - stderr exceeded 10KB]"

            return ExecutionResult(
                success=result.returncode == 0, stdout=stdout, stderr=stderr
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message=f"Execution timed out after {self.timeout} seconds",
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message=f"Subprocess execution error: {e!s}",
            )


class LLMSandboxBashEnvironment(BashEnvironment):
    """Environment using llm-sandbox for secure Docker-based bash execution."""

    def execute(self, command: str) -> ExecutionResult:
        """Execute bash command using llm-sandbox in an isolated Docker container.

        Validates command safety first, then delegates to ``SandboxSession``
        from the ``llm-sandbox`` package. Returns a skipped result if
        ``llm-sandbox`` is not installed.

        Args:
            command (str): The bash command to execute.

        Returns:
            ExecutionResult: Execution outcome with stdout/stderr and success
            flag, or a skipped result on safety check failure or sandbox error.
        """
        # Parse and validate
        try:
            argv = shlex.split(command)
        except ValueError as e:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message=f"Failed to parse command: {e!s}",
            )

        if not argv:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message="Empty command",
            )

        # Check for dangerous commands
        is_dangerous, reason = _is_dangerous_command(argv)
        if is_dangerous:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message=reason,
            )

        # Check for dangerous paths
        has_dangerous, reason = _check_dangerous_paths(argv)
        if has_dangerous:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message=reason,
            )

        # Check working directory restriction (note: may not apply in sandbox, but check anyway)
        violates_restriction, reason = _check_working_dir_restriction(
            argv, self.working_dir
        )
        if violates_restriction:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message=reason,
            )

        try:
            from llm_sandbox import SandboxSession
        except ImportError:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message="llm-sandbox not installed. Install with: pip install 'mellea[sandbox]'",
            )

        try:
            with SandboxSession(
                lang="sh", verbose=False, keep_template=False
            ) as session:
                result = session.run(command, timeout=self.timeout)

                stdout, stdout_truncated = _truncate_output(result.stdout.strip())
                stderr, stderr_truncated = _truncate_output(result.stderr.strip())

                # Append truncation warnings if needed
                if stdout_truncated:
                    stdout += "\n[Output truncated - stdout exceeded 10KB]"
                if stderr_truncated:
                    stderr += "\n[Output truncated - stderr exceeded 10KB]"

                return ExecutionResult(
                    success=result.exit_code == 0, stdout=stdout, stderr=stderr
                )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message=f"Sandbox execution timed out after {self.timeout} seconds",
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message=f"Sandbox execution error: {e!s}",
            )


def bash_executor(command: str, working_dir: str | None = None) -> ExecutionResult:
    """Execute a bash command in a Docker-isolated sandbox.

    This is the recommended entry point for production use. Commands are validated
    against a conservative safety denylist before execution. Execution happens
    in an isolated Docker container via llm-sandbox.

    Safety defaults: Refuses sudo, interactive shells, destructive operations
    (rm -rf, git push --force), and writes to system paths (/etc, /sys, /proc, etc.).

    Args:
        command: The bash command to execute.
        working_dir: Optional directory to restrict file operations to. If specified,
            all file writes are sandboxed to this directory and /tmp.

    Returns:
        An ``ExecutionResult`` with stdout, stderr, and a success flag. If the command
        was rejected for safety reasons, ``skipped=True`` and ``skip_message`` contains
        the reason.
    """
    env = LLMSandboxBashEnvironment(working_dir=working_dir)
    return env.execute(command)


def local_bash_executor(
    command: str, working_dir: str | None = None
) -> ExecutionResult:
    """Execute a bash command in the current shell (unsafe for LLM-generated code).

    This is for local development and testing only. Commands are validated against
    a conservative safety denylist, but execution happens directly in the current
    shell with no isolation.

    Safety defaults: Refuses sudo, interactive shells, destructive operations
    (rm -rf, git push --force), and writes to system paths (/etc, /sys, /proc, etc.).

    Args:
        command: The bash command to execute.
        working_dir: Optional directory to restrict file operations to. If specified,
            the command is executed with this directory as the working directory.

    Returns:
        An ``ExecutionResult`` with stdout, stderr, and a success flag. If the command
        was rejected for safety reasons, ``skipped=True`` and ``skip_message`` contains
        the reason.
    """
    env = UnsafeBashEnvironment(working_dir=working_dir)
    return env.execute(command)
