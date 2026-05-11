"""Bash shell command execution tool and execution environments for agentic workflows.

Provides ``BashEnvironment`` (abstract base for bash execution) and three concrete
implementations: ``StaticBashEnvironment`` (parse and safety-check only, no execution),
``UnsafeBashEnvironment`` (subprocess execution in the current shell), and
``LLMSandboxBashEnvironment`` (Docker-isolated execution via ``llm-sandbox``). All
environments enforce a conservative safety denylist (sudo, rm -rf, git push --force,
system paths, interactive shells). Write operations may also be constrained by
``working_dir`` and ``allowed_paths``. The top-level ``bash_executor`` and
``local_bash_executor`` functions are ready to be wrapped as ``MelleaTool`` instances
for ReACT or other agentic loops.
"""

import shlex
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

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

    # Check for shell metacharacters that would need shell interpretation
    # After shlex.split(), these characters in argv indicate shell operators (not quoted strings)
    # These are only dangerous if they're standalone tokens or at token boundaries
    shell_operators = {"<", ">", "|", ";", "&", "&&", "||"}
    shell_operator_sequences = (">>", ">&", "<<", "|&", "&&", "||")

    for arg in argv:
        # Check for redirect/pipe/logic operators (these are shell operators)
        if arg in shell_operators:
            return True, f"Shell operator '{arg}' is not allowed"
        # Check for combined operators or operators within tokens
        if any(op in arg for op in shell_operator_sequences):
            return True, "Shell operator is not allowed"
        # Check for semicolon (command separator) within or as token
        if ";" in arg:
            return True, "Command chaining (;) is not allowed"

    # Check for command substitution (backticks, $(...))
    for arg in argv:
        if "`" in arg or "$(" in arg:
            return True, "Command substitution is not allowed"
        # Check for variable expansion patterns
        if "${" in arg:
            return True, "Variable expansion is not allowed"

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


def _is_path_within(path_str: str, allowed_root: str) -> bool:
    """Return whether a resolved path is equal to or nested under an allowed root."""
    if path_str == allowed_root:
        return True
    allowed_root_prefix = (
        allowed_root if allowed_root.endswith("/") else allowed_root + "/"
    )
    return path_str.startswith(allowed_root_prefix)


def _normalize_allowed_path(allowed_path: str) -> str:
    """Normalize an allowlisted path for string-prefix containment checks."""
    return str(Path(allowed_path).expanduser())


def _resolve_allowed_paths(allowed_paths: list[str]) -> list[str]:
    """Normalize allowed path roots for prefix-based containment checks."""
    normalized_paths: list[str] = []
    for allowed_path in allowed_paths:
        try:
            normalized_paths.append(_normalize_allowed_path(allowed_path))
        except Exception:
            logger.warning("Skipping invalid allowed path: %s", allowed_path)
    return normalized_paths


def _is_default_safe_write_path(path_str: str) -> bool:
    """Return whether a path is in a default safe write location."""
    home_dir = str(Path.home())
    return path_str.startswith(("/tmp", "/private/tmp")) or _is_path_within(
        path_str, home_dir
    )


def _check_dangerous_paths(
    argv: list[str], allowed_paths: list[str] | None = None
) -> tuple[bool, str]:
    """Check if command targets dangerous or disallowed filesystem paths.

    Args:
        argv: Tokenized command arguments.
        allowed_paths: Optional additional resolved path roots where writes are allowed.

    Returns:
        A tuple of (has_dangerous_paths, reason_message).
    """
    write_commands = {"rm", "touch", "cp", "mv", "mkdir", "mkfifo", "mknod", "tee"}
    if not argv or argv[0].split("/")[-1] not in write_commands:
        return False, ""

    resolved_allowed_paths = _resolve_allowed_paths(allowed_paths or [])

    for arg in argv[1:]:
        if arg.startswith("-"):
            continue

        expanded_arg = str(Path(arg).expanduser())
        allowlist_candidate_path: str | None = None
        if arg.startswith(("/", "~")):
            allowlist_candidate_path = expanded_arg

        for danger_path in DANGEROUS_PATHS:
            if expanded_arg == danger_path or expanded_arg.startswith(
                danger_path + "/"
            ):
                return True, f"Writing to '{expanded_arg}' is not allowed"

        if resolved_allowed_paths:
            candidate_for_allowlist = allowlist_candidate_path
            if candidate_for_allowlist is None:
                candidate_for_allowlist = str(Path(arg).resolve(strict=False))

            if not any(
                _is_path_within(candidate_for_allowlist, allowed_root)
                for allowed_root in resolved_allowed_paths
            ):
                return (
                    True,
                    f"Path '{arg}' is outside explicitly allowed paths: {', '.join(allowed_paths or [])}",
                )

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
        allowed_paths (list[str] | None): Optional explicit write allowlist. When
            provided, write-target paths must fall under one of these roots in
            addition to passing the default dangerous-path checks.
        working_dir (str | None): Optional directory restriction for write
            operations. When specified, writes must remain within this directory
            or ``/tmp``.
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

    def _validate_command(self, command: str) -> ExecutionResult | list[str]:
        """Parse and validate a command before execution.

        The shared validation step performs argv parsing, rejects dangerous shell
        constructs, applies path safety checks, and enforces ``allowed_paths`` and
        ``working_dir`` restrictions for write operations.

        Args:
            command: The bash command string to validate.

        Returns:
            Either the validated argv list or a skipped ``ExecutionResult``
            describing why validation failed.
        """
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

        is_dangerous, reason = _is_dangerous_command(argv)
        if is_dangerous:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message=reason,
            )

        has_dangerous, reason = _check_dangerous_paths(argv, self.allowed_paths)
        if has_dangerous:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message=reason,
            )

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

        return argv

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
        validated = self._validate_command(command)
        if isinstance(validated, ExecutionResult):
            return validated

        argv = validated

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
        validated = self._validate_command(command)
        if isinstance(validated, ExecutionResult):
            return validated

        argv = validated

        # Execute command with shell=False to prevent shell metacharacter bypass
        try:
            result = subprocess.run(
                argv,
                shell=False,
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

        Validates command safety first, then runs the command inside a Python-based
        sandbox session. The validated shell command is executed via
        ``subprocess.run(..., cwd=working_dir)`` inside the container so that
        sandbox execution honors ``self.working_dir`` when provided. Returns a
        skipped result if ``llm-sandbox`` is not installed.

        Args:
            command (str): The bash command to execute.

        Returns:
            ExecutionResult: Execution outcome with stdout/stderr and success
            flag, or a skipped result on safety check failure, timeout, or
            sandbox error.
        """
        validated = self._validate_command(command)
        if isinstance(validated, ExecutionResult):
            return validated

        argv = validated

        try:
            from llm_sandbox import SandboxSession
            from llm_sandbox.exceptions import SandboxTimeoutError
        except ImportError:
            return ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                skipped=True,
                skip_message="llm-sandbox not installed. Install with: pip install 'mellea[sandbox]'",
            )

        sandbox_workdir = self.working_dir or "/sandbox"
        shell_command = " ".join(shlex.quote(arg) for arg in argv)
        python_wrapper = (
            "import subprocess\n"
            "import sys\n"
            f"result = subprocess.run({shell_command!r}, shell=True, cwd={sandbox_workdir!r}, "
            "capture_output=True, text=True)\n"
            "sys.stdout.write(result.stdout)\n"
            "sys.stderr.write(result.stderr)\n"
            "raise SystemExit(result.returncode)\n"
        )

        try:
            with SandboxSession(
                lang="python",
                verbose=False,
                keep_template=False,
                workdir=sandbox_workdir,
                execution_timeout=self.timeout,
            ) as session:
                result = session.run(python_wrapper, timeout=self.timeout)

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
        except SandboxTimeoutError:
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
        working_dir: Optional sandbox working directory. When specified, sandboxed
            command execution uses this directory as its cwd, and write-path
            validation also restricts writes to this directory and ``/tmp``.

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
