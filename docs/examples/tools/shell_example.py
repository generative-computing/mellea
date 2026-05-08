# pytest: unit, qualitative
"""Example usage patterns for bash_executor and local_bash_executor tools.

Demonstrates three ways to use Mellea's bash execution capabilities:
1. Direct execution for non-LLM tasks
2. Wrapping as a MelleaTool for agent use
3. Integration with requirements framework for rejection sampling

Safety note: bash_executor uses Docker isolation via llm-sandbox (recommended
for production). local_bash_executor runs commands directly (for dev/testing only).
Both enforce a conservative safety denylist: no sudo, no rm -rf, no destructive
git operations, no writes to /etc, /sys, /proc, etc.
"""

from mellea.backends.tools import MelleaTool
from mellea.stdlib.tools.shell import bash_executor, local_bash_executor


def example_1_direct_execution() -> None:
    """Example 1: Execute bash commands directly."""
    print("=== Example 1: Direct Execution ===")

    # Execute a simple command
    result = local_bash_executor("echo 'Hello from Bash'")
    print("Command: echo 'Hello from Bash'")
    print(f"Success: {result.success}")
    print(f"Output: {result.stdout}")
    print()

    # Execute a command with pipes and redirects
    result = local_bash_executor("ls -la | wc -l")
    print("Command: ls -la | wc -l")
    print(f"Success: {result.success}")
    print(f"Output: {result.stdout}")
    print()

    # Attempt a dangerous command (will be rejected)
    result = local_bash_executor("sudo echo unsafe")
    print("Command: sudo echo unsafe")
    print(f"Skipped: {result.skipped}")
    print(f"Reason: {result.skip_message}")
    print()


def example_2_wrapped_as_tool() -> None:
    """Example 2: Wrap bash executor as a MelleaTool for LLM use."""
    print("=== Example 2: Wrapped as MelleaTool ===")

    # Create tool from bash executor
    bash_tool = MelleaTool.from_callable(local_bash_executor)
    print(f"Tool name: {bash_tool.name}")
    print(f"Tool schema keys: {bash_tool.as_json_tool.keys()}")
    print()

    # Invoke the tool directly (normally LLM would call this)
    result = bash_tool.run("pwd")
    print("Tool invocation result:")
    print(f"  Success: {result.success}")
    print(f"  Output: {result.stdout}")
    print()


def example_3_with_working_dir() -> None:
    """Example 3: Restrict command execution to a specific directory."""
    print("=== Example 3: Working Directory Restriction ===")

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Working directory: {tmpdir}")

        # Create a file in the working directory
        result = local_bash_executor(
            f"echo 'project content' > {tmpdir}/myfile.txt", working_dir=tmpdir
        )
        print(f"Command: echo 'project content' > {tmpdir}/myfile.txt")
        print(f"Success: {result.success}")
        print()

        # Read it back
        result = local_bash_executor(f"cat {tmpdir}/myfile.txt", working_dir=tmpdir)
        print(f"Command: cat {tmpdir}/myfile.txt")
        print(f"Output: {result.stdout}")
        print()

        # Attempt to write outside working directory (will be rejected)
        result = local_bash_executor(
            "echo 'bad' > /tmp/outside.txt", working_dir=tmpdir
        )
        print(f"Command: echo 'bad' > /tmp/outside.txt (with working_dir={tmpdir})")
        print(f"Skipped: {result.skipped}")
        print(f"Reason: {result.skip_message}")
        print()


def example_4_safety_features() -> None:
    """Example 4: Demonstrate safety features."""
    print("=== Example 4: Safety Features ===")

    dangerous_commands = [
        ("rm -rf /home", "Recursive force delete"),
        ("git push --force", "Force git push"),
        ("sudo whoami", "Privilege escalation"),
        ("bash -i", "Interactive shell"),
        ("touch /etc/config", "Write to system path"),
    ]

    for cmd, description in dangerous_commands:
        result = local_bash_executor(cmd)
        print(f"{description}: {cmd}")
        print(f"  Rejected: {result.skipped}")
        print(f"  Reason: {result.skip_message}")
        print()


def example_5_error_handling() -> None:
    """Example 5: Handle execution errors gracefully."""
    print("=== Example 5: Error Handling ===")

    # Command that fails (returns non-zero exit code)
    result = local_bash_executor("exit 1")
    print("Command: exit 1")
    print(f"Success: {result.success}")
    print(f"Stderr: {result.stderr}")
    print()

    # Command that doesn't exist
    result = local_bash_executor("nonexistent_command_xyz")
    print("Command: nonexistent_command_xyz")
    print(f"Success: {result.success}")
    if not result.success and result.stderr is not None:
        print(f"Error output: {result.stderr[:100]}")
    print()


if __name__ == "__main__":
    example_1_direct_execution()
    example_2_wrapped_as_tool()
    example_3_with_working_dir()
    example_4_safety_features()
    example_5_error_handling()
