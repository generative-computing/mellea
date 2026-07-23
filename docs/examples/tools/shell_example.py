# pytest: e2e, ollama, qualitative
"""Example usage patterns for bash_executor tool.

Demonstrates multiple ways to use Mellea's bash execution capabilities:
1. Direct execution for local commands
2. Wrapping as a MelleaTool for agent use
3. LLM-based tool calling with forced tool use
4. Working directory and path restrictions
5. Demonstrate safety features
6. Integration with error handling

⚠️  Security note: bash_executor runs commands locally with a conservative
safety denylist (recommended for typical agentic workflows). The denylist
enforces: no sudo, no rm -rf, no destructive git operations, no writes to
/etc, /sys, /proc, etc. Write operations can also be constrained with
``working_dir`` and explicit ``allowed_paths``.

For higher isolation requirements (untrusted code, security research),
provide isolation at the application layer (containers, VMs).

Note: Commands must use argv-friendly syntax (no pipes, redirects, or shell builtins).
Use individual commands and compose them in Python instead.
"""

from mellea import MelleaSession, start_session
from mellea.backends import ModelOption
from mellea.backends.tools import MelleaTool
from mellea.stdlib.requirements import uses_tool
from mellea.stdlib.tools.shell import bash_executor


def example_1_direct_execution() -> None:
    """Example 1: Execute bash commands locally (default)."""
    print("=== Example 1: Local Execution (Default) ===")

    # Execute a simple command locally
    result = bash_executor("echo 'Hello from Bash'")
    print("Command: echo 'Hello from Bash'")
    print(f"Success: {result.success}")
    print(f"Output: {result.stdout}")
    print()

    # Execute a command to list files (no pipes/redirects)
    result = bash_executor("ls -la")
    print("Command: ls -la")
    print(f"Success: {result.success}")
    if result.stdout:
        # Show first few lines
        lines = result.stdout.split("\n")[:3]
        print("Output (first 3 lines):\n" + "\n".join(lines))
    print()

    # Demonstrate that pipes are blocked (for security)
    result = bash_executor("ls -la | wc -l")
    print("Command: ls -la | wc -l (pipe operator blocked)")
    print(f"Rejected: {result.skipped}")
    print(f"Reason: {result.skip_message}")
    print()

    # Attempt a dangerous command (will be rejected)
    result = bash_executor("sudo echo unsafe")
    print("Command: sudo echo unsafe")
    print(f"Skipped: {result.skipped}")
    print(f"Reason: {result.skip_message}")
    print()


def example_2_wrapped_as_tool() -> None:
    """Example 2: Wrap bash executor as a MelleaTool for LLM use."""
    print("=== Example 2: Wrapped as MelleaTool ===")

    # Create tool from bash executor (local execution by default)
    bash_tool = MelleaTool.from_callable(bash_executor)
    print(f"Tool name: {bash_tool.name}")
    print(f"Tool schema keys: {bash_tool.as_json_tool.keys()}")
    print()

    # Invoke the tool directly (normally LLM would call this)
    result = bash_tool.run("pwd")
    print("Tool invocation result:")
    print(f"  Success: {result.success}")
    print(f"  Output: {result.stdout}")
    print()


def example_3_llm_with_forced_tool_use(m: MelleaSession) -> None:
    """Example 3: LLM generates bash commands with forced tool use (requires Ollama).

    This mirrors the Python interpreter pattern: ask the LLM to generate
    a bash command, force it to use the tool, then execute the command.

    Requirements:
        - Ollama running locally (or compatible LLM configured)
        - Run: ollama serve
    """
    print("=== Example 3: LLM-Generated Bash Commands with Forced Tool Use ===")

    result = m.instruct(
        description="Use bash to find Python files in the current directory. "
        "Generate a single command using find or ls (no pipes, redirects, or shell operators allowed).",
        requirements=[uses_tool(bash_executor)],
        model_options={ModelOption.TOOLS: [MelleaTool.from_callable(bash_executor)]},
        tool_calls=True,
    )

    if result.tool_calls is None:
        raise ValueError("Expected tool_calls but got None")

    tool_call = next(
        (tc for tc in result.tool_calls if tc.name == "bash_executor"), None
    )
    if tool_call is None:
        available_tools = [tc.name for tc in result.tool_calls]
        raise ValueError(
            f"Expected tool 'bash_executor' in tool_calls, but got: {available_tools}"
        )
    if "command" not in tool_call.args:
        raise ValueError(
            f"Expected 'command' argument in tool call args, "
            f"but got: {list(tool_call.args.keys())}"
        )

    command = tool_call.args["command"]
    print(f"LLM generated bash command:\n  {command}\n")

    # Execute the command
    exec_result = tool_call.call_func()

    print("Execution result:")
    print(f"  Success: {exec_result.success}")
    print(f"  Skipped: {exec_result.skipped}")
    if exec_result.skip_message:
        print(f"  Skip reason: {exec_result.skip_message}")
    print(f"  Output: {exec_result.stdout}")
    if exec_result.stderr:
        print(f"  Error: {exec_result.stderr}")
    print()


def example_4_with_working_dir() -> None:
    """Example 4: Restrict write validation and execution cwd to a directory."""
    print("=== Example 4: Working Directory Restriction ===")

    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Working directory: {tmpdir}")

        # Create a file using touch within the working directory (redirects blocked)
        result = bash_executor("touch myfile.txt", working_dir=tmpdir)
        print(f"Command: touch myfile.txt (relative path, executed in {tmpdir})")
        print(f"Success: {result.success}")
        print()

        # Verify the file was created
        file_path = os.path.join(tmpdir, "myfile.txt")
        if os.path.exists(file_path):
            print(f"✓ File created at: {file_path}")
        print()

        # Read it back
        result = bash_executor("cat myfile.txt", working_dir=tmpdir)
        print("Command: cat myfile.txt")
        print(f"Output: {result.stdout}")
        print()

        # Writing to /tmp is always allowed (temp directory exception)
        result = bash_executor("touch /tmp/tmpfile.txt", working_dir=tmpdir)
        print(f"Command: touch /tmp/tmpfile.txt (with working_dir={tmpdir})")
        print(f"Success: {result.success} (note: /tmp is always allowed)")
        print()

        # Attempt to write to system paths (will be rejected)
        result = bash_executor("touch /etc/config.txt", working_dir=tmpdir)
        print(f"Command: touch /etc/config.txt (with working_dir={tmpdir})")
        print(f"Rejected: {result.skipped}")
        print(f"Reason: {result.skip_message}")
        print()


def example_5_safety_features() -> None:
    """Example 5: Demonstrate safety features."""
    print("=== Example 5: Safety Features ===")

    dangerous_commands = [
        ("rm -rf /home", "Recursive force delete"),
        ("git push --force", "Force git push"),
        ("sudo whoami", "Privilege escalation"),
        ("bash -i", "Interactive shell"),
        ("touch /etc/config", "Write to system path"),
    ]

    for cmd, description in dangerous_commands:
        result = bash_executor(cmd)
        print(f"{description}: {cmd}")
        print(f"  Rejected: {result.skipped}")
        print(f"  Reason: {result.skip_message}")
        print()


def example_6_error_handling() -> None:
    """Example 6: Handle execution errors gracefully."""
    print("=== Example 6: Error Handling ===")

    # Command that fails (returns non-zero exit code)
    result = bash_executor("false")
    print("Command: false (POSIX command that returns exit code 1)")
    print(f"Success: {result.success}")
    print(f"Return code indicates failure: {not result.success}")
    print()

    # Command that doesn't exist
    result = bash_executor("nonexistent_command_xyz")
    print("Command: nonexistent_command_xyz")
    print(f"Success: {result.success}")
    if not result.success and result.stderr is not None:
        print(f"Error output: {result.stderr[:100]}")
    print()


if __name__ == "__main__":
    example_1_direct_execution()
    example_2_wrapped_as_tool()

    # Example 3: Run with LLM-based tool calling (requires Ollama or compatible LLM)
    try:
        m = start_session()
        example_3_llm_with_forced_tool_use(m)
    except Exception as e:
        print(f"Example 3 skipped: {e!s}")
        print("  Requires: Ollama running locally or compatible LLM configured")
        print("  See: https://docs.ollama.ai/")

    example_4_with_working_dir()
    example_5_safety_features()
    example_6_error_handling()
