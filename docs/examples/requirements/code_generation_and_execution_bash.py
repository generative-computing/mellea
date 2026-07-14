# pytest: e2e, ollama, qualitative, slow
"""Example demonstrating Python code generation for CLI data processing tasks.

This example shows how to use Mellea to:
1. Accept user input specifying file/directory management and CLI tasks
2. Generate Python code with bash_executor() calls for data processing workflows
3. Execute the generated Python code to orchestrate multiple bash_executor calls

The pipeline implements a 4-step process:
1. User Input - Accept natural language request for file/CLI operations
2. Context Loading - Load file listings and sample data
3. Code Generation - Generate Python code with sequential bash_executor calls
4. Code Execution - Execute the generated Python code to process data

Example tasks:
- Find and process files matching patterns
- Transform and reorganize directory structures
- Extract and aggregate data from multiple files
- Generate reports from log files or data files
- Search and filter data across multiple files

This approach avoids shell operators (pipes, redirects) by generating Python control flow instead.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import mellea
from mellea.stdlib.requirements.python_tools import PythonSyntaxValid
from mellea.stdlib.sampling import ModelFriendlyRepairStrategy
from mellea.stdlib.tools.interpreter import python_tool
from mellea.stdlib.tools.shell import bash_executor


def _extract_python_code_from_output(generated: str) -> str | None:
    """Extract Python code block from model output.

    Looks for code blocks marked with ```python or ``` markers.

    Args:
        generated: Raw model output string.

    Returns:
        Extracted Python code string, or None if extraction failed.
    """
    lines = generated.split("\n")
    in_code_block = False
    code_lines = []

    for line in lines:
        if line.strip().startswith("```python"):
            in_code_block = True
            continue
        elif line.strip().startswith("```") and in_code_block:
            in_code_block = False
            break
        elif in_code_block:
            code_lines.append(line)

    if code_lines:
        code = "\n".join(code_lines).strip()
        if code:
            return code

    return None


def create_sample_workspace(workspace_dir: str) -> None:
    """Create sample files for demonstration."""
    Path(workspace_dir).mkdir(exist_ok=True)

    # Create logs directory with sample log files
    logs_dir = Path(workspace_dir) / "logs"
    logs_dir.mkdir(exist_ok=True)

    log_content = """2024-01-15 10:23:45 INFO Starting application
2024-01-15 10:24:01 DEBUG Connecting to database
2024-01-15 10:24:05 ERROR Database connection failed
2024-01-15 10:24:06 WARN Retrying connection
2024-01-15 10:24:10 INFO Connected successfully
2024-01-15 10:25:30 DEBUG Processing user request
2024-01-15 10:25:45 INFO Request completed
2024-01-15 10:26:00 ERROR Out of memory exception
2024-01-15 10:26:01 WARN Application degraded"""

    for i in range(1, 4):
        (logs_dir / f"app_{i}.log").write_text(log_content)

    # Create data directory with sample data files
    data_dir = Path(workspace_dir) / "data"
    data_dir.mkdir(exist_ok=True)

    data_files = {
        "sales_2024_q1.txt": """Product,Revenue,Units
Widget A,15000,300
Widget B,22000,400
Gadget X,18000,250
Gadget Y,25000,350""",
        "sales_2024_q2.txt": """Product,Revenue,Units
Widget A,16500,320
Widget B,24000,420
Gadget X,19500,270
Gadget Y,27500,380""",
        "sales_2024_q3.txt": """Product,Revenue,Units
Widget A,17000,340
Widget B,26000,450
Gadget X,21000,290
Gadget Y,29000,400""",
    }

    for filename, content in data_files.items():
        (data_dir / filename).write_text(content)

    # Create config directory
    config_dir = Path(workspace_dir) / "config"
    config_dir.mkdir(exist_ok=True)

    (config_dir / "app.conf").write_text("APP_NAME=DataProcessor\nVERSION=1.0.0")
    (config_dir / "db.conf").write_text("HOST=localhost\nPORT=5432\nDB=myapp")


def get_workspace_context(workspace_dir: str) -> str:
    """Get file listing and structure of workspace."""
    context = f"Workspace directory: {workspace_dir}\n\n"
    context += "Directory structure:\n"

    for root, dirs, files in os.walk(workspace_dir):
        level = root.replace(workspace_dir, "").count(os.sep)
        indent = " " * 2 * level
        context += f"{indent}{os.path.basename(root)}/\n"

        subindent = " " * 2 * (level + 1)
        for file in sorted(files):
            file_path = Path(root) / file
            size = file_path.stat().st_size
            context += f"{subindent}{file} ({size} bytes)\n"

    return context


def process_user_request(
    user_request: str, workspace_dir: str, request_number: int = 1
) -> None:
    """Process a user request to generate and execute Bash code.

    Args:
        user_request: Natural language request for file/CLI operations
        workspace_dir: Working directory with sample files
        request_number: Number for naming purposes
    """
    print(f"\nUser request: {user_request}")
    print("-" * 70)

    m = mellea.start_session()

    workspace_context = get_workspace_context(workspace_dir)
    sanitized_request = repr(user_request)

    prompt = f"""You are a Python code generator that uses bash_executor for CLI operations.

User request:
{sanitized_request}

Current workspace context:
{workspace_context}

Generate Python code that:
1. Imports bash_executor from mellea.stdlib.tools.shell
2. Uses sequential bash_executor() calls to process files and data
3. Breaks complex tasks into multiple simple commands (NO pipes, redirects, or shell chaining)
4. Uses Python loops, conditionals, and string operations to orchestrate commands
5. Processes intermediate results in Python variables
6. Prints the final results

IMPORTANT CONSTRAINTS:
- NO shell operators (pipes |, redirects >, >>, semicolons ;)
- Each bash_executor call executes ONE simple command
- Use Python control flow (for loops, if statements) to sequence commands
- Extract and process bash_executor output in Python
- Working directory is: {workspace_dir}

IMPORTANT API DETAILS:
- bash_executor() returns an ExecutionResult object
- Check result.success (bool) to verify command executed successfully
- Check result.stdout (str) for command output (or None if failed)
- Check result.skipped (bool) to see if command was blocked by guardrails
- Check result.skip_message (str) for why a command was skipped

IMPORTANT PARSING NOTES:
- wc -l FILE outputs: "N FILE" (number + filename), extract the first word only
- grep -c PATTERN FILE outputs just the count (easier to parse)
- grep PATTERN FILE outputs matching lines (use for displaying content)
- cat FILE outputs entire file contents (use for showing all data)
- Always strip() output and check if parsing will work
- When in doubt, use commands that output just data without filenames

DIRECTORY STRUCTURE:
The workspace contains:
- logs/ — log files (*.log)
- data/ — data files (*.txt, *.csv)
- config/ — configuration files

COMMON FILE OPERATIONS:
- Show file contents: cat data/sales_2024_q1.txt
- Find files: find data -name '*.txt' or find data -name '*.csv'
- Search in files: grep PATTERN filename (finds lines containing PATTERN anywhere)
- Count matching lines: grep -c PATTERN filename (outputs just the number)
- Count all lines: wc -l filename (output: "N filename", extract first word)

GREP PATTERN TIPS:
- To find lines containing a word: grep -c ERROR logs/app.log (matches ERROR anywhere in line)
- Do NOT use anchors like ^ERROR or ^WARN unless you know the line starts with that text
- For log files with timestamps: use grep -c ERROR (without ^) to match anywhere in the line
- Example log format: "2024-01-15 10:24:05 ERROR Database failed" → grep ERROR will match
- grep -c outputs just the count, perfect for parsing with int()

BANNED OPERATIONS (will be blocked):
- Input/output redirection: >, >>, <, 2>&1, >&2
- Pipes: |, |&
- Command chaining: ;, &&, ||
- Variable expansion: $(...), `...`, ${{...}}
- Code execution: python -c, bash -c, etc.

Example pattern for showing file contents:
```python
from mellea.stdlib.tools.shell import bash_executor

# Step 1: Find all data files in data directory
result = bash_executor("find data -name '*.txt'", working_dir="{workspace_dir}")
if not result.success:
    print(f"Failed: {{result.skip_message}}")
else:
    files = result.stdout.strip().split('\\n')

    # Step 2: Display each file's contents
    for file in files:
        if file:
            result = bash_executor(f"cat {{file}}", working_dir="{workspace_dir}")
            if result.success:
                print(f"\\nContents of {{file}}:")
                print(result.stdout)
```

Example pattern for counting with grep (CORRECT - no anchors):
```python
# Count ERROR entries - use grep -c ERROR (NOT ^ERROR)
result = bash_executor(f"grep -c ERROR {{file}}", working_dir="{workspace_dir}")
if result.success:
    count = int(result.stdout.strip())  # grep -c outputs just the number
    print(f"{{file}}: {{count}} errors")

# WRONG: Do NOT use this - anchors won't match timestamped logs:
# result = bash_executor(f"grep -c ^ERROR {{file}}", ...)  # WRONG!
```

ALWAYS use working_dir="{workspace_dir}" (the main workspace), NOT subdirectories like "logs/" or "data/".

Generate the Python code now:"""

    print("Generating Python code for data processing...")
    requirements: list = [PythonSyntaxValid()]
    strategy = ModelFriendlyRepairStrategy(loop_budget=3, requirements=requirements)
    generated = m.instruct(prompt, strategy=strategy)

    if generated is None:
        print("  ✗ Model failed to generate output (requirements loop exhausted)")
        return

    generated_str = str(generated)
    if not generated_str.strip():
        print("  ✗ Model failed to generate output")
        return

    code = _extract_python_code_from_output(generated_str)
    if code is None:
        print("  ✗ Failed to extract Python code from model output")
        print(f"\nModel output:\n{generated_str}")
        return

    print("\nGenerated Python code:")
    print(code)

    # Execute the generated Python code using python_tool
    print("\nExecuting generated code...")

    # Create a python execution tool that allows bash_executor imports
    tool = python_tool(
        tier="local_unsafe", allowed_imports=["mellea"], name="bash_executor_runner"
    )

    # Prepend the bash_executor import if not already present
    if "from mellea.stdlib.tools.shell import bash_executor" not in code:
        code = "from mellea.stdlib.tools.shell import bash_executor\n" + code

    # Execute the code
    result = tool.run(code=code)

    if result.skipped:
        print(f"  ✗ Code execution was skipped: {result.skip_message}")
        return

    if not result.success:
        print("  ✗ Code execution failed")
        if result.stderr:
            print(f"  Error: {result.stderr}")
        if result.exit_code is not None:
            print(f"  Exit code: {result.exit_code}")
        return

    # Print any output from the code
    if result.stdout:
        print(result.stdout)

    print("\n  ✓ Code executed successfully")


def main():
    """Demonstrate complete pipeline for CLI data processing tasks."""
    parser = argparse.ArgumentParser(
        description="Bash Code Generation: CLI Data Processing and File Operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default sample workspace and predefined requests
  uv run python code_generation_and_execution_bash.py

  # Use a custom workspace directory
  uv run python code_generation_and_execution_bash.py --workspace /path/to/workspace

  # Accept user input interactively
  uv run python code_generation_and_execution_bash.py --interactive
        """,
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Workspace directory (uses temp directory if not specified)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Accept user input interactively instead of using predefined requests",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Python Code Generation: CLI Data Processing and File Operations")
    print("=" * 70)

    # Use provided workspace or create temporary one
    if args.workspace:
        workspace_dir = args.workspace
        Path(workspace_dir).mkdir(parents=True, exist_ok=True)
    else:
        workspace_dir = tempfile.mkdtemp(prefix="mellea_bash_")

    print(f"\nWorkspace directory: {workspace_dir}")

    # Create sample files
    create_sample_workspace(workspace_dir)
    print("Created sample workspace with files and directories")

    if args.interactive:
        print("\n" + "=" * 70)
        print("Interactive Mode")
        print("=" * 70)
        print("\nExamples of requests:")
        print("  - Find all ERROR entries in log files and count them")
        print("  - Count the total number of lines in all log files")
        print("  - Find all WARN entries and show which files contain them")
        print("  - Count how many log files contain ERROR entries")

        request_number = 1
        while True:
            print("\nEnter a request (or 'quit' to exit):")
            user_input = input("\n> ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                break

            if not user_input:
                print("Please enter a valid request.")
                continue

            process_user_request(
                user_input, workspace_dir, request_number=request_number
            )
            request_number += 1

    else:
        # Use predefined requests that benefit from Python orchestration
        user_requests = [
            "Find all ERROR entries in the log files and count them",
            "List all log files and count the total number of lines across all of them",
            "Show the contents of all data files in the data directory",
            "Find all WARN entries in the log files and display the files containing them",
            "Count how many log files contain ERROR entries",
        ]

        # Process each user request
        for i, request in enumerate(user_requests, 1):
            print("\n" + "=" * 70)
            print(f"Example {i}")
            print("=" * 70)

            process_user_request(request, workspace_dir, request_number=i)

    print("\n" + "=" * 70)
    print("Pipeline completed!")
    print(f"Workspace preserved at: {workspace_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
