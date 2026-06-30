# pytest: e2e, ollama, qualitative, slow
"""Example demonstrating Bash code generation for CLI data processing tasks.

This example shows how to use Mellea to:
1. Accept user input specifying file/directory management and CLI tasks
2. Generate Bash code for data processing workflows
3. Execute the code using bash_executor() from mellea.stdlib.tools.shell

The pipeline implements a 4-step process:
1. User Input - Accept natural language request for file/CLI operations
2. Context Loading - Load file listings and sample data
3. Code Generation - Generate Bash code for data processing workflows
4. Code Execution - Execute the code safely using bash_executor()

Example tasks:
- Find and process files matching patterns
- Transform and reorganize directory structures
- Extract and aggregate data from multiple files
- Generate reports from log files or data files
- Search and filter data across multiple files
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import mellea
from mellea.stdlib.requirements.python_tools import (
    PythonCodeExtraction,
    PythonSyntaxValid,
)
from mellea.stdlib.sampling import ModelFriendlyRepairStrategy
from mellea.stdlib.tools.shell import bash_executor


def _extract_bash_code_from_output(generated: str) -> str | None:
    """Extract Bash code block from model output.

    Looks for code blocks marked with ```bash or ``` markers.
    Falls back to returning entire output if it looks like shell code.

    Args:
        generated: Raw model output string.

    Returns:
        Extracted Bash code string, or None if extraction failed.
    """
    lines = generated.split("\n")
    in_code_block = False
    code_lines = []

    for line in lines:
        if line.strip().startswith("```bash"):
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

    # Fallback: if output looks like shell code, return it
    if any(
        line.strip().startswith(p)
        for line in lines
        for p in ["find ", "grep ", "awk ", "sed ", "cut ", "cat ", "sort "]
    ):
        return generated.strip()

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

    prompt = f"""You are a Bash command generator for file and directory operations.

User request:
{sanitized_request}

Current workspace context:
{workspace_context}

Generate a single Bash command that:
1. Performs the requested file/directory operations and transformations
2. Uses standard CLI tools (find, grep, awk, sed, cut, sort, cat, ls, etc.)
3. Processes files and data according to the user request
4. Outputs results to stdout

The command will be executed in: {workspace_dir}

IMPORTANT CONSTRAINTS:
- Generate ONLY valid Bash code in a code block
- Use simple, single commands - NO pipes (|), redirects (>, >>), or semicolons (;)
- Each command should be self-contained and executable
- Use grep, find, cat, awk, sort, uniq, wc, head, tail, cut commands
- Do NOT use complex shell syntax or chaining

Examples of valid commands:
```bash
find logs -name "*.log"
```
```bash
grep ERROR logs/app_1.log
```
```bash
cat data/sales_2024_q1.txt
```
"""

    all_reqs = [PythonCodeExtraction(), PythonSyntaxValid()]

    strategy = ModelFriendlyRepairStrategy(loop_budget=3, requirements=all_reqs)

    print("Generating Bash code for data processing...")
    generated = m.instruct(prompt, strategy=strategy)

    if generated is None:
        print("  ✗ Model failed to generate output (requirements loop exhausted)")
        return

    generated_str = str(generated)
    if not generated_str.strip():
        print("  ✗ Model failed to generate output")
        return

    code = _extract_bash_code_from_output(generated_str)
    if code is None:
        print("  ✗ Failed to extract Bash code from model output")
        print(f"\nModel output:\n{generated_str}")
        return

    print("\nGenerated Bash code:")
    print(code)

    # Execute the generated Bash code using bash_executor
    print("\nExecuting generated code...")
    result = bash_executor(code, working_dir=workspace_dir)

    if result.skipped:
        print(f"  ✗ Command execution was skipped: {result.skip_message}")
        return

    if not result.success:
        print("  ✗ Command execution failed")
        if result.stderr:
            print(f"  Error: {result.stderr}")
        return

    print("\nExecution output:")
    if result.stdout:
        print(result.stdout)
    else:
        print("  (no output)")

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
    print("Bash Code Generation: CLI Data Processing and File Operations")
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
        print("  - List all files in the data directory")
        print("  - Count total revenue from all sales files")
        print("  - Find all products mentioned across all files")

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
        # Use predefined requests - focusing on commands that work with bash_executor constraints
        user_requests = [
            "Find all ERROR entries in the log files",
            "Find all WARN entries in the log files",
            "List all files in the logs directory",
            "Show the contents of the first data file",
            "Count lines in all log files",
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
