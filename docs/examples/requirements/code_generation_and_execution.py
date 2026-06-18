# pytest: e2e, ollama, qualitative, slow
"""Example demonstrating code generation, data extraction, and graph file generation.

This example shows how to use Mellea to:
1. Accept user input specifying what data to extract AND how to visualize it
2. Read a CSV file
3. Generate Python code to extract and visualize data in one step
4. Execute the code to generate and save the graph to a file using headless matplotlib

The pipeline implements a 4-step process:
1. User Input - Accept natural language request specifying data extraction AND visualization
2. CSV Loading - Read data from a CSV file
3. Code Generation - Generate code to extract data and create visualization using headless matplotlib
4. Code Execution - Execute the code to generate and save graph to file (no display)
"""

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import mellea
from mellea.stdlib.components import Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements.plotting import MatplotlibHeadlessBackend, PlotFileSaved
from mellea.stdlib.requirements.python_reqs import PythonExecutionReq
from mellea.stdlib.requirements.python_tools import (
    NoImportRestrictions,
    PythonCodeExtraction,
    PythonSyntaxValid,
)
from mellea.stdlib.sampling import ModelFriendlyRepairStrategy
from mellea.stdlib.tools.execution_policy import CapabilityPolicy


def load_csv_data(csv_path: str) -> tuple[list[dict], str]:
    """Load CSV file and return data with preview.

    Args:
        csv_path: Path to CSV file

    Returns:
        tuple of (list of dicts, CSV preview string)
    """
    data = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        data = list(reader)

    preview_lines = []
    with open(csv_path) as f:
        for i, line in enumerate(f):
            if i < 5:
                preview_lines.append(line.rstrip())

    preview = "\n".join(preview_lines)
    return data, preview


def _extract_code_from_output(generated: str) -> str | None:
    """Extract the highest-scoring Python code block from model output.

    Uses PythonCodeExtraction to intelligently extract code blocks, scoring
    them by length, complexity, and content to prioritize the main implementation.
    Falls back to returning entire output if it looks like unwrapped Python code.

    Args:
        generated: Raw model output string.

    Returns:
        Extracted Python code string, or None if extraction failed.
    """
    from mellea.stdlib.requirements.python_reqs import _has_python_code_listing

    ctx = ChatContext().add(Message("assistant", generated))
    result = _has_python_code_listing(ctx)
    if result.as_bool():
        return result.reason

    # Fallback: if extraction failed but output looks like Python code,
    # the LLM may have generated code without backtick markers
    if any(
        line.strip().startswith(p)
        for line in generated.split("\n")
        for p in ["import ", "from ", "def ", "class ", "plt.", "pandas"]
    ):
        return generated.strip()

    return None


# NOTE: Code execution has already occurred as a side effect of requirement validation.
# When m.instruct() is called with PythonExecutionReq, the validation_fn executes the
# code in a subprocess to verify it runs without errors. The code execution below is
# documented for educational purposes to show what happens during validation, but is
# not needed since validation already ran the code. To verify the graph was created,
# we only need to check if the output file exists (see process_user_request line ~310).
#
# def execute_python_code(code: str, timeout: int = 10) -> dict:
#     """Execute Python code in a subprocess and capture output.
#
#     Args:
#         code: Python code to execute
#         timeout: Maximum execution time in seconds
#
#     Returns:
#         dict with 'success', 'output', and 'error' keys
#     """
#     try:
#         with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
#             f.write(code)
#             temp_file = f.name
#
#         try:
#             result = subprocess.run(
#                 [sys.executable, temp_file],
#                 capture_output=True,
#                 text=True,
#                 timeout=timeout,
#             )
#
#             return {
#                 "success": result.returncode == 0,
#                 "output": result.stdout,
#                 "error": result.stderr,
#                 "return_code": result.returncode,
#             }
#         finally:
#             Path(temp_file).unlink()
#
#     except subprocess.TimeoutExpired:
#         return {
#             "success": False,
#             "output": "",
#             "error": f"Code execution timed out after {timeout} seconds",
#             "return_code": -1,
#         }
#     except Exception as e:
#         return {"success": False, "output": "", "error": str(e), "return_code": -1}


def create_sample_csv(csv_path: str) -> None:
    """Create a sample CSV file with 100 employees across multiple work locations."""
    locations = [
        "New York",
        "San Francisco",
        "Chicago",
        "Austin",
        "Boston",
        "Seattle",
        "Denver",
        "Miami",
        "Portland",
        "Atlanta",
    ]
    departments = ["Engineering", "Sales", "HR", "Marketing", "Finance"]
    names = [
        "Alice",
        "Bob",
        "Charlie",
        "Diana",
        "Eve",
        "Frank",
        "Grace",
        "Henry",
        "Iris",
        "Jack",
        "Karen",
        "Leo",
        "Maya",
        "Nathan",
        "Olivia",
        "Paul",
        "Quinn",
        "Rachel",
        "Sam",
        "Tina",
        "Uma",
        "Victor",
        "Wendy",
        "Xavier",
        "Yara",
        "Zoe",
        "Adam",
        "Bella",
        "Carl",
        "Dana",
        "Ethan",
        "Fiona",
        "George",
        "Hannah",
        "Ian",
        "Julia",
        "Kevin",
        "Laura",
        "Mike",
        "Nancy",
        "Oscar",
        "Piper",
        "Rosa",
        "Steve",
        "Tara",
        "Ulysses",
        "Vera",
        "Wade",
        "Xena",
    ]

    with open(csv_path, "w") as f:
        f.write("name,age,salary,department,years_experience,work_location\n")

        for i in range(100):
            name = (
                names[i % len(names)] + str(i // len(names) + 1)
                if i >= len(names)
                else names[i]
            )
            age = 24 + (i % 40)
            salary = 40000 + (i % 60) * 1000
            department = departments[i % len(departments)]
            years = i % 15
            location = locations[i % len(locations)]

            f.write(f"{name},{age},{salary},{department},{years},{location}\n")


def process_user_request(
    user_request: str,
    csv_path: str,
    csv_preview: str,
    output_dir: str,
    request_number: int = 1,
) -> None:
    """Process a user request to extract data and generate a graph.

    The user input specifies both:
    - What data to extract
    - How to visualize the extracted data

    Args:
        user_request: Natural language request specifying data extraction and visualization
        csv_path: Path to CSV file
        csv_preview: Preview of CSV file contents
        output_dir: Directory to save generated graph
        request_number: Number for naming output file
    """
    print(f"\nUser request: {user_request}")
    print("-" * 70)

    m = mellea.start_session()

    output_path = f"{output_dir}/graph_{request_number}.png"

    prompt = f"""
    The user has this request:
    "{user_request}"

    This request specifies BOTH:
    1. What data to extract from the CSV
    2. How to visualize that extracted data as a graph

    CSV file path: {csv_path}
    CSV preview:
    {csv_preview}

    Write Python code to:
    1. Load the CSV file using csv module or pandas
    2. Extract the data as specified in the user request
    3. Use matplotlib with headless backend (set matplotlib.use('Agg') at start)
    4. Create the visualization (graph type) specified by the user
    5. Save the graph to {output_path} using plt.savefig()
    6. Do NOT call plt.show() - only save to file
    7. Print a message indicating success

    The visualization should have:
    - Clear and descriptive title
    - Properly labeled axes
    - Legend if applicable
    - Appropriate figure size (e.g., figsize=(10, 6))

    Generate only valid Python code without explanations.
    """

    all_reqs = [
        PythonCodeExtraction(),
        PythonSyntaxValid(),
        PythonExecutionReq(
            execution_tier="local",
            policy=CapabilityPolicy(timeout=20),
            max_output_chars=10_000,
        ),
        NoImportRestrictions(),
        MatplotlibHeadlessBackend(),
        PlotFileSaved(output_path=output_path),
    ]

    strategy = ModelFriendlyRepairStrategy(loop_budget=5, requirements=all_reqs)

    print("Generating code to extract data and create graph...")
    generated = m.instruct(
        prompt,
        requirements=all_reqs,  # type: ignore[arg-type]
        strategy=strategy,
    )
    generated_str = str(generated)
    code = _extract_code_from_output(generated_str)
    if code is None:
        print("  ✗ Failed to extract Python code from model output")
        print(f"\nModel output:\n{generated_str}")
        return

    print("\nGenerated code:")
    print(code)

    # NOTE: Code execution already occurred as a side effect of PythonExecutionReq
    # validation during m.instruct(). Because PythonExecutionReq uses execution_tier="local"
    # (not "static"), the code was executed in a subprocess during validation. The lines
    # below are kept as comments to show what would happen if we were executing separately:
    #
    # print("\nExecuting code...")
    # result = execute_python_code(code, timeout=15)
    #
    # if not result["success"]:
    #     print(f"  ✗ Error during execution: {result['error']}")
    #     return
    #
    # print("  ✓ Code executed successfully")
    # if result["output"]:
    #     print(f"\n  Output: {result['output']}")

    # Check if graph file was created
    graph_path = Path(output_path)
    if graph_path.exists():
        print(f"\n  ✓ Graph saved to: {graph_path}")
        print(f"    File size: {graph_path.stat().st_size} bytes")
    else:
        print(f"\n  ⚠ Graph file not found at {graph_path}")


def main():
    """Demonstrate complete pipeline: accept user input and generate graphs."""
    parser = argparse.ArgumentParser(
        description="Code Generation: Extract Data and Generate Graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default sample data and predefined requests
  python code_generation_and_execution.py

  # Use a custom CSV file
  python code_generation_and_execution.py --csv /path/to/data.csv

  # Accept user input interactively
  python code_generation_and_execution.py --interactive

  # Both custom CSV and interactive mode
  python code_generation_and_execution.py --csv /path/to/data.csv --interactive
        """,
    )
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV file")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Accept user input interactively instead of using predefined requests",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Directory to save generated graphs"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Code Generation: Extract Data and Generate Graphs")
    print("=" * 70)

    # Use provided options or defaults
    temp_dir = tempfile.gettempdir()
    csv_path = args.csv if args.csv else str(Path(temp_dir) / "sample_data.csv")
    output_dir = args.output if args.output else temp_dir

    # Create sample CSV if using default path
    if csv_path == str(Path(temp_dir) / "sample_data.csv"):
        create_sample_csv(csv_path)
        print(f"\nCreated sample CSV file: {csv_path}")
    elif not Path(csv_path).exists():
        print(f"\nError: CSV file not found: {csv_path}")
        sys.exit(1)

    # Load and display CSV preview
    data, preview = load_csv_data(csv_path)
    print(f"\nUsing CSV file: {csv_path}")
    print("\nCSV Preview:")
    print(preview)
    print(f"\nTotal rows: {len(data)}")

    # Detect columns dynamically
    if data:
        columns = list(data[0].keys())
        print(f"\nAvailable columns: {', '.join(columns)}")

    if args.interactive:
        print("\n" + "=" * 70)
        print("Interactive Mode")
        print("=" * 70)
        request_number = 1
        while True:
            print("\nEnter a request (or 'quit' to exit):")
            print(
                "Example: 'Extract average salary by department and create a bar chart'"
            )
            user_input = input("\n> ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                break

            if not user_input:
                print("Please enter a valid request.")
                continue

            process_user_request(
                user_input,
                csv_path,
                preview,
                output_dir=output_dir,
                request_number=request_number,
            )
            request_number += 1

    else:
        # Use predefined requests
        user_requests = [
            "Extract average salary by department and create a bar chart",
            "Extract employee count by work location and create a bar chart",
            "Extract age and salary for all employees and create a scatter plot",
            "Extract salary by years of experience and create a line plot",
            "Extract employee distribution across work locations and create a pie chart",
        ]

        # Process each user request
        for i, request in enumerate(user_requests, 1):
            print("\n" + "=" * 70)
            print(f"Example {i}")
            print("=" * 70)

            process_user_request(
                request, csv_path, preview, output_dir=output_dir, request_number=i
            )

    print("\n" + "=" * 70)
    print("Pipeline completed!")
    print(f"Generated graphs saved to {output_dir}/graph_*.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
