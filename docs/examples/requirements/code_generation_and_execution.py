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
import sys
import tempfile
from pathlib import Path

import mellea
from mellea.stdlib.components import Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements.python_reqs import PythonExecutionReq
from mellea.stdlib.requirements.python_tools import (
    NoImportRestrictions,
    PythonCodeExtraction,
    PythonSyntaxValid,
)
from mellea.stdlib.sampling import ModelFriendlyRepairStrategy
from mellea.stdlib.tools.execution_policy import CapabilityPolicy

try:
    from mellea.stdlib.requirements.plotting import (
        MatplotlibHeadlessBackend,
        PlotFileSaved,
    )
except ImportError as e:
    raise ImportError(
        "The code_generation_and_execution example requires matplotlib and numpy. "
        "Install them with: `uv pip install matplotlib numpy`"
    ) from e


def load_csv_data(csv_path: str) -> tuple[list[dict], str]:
    """Load CSV file and return data with preview.

    Args:
        csv_path: Path to CSV file

    Returns:
        tuple of (list of dicts, CSV preview string)
    """
    preview_lines = []
    data = []
    with open(csv_path) as f:
        for i, line in enumerate(f):
            if i < 5:
                preview_lines.append(line.rstrip())
        f.seek(0)
        reader = csv.DictReader(f)
        data = list(reader)

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
    # Import _has_python_code_listing locally to avoid importing private functions at module level.
    # _has_python_code_listing is a private API (leading underscore), so importing it at the top
    # would blur the import contract, make it look like a public API, and obscure that it's an
    # implementation detail. Local import makes the dependency on internals explicit and localized.
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


# See README.md "How Code Execution Works" section for details on the execute_python_code
# mechanism that PythonExecutionReq uses under the hood.


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

    output_path = str(Path(output_dir) / f"graph_{request_number}.png")

    # Quote and escape user input and CSV preview to reduce accidental prompt
    # formatting breakage. repr() wraps strings in quotes and escapes special
    # characters, preventing markdown or code-block markers from being interpreted.
    sanitized_request = repr(user_request)
    sanitized_preview = repr(csv_preview)

    prompt = f"""
    The user has this request:
    {sanitized_request}

    This request specifies BOTH:
    1. What data to extract from the CSV
    2. How to visualize that extracted data as a graph

    CSV file path: {csv_path}
    CSV preview:
    {sanitized_preview}

    Write Python code to:
    1. Load the CSV file using csv module or pandas
    2. Extract the data as specified in the user request
    3. Use matplotlib with headless backend (set matplotlib.use('Agg') at start)
    4. Create the visualization (graph type) specified by the user
    5. Save the graph to {output_path} using plt.savefig('{output_path}')
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
    generated = m.instruct(prompt, strategy=strategy)

    if generated is None:
        print("  ✗ Model failed to generate output (requirements loop exhausted)")
        return

    generated_str = str(generated)
    if not generated_str.strip():
        print("  ✗ Model failed to generate output")
        return

    code = _extract_code_from_output(generated_str)
    if code is None:
        print("  ✗ Failed to extract Python code from model output")
        print(f"\nModel output:\n{generated_str}")
        return

    print("\nGenerated code:")
    print(code)

    # Code execution already occurred as a side effect of PythonExecutionReq validation.
    # See README.md for explanation of how execution works.

    # Check if graph file was created
    graph_path = Path(output_path)
    if not graph_path.exists():
        print(f"  ✗ Graph was not created at {graph_path}")
        return
    print(f"\n  ✓ Graph saved to: {graph_path}")
    print(f"    File size: {graph_path.stat().st_size} bytes")


def main():
    """Demonstrate complete pipeline: accept user input and generate graphs."""
    parser = argparse.ArgumentParser(
        description="Code Generation: Extract Data and Generate Graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default sample data and predefined requests
  uv run python code_generation_and_execution.py

  # Use a custom CSV file
  uv run python code_generation_and_execution.py --csv /path/to/data.csv

  # Accept user input interactively
  uv run python code_generation_and_execution.py --interactive

  # Both custom CSV and interactive mode
  uv run python code_generation_and_execution.py --csv /path/to/data.csv --interactive
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
