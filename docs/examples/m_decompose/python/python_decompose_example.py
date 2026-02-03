#!/usr/bin/env python3
"""
Example: Using Mellea's decompose functionality programmatically

This script demonstrates how to use the decompose pipeline from Python code
to break down a complex task into subtasks with generated prompts.
"""

import json
import subprocess
import textwrap
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

# Import the decompose pipeline from the CLI module
from cli.decompose.pipeline import DecompBackend, DecompPipelineResult, decompose


def run_decompose(task_prompt: str) -> DecompPipelineResult:
    """
    Run the decompose pipeline on a task prompt.

    Args:
        task_prompt: The task description to decompose

    Returns:
        Dictionary containing decomposition results
    """
    print("Running decomposition pipeline...\n")

    result = decompose(
        task_prompt=task_prompt,
        model_id="granite4:micro",
        backend=DecompBackend.ollama,  # Use Ollama backend
        backend_req_timeout=300,  # 5 minute timeout
    )

    return result


def save_decompose_json(
    result: DecompPipelineResult,
    output_dir: Path,
    filename: str = "python_decompose_result.json",
) -> Path:
    """
    Save decomposition results to a JSON file.

    Args:
        result: Decomposition results dictionary
        output_dir: Directory to save the file
        filename: Name of the output file

    Returns:
        Path to the saved JSON file
    """
    json_output_file = output_dir / filename

    with open(json_output_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"💾 JSON results saved to: {json_output_file}")
    return json_output_file


def generate_python_script(
    result: DecompPipelineResult,
    output_dir: Path,
    filename: str = "python_decompose_result.py",
) -> Path:
    """
    Generate an executable Python script from decomposition results.

    Args:
        result: Decomposition results dictionary
        output_dir: Directory to save the file
        filename: Name of the output Python file

    Returns:
        Path to the generated Python script
    """
    print("\n📝 Generating executable Python script...")

    # Load the template from the CLI decompose directory
    cli_decompose_dir = Path(__file__).parent.parent.parent.parent / "cli" / "decompose"
    environment = Environment(
        loader=FileSystemLoader(cli_decompose_dir), autoescape=False
    )
    m_template = environment.get_template("m_decomp_result_v1.py.jinja2")

    # Render the template with the decomposition results
    python_script_content = m_template.render(
        subtasks=result["subtasks"],
        user_inputs=[],  # No user inputs for this simple example
    )

    # Save the generated Python script
    py_output_file = output_dir / filename
    with open(py_output_file, "w") as f:
        f.write(python_script_content + "\n")

    print(f"💾 Generated Python script saved to: {py_output_file}")
    return py_output_file


def run_generated_script(
    script_path: Path, output_dir: Path, timeout: int = 600
) -> Path | None:
    """
    Execute the generated Python script to produce final output.

    Args:
        script_path: Path to the Python script to execute
        output_dir: Directory to save the final output
        timeout: Maximum execution time in seconds

    Returns:
        Path to the final output file if successful, None otherwise
    """
    print("\n🚀 Running the generated script to produce final output...")
    print("   (This may take a few minutes as it calls the LLM for each subtask)")

    try:
        result_output = subprocess.run(
            ["python3", str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=output_dir,
        )

        if result_output.returncode == 0:
            # Save the final output
            final_output_file = output_dir / "python_decompose_final_output.txt"
            with open(final_output_file, "w") as f:
                f.write(result_output.stdout)

            print(f"✅ Final output saved to: {final_output_file}")
            print("\n" + "=" * 70)
            print("Final Output:")
            print("=" * 70)
            preview = result_output.stdout
            print(preview)
            return final_output_file
        else:
            print(
                f"❌ Script execution failed with return code {result_output.returncode}"
            )
            print(f"Error: {result_output.stderr}")
            return None
    except subprocess.TimeoutExpired:
        print(f"⏱️  Script execution timed out after {timeout} seconds")
        return None
    except Exception as e:
        print(f"❌ Error running script: {e}")
        return None


def display_results(result: DecompPipelineResult):
    """
    Display decomposition results in a formatted way.

    Args:
        result: Decomposition results dictionary
    """
    print("=" * 70)
    print("Decomposition Results")
    print("=" * 70)

    print(f"\n📋 Subtasks Identified ({len(result['subtask_list'])}):")
    for i, subtask in enumerate(result["subtask_list"], 1):
        print(f"  {subtask}")

    print(f"\n🔍 Constraints Identified ({len(result['identified_constraints'])}):")
    for i, constraint in enumerate(result["identified_constraints"], 1):
        print(f"  {i}. {constraint['constraint']}")
        print(f"     Validation: {constraint['validation_strategy']}")

    print(f"\n🎯 Detailed Subtasks ({len(result['subtasks'])}):")
    for i, subtask_detail in enumerate(result["subtasks"], 1):
        print(f"\n  Subtask {subtask_detail['subtask']}")
        print(f"  Tag: {subtask_detail['tag']}")
        print(f"  Dependencies: {subtask_detail['depends_on'] or 'None'}")
        print(f"  Input Variables: {subtask_detail['input_vars_required'] or 'None'}")
        print(f"  Constraints: {len(subtask_detail['constraints'])}")


def main():
    # Define a simple task prompt to decompose
    task_prompt = textwrap.dedent("""
    Write a short blog post about the benefits of morning exercise.
    Include a catchy title, an introduction paragraph, three main benefits
    with explanations, and a conclusion that encourages readers to start
    their morning exercise routine.
    """).strip()

    print("=" * 70)
    print("Mellea Decompose Example")
    print("=" * 70)
    print(f"\nOriginal Task:\n\n{task_prompt.strip()}\n")

    # Step 1: Run decomposition
    result = run_decompose(task_prompt)

    # Step 2: Display results
    display_results(result)

    # Step 3: Save JSON results
    output_dir = Path(__file__).parent
    save_decompose_json(result, output_dir)

    # Step 4: Generate Python script
    script_path = generate_python_script(result, output_dir)

    # Step 5: Run the generated script (optional)
    run_generated_script(script_path, output_dir)

    print("\n" + "=" * 70)
    print("✅ Decomposition complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
