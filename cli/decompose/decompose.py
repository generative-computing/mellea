import json
import keyword
from enum import Enum
from graphlib import TopologicalSorter
from pathlib import Path
from typing import Annotated

import typer

from .pipeline import DecompBackend, DecompPipelineResult, DecompSubtasksResult


# Must maintain declaration order
# Newer versions must be declared on the bottom
class DecompVersion(str, Enum):
    latest = "latest"
    v1 = "v1"
    # v2 = "v2"


this_file_dir = Path(__file__).resolve().parent


def reorder_subtasks(
    subtasks: list[DecompSubtasksResult],
) -> list[DecompSubtasksResult]:
    """Reorder subtasks based on their dependencies using topological sort.

    Uses Python's graphlib.TopologicalSorter to perform a topological sort of
    subtasks based on their depends_on relationships.

    Args:
        subtasks: List of subtask dictionaries with 'tag' and 'depends_on' fields

    Returns:
        Reordered list of subtasks where all dependencies appear before dependents

    Raises:
        ValueError: If a circular dependency is detected
    """
    # Build dependency graph
    subtask_map = {subtask["tag"].lower(): subtask for subtask in subtasks}

    # Build graph for TopologicalSorter
    # Format: {node: {dependencies}}
    graph = {}
    for tag, subtask in subtask_map.items():
        deps = subtask.get("depends_on", [])
        # Filter to only include dependencies that exist in subtask_map
        valid_deps = {dep.lower() for dep in deps if dep.lower() in subtask_map}
        graph[tag] = valid_deps

    # Perform topological sort
    try:
        ts = TopologicalSorter(graph)
        sorted_tags = list(ts.static_order())
    except ValueError as e:
        # TopologicalSorter raises ValueError for circular dependencies
        raise ValueError(
            "Circular dependency detected in subtasks. Cannot automatically reorder."
        ) from e

    # Return reordered subtasks
    return [subtask_map[tag] for tag in sorted_tags]


def verify_user_variables(
    decomp_data: DecompPipelineResult, input_var: list[str] | None
) -> DecompPipelineResult:
    """Verify and fix user variable ordering in subtasks.

    Validates that:
    1. All input_vars_required exist in the provided input_var list
    2. All depends_on variables reference existing subtasks
    3. Subtasks are ordered so dependencies appear before dependents

    If dependencies are out of order, automatically reorders them using topological sort.

    Args:
        decomp_data: The decomposition pipeline result containing subtasks
        input_var: List of user-provided input variable names

    Returns:
        The decomp_data with potentially reordered subtasks

    Raises:
        ValueError: If a required input variable is missing or dependencies are invalid
    """
    if input_var is None:
        input_var = []

    # Normalize input variables to lowercase for comparison
    available_input_vars = {var.lower() for var in input_var}

    # Build set of all subtask tags
    all_subtask_tags = {subtask["tag"].lower() for subtask in decomp_data["subtasks"]}

    # Validate that all required variables exist
    for subtask in decomp_data["subtasks"]:
        subtask_tag = subtask["tag"].lower()

        # Check input_vars_required exist in provided input variables
        for required_var in subtask.get("input_vars_required", []):
            var_lower = required_var.lower()
            if var_lower not in available_input_vars:
                raise ValueError(
                    f'Subtask "{subtask_tag}" requires input variable '
                    f'"{required_var}" which was not provided in --input-var. '
                    f"Available input variables: {sorted(available_input_vars) if available_input_vars else 'none'}"
                )

        # Check that all dependencies exist somewhere in the subtasks
        for dep_var in subtask.get("depends_on", []):
            dep_lower = dep_var.lower()
            if dep_lower not in all_subtask_tags:
                raise ValueError(
                    f'Subtask "{subtask_tag}" depends on variable '
                    f'"{dep_var}" which does not exist in any subtask. '
                    f"Available subtask tags: {sorted(all_subtask_tags)}"
                )

    # Check if reordering is needed
    needs_reordering = False
    defined_subtask_tags = set()

    for subtask in decomp_data["subtasks"]:
        subtask_tag = subtask["tag"].lower()

        # Check if any dependency hasn't been defined yet
        for dep_var in subtask.get("depends_on", []):
            dep_lower = dep_var.lower()
            if dep_lower not in defined_subtask_tags:
                needs_reordering = True
                break

        if needs_reordering:
            break

        defined_subtask_tags.add(subtask_tag)

    # Reorder if needed
    if needs_reordering:
        decomp_data["subtasks"] = reorder_subtasks(decomp_data["subtasks"])

    return decomp_data


def run(
    out_dir: Annotated[
        Path,
        typer.Option(help="Path to an existing directory to save the output files."),
    ],
    out_name: Annotated[
        str, typer.Option(help='Name for the output files. Defaults to "m_result".')
    ] = "m_decomp_result",
    prompt_file: Annotated[
        typer.FileText | None,
        typer.Option(help="Path to a raw text file containing a task prompt."),
    ] = None,
    model_id: Annotated[
        str,
        typer.Option(
            help=(
                "Model name/id to be used to run the decomposition pipeline."
                + ' Defaults to "mistral-small3.2:latest", which is valid for the "ollama" backend.'
                + " If you have a vLLM instance serving a model from HF with vLLM's OpenAI"
                + " compatible endpoint, then this option should be set to the model's HF name/id,"
                + ' e.g. "mistralai/Mistral-Small-3.2-24B-Instruct-2506" and the "--backend" option'
                + ' should be set to "openai".'
            )
        ),
    ] = "mistral-small3.2:latest",
    backend: Annotated[
        DecompBackend,
        typer.Option(
            help=(
                'Backend to be used for inference. Defaults to "ollama".'
                + ' Options are: "ollama" and "openai".'
                + ' The "ollama" backend runs a local inference server.'
                + ' The "openai" backend will send inference requests to any'
                + " endpoint that's OpenAI compatible."
            ),
            case_sensitive=False,
        ),
    ] = DecompBackend.ollama,
    backend_req_timeout: Annotated[
        int,
        typer.Option(
            help='Time (in seconds) for timeout to be passed on the model inference requests. Defaults to "300"'
        ),
    ] = 300,
    backend_endpoint: Annotated[
        str | None,
        typer.Option(
            help=(
                'The "endpoint URL", sometimes called "base URL",'
                + ' to reach the model when using the "openai" backend.'
                + ' This option is required if using "--backend openai".'
            )
        ),
    ] = None,
    backend_api_key: Annotated[
        str | None,
        typer.Option(
            help=(
                'The API key for the configured "--backend-endpoint".'
                + ' If using "--backend openai" this option must be set,'
                + " even if you are running locally (an OpenAI compatible server), you"
                + ' must set this option, it can be set to "EMPTY" if your local'
                + " server doesn't need it."
            )
        ),
    ] = None,
    version: Annotated[
        DecompVersion,
        typer.Option(
            help=("Version of the mellea program generator template to be used."),
            case_sensitive=False,
        ),
    ] = DecompVersion.latest,
    input_var: Annotated[
        list[str] | None,
        typer.Option(
            help=(
                "If your task needs user input data, you must pass"
                + " a descriptive variable name using this option, this way"
                + " the variable names can be templated into the generated prompts."
                + " You can pass this option multiple times, one for each input variable name."
                + " These names must be all uppercase, alphanumeric, with words separated by underscores."
            )
        ),
    ] = None,
) -> None:
    """Runs the decomposition pipeline."""
    try:
        from jinja2 import Environment, FileSystemLoader

        from . import pipeline
        from .utils import validate_filename

        environment = Environment(
            loader=FileSystemLoader(this_file_dir), autoescape=False
        )

        ver = (
            list(DecompVersion)[-1].value
            if version == DecompVersion.latest
            else version.value
        )
        m_template = environment.get_template(f"m_decomp_result_{ver}.py.jinja2")

        out_name = out_name.strip()
        assert validate_filename(out_name), (
            'Invalid file name on "out-name". Characters allowed: alphanumeric, underscore, hyphen, period, and space'
        )

        assert out_dir.exists() and out_dir.is_dir(), (
            f'Path passed in the "out-dir" is not a directory: {out_dir.as_posix()}'
        )

        if input_var is not None and len(input_var) > 0:
            assert all(
                var.isidentifier() and not keyword.iskeyword(var) for var in input_var
            ), (
                'One or more of the "input-var" are not valid. The input variables\' names must be a valid Python identifier'
            )

        if prompt_file:
            decomp_data = pipeline.decompose(
                task_prompt=prompt_file.read(),
                user_input_variable=input_var,
                model_id=model_id,
                backend=backend,
                backend_req_timeout=backend_req_timeout,
                backend_endpoint=backend_endpoint,
                backend_api_key=backend_api_key,
            )
        else:
            task_prompt: str = typer.prompt(
                (
                    "\nThis mode doesn't support tasks that need input data."
                    + '\nInput must be provided in a single line. Use "\\n" for new lines.'
                    + "\n\nInsert the task prompt to decompose"
                ),
                type=str,
            )
            task_prompt = task_prompt.replace("\\n", "\n")
            decomp_data = pipeline.decompose(
                task_prompt=task_prompt,
                user_input_variable=None,
                model_id=model_id,
                backend=backend,
                backend_req_timeout=backend_req_timeout,
                backend_endpoint=backend_endpoint,
                backend_api_key=backend_api_key,
            )

        # Verify that all user variables are properly defined before use
        # This may reorder subtasks if dependencies are out of order
        decomp_data = verify_user_variables(decomp_data, input_var)

        with open(out_dir / f"{out_name}.json", "w") as f:
            json.dump(decomp_data, f, indent=2)

        with open(out_dir / f"{out_name}.py", "w") as f:
            f.write(
                m_template.render(
                    subtasks=decomp_data["subtasks"], user_inputs=input_var
                )
                + "\n"
            )
    except Exception:
        created_json = Path(out_dir / f"{out_name}.json")
        created_py = Path(out_dir / f"{out_name}.py")

        if created_json.exists() and created_json.is_file():
            created_json.unlink()
        if created_py.exists() and created_py.is_file():
            created_py.unlink()

        raise Exception
