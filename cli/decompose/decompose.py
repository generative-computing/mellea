import json
from pathlib import Path
from typing import Annotated

import typer

this_file_dir = Path(__file__).resolve().parent


def run(
    out_dir: Annotated[
        Path,
        typer.Option(help="Path to an existing directory to save the output files."),
    ],
    out_name: Annotated[
        str, typer.Option(help='Name for the output files. Defaults to "m_result"')
    ] = "m_decomp_result",
    prompt_file: Annotated[
        typer.FileText | None,
        typer.Option(help="Path to a raw text file containing a task prompt."),
    ] = None,
    input_var: Annotated[
        list[str] | None,
        typer.Option(
            help=(
                "If your task prompt needs user input data, you must pass"
                + " a descriptive variable name using this option,"
                + " so the name can be included when generating the prompt templates."
                + " You can pass this option multiple times, one for each input variable name."
                + " These names must be all uppercase, alphanumeric with words separated by underscores."
            )
        ),
    ] = None,
) -> None:
    """Run the decomposition pipeline."""
    from jinja2 import Environment, FileSystemLoader

    from . import pipeline
    from .utils import validate_filename

    environment = Environment(loader=FileSystemLoader(this_file_dir), autoescape=False)
    m_template = environment.get_template("m_decomp_result.py.jinja2")

    out_name = out_name.strip()
    assert validate_filename(out_name), (
        'Invalid file name on "out-name". Characters allowed: alphanumeric, underscore, hyphen, period, and space'
    )

    assert out_dir.exists() and out_dir.is_dir(), (
        f'Path passed in the "out-dir" is not a directory: {out_dir.as_posix()}'
    )

    if prompt_file:
        decomp_data = pipeline.decompose(
            task_prompt=prompt_file.read(), user_input_variable=input_var
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
            task_prompt=task_prompt, user_input_variable=None
        )

    with open(out_dir / f"{out_name}.json", "w") as f:
        json.dump(decomp_data, f, indent=2)

    with open(out_dir / f"{out_name}.py", "w") as f:
        f.write(m_template.render(subtasks=decomp_data["subtasks"]) + "\n")


# def decompose_old(  # type: ignore[no-untyped-def]
#     query: Annotated[
#         str | None,
#         typer.Option(help="Path to file containing one or more task queries."),
#     ] = None,
#     out_dir: Annotated[
#         str, typer.Option(help="Path to file containing one or more task queries.")
#     ] = ".",
#     dry_run: Annotated[
#         bool, typer.Option(help="Only decompose the task, skip execution.")
#     ] = False,
#     print_only: Annotated[
#         bool, typer.Option(help="Only print outputs to console, do not save any files.")
#     ] = False,
#     generate_py_files: Annotated[
#         bool, typer.Option(help="Save M program files in the out_dir under m_programs/")
#     ] = False,
#     model_id: Annotated[
#         str | None,
#         typer.Option(
#             help="If set, overrides both decomposer_model_id and executor_model_id."
#         ),
#     ] = None,
#     decomposer_model_id: Annotated[
#         str | None,
#         typer.Option(
#             "-dm",
#             help="Model ID to use for decomposer backend session. Is overridden by `model_id` if set",
#         ),
#     ] = None,
#     executor_model_id: Annotated[
#         str | None,
#         typer.Option(
#             "-em",
#             help="Model ID to use for executor backend session. Is overridden by `model_id` if set",
#         ),
#     ] = None,
#     backend_type: Annotated[
#         str | None,
#         typer.Option(
#             help="If set, overrides both decomposer_backend_type and executor_backend_type."
#         ),
#     ] = None,
#     decomposer_backend_type: Annotated[
#         str | None,
#         typer.Option(
#             help="Backend type for decomposer session (e.g., huggingface, ollama, vllm)."
#         ),
#     ] = "ollama",
#     executor_backend_type: Annotated[
#         str | None,
#         typer.Option(
#             help="Backend type for executor session (e.g., huggingface, ollama, vllm)."
#         ),
#     ] = "ollama",
# ):
#     """Run the M prompt decomposition pipeline. Uses `mistral-small:latest` running on Ollama.

#     If no `QUERY` value is provided, the command will prompt for input from stdin.
#     """

#     # Import here so that imports (especially torch) don't slow down other cli commands and during cli --help.
#     from .utils import create_model, generate_python_template, run_pipeline

#     # If model_id is set, override both decomposer_model_id and executor_model_id
#     if model_id is not None:
#         decomposer_model_id = model_id
#         executor_model_id = model_id

#     # If backend_type is set, override both decomposer_backend_type and executor_backend_type
#     if backend_type is not None:
#         decomposer_backend_type = backend_type
#         executor_backend_type = backend_type

#     decompose_session = create_model(
#         model_id=decomposer_model_id,
#         backend_type=decomposer_backend_type,  # type: ignore
#     )
#     execute_session = create_model(
#         model_id=executor_model_id,
#         backend_type=executor_backend_type,  # type: ignore
#     )

#     all_results = []

#     if query:
#         try:
#             with open(query) as f:
#                 content = f.read()
#             task_sections = content.split("# Task")[1:]
#             tasks = [section.strip() for section in task_sections]
#             for i, task_input in enumerate(tasks):
#                 result = run_pipeline(
#                     task_input,
#                     index=i,
#                     decompose_session=decompose_session,
#                     execute_session=execute_session,
#                     out_dir=out_dir,
#                     dry_run=dry_run,
#                     print_only=print_only,
#                 )
#                 all_results.append(result)
#                 if generate_py_files:
#                     generate_python_template(
#                         subtask_data=result["executed_results"]["subtask_data"],
#                         output_dir=out_dir,
#                         index=i,
#                     )
#             if not print_only:
#                 os.makedirs(out_dir, exist_ok=True)
#                 with open(os.path.join(out_dir, "combined_results.json"), "w") as f:
#                     json.dump(all_results, f, indent=2)
#                 print(
#                     f"\nSaved combined results to: {os.path.join(out_dir, 'combined_results.json')}"
#                 )

#         except Exception as e:
#             print(f"Error reading query file: {e}")
#             exit(1)
#     else:
#         task_input = typer.prompt(
#             "Hi Welcome to use the M - Task Decomposition Pipeline! What can I do for you? \nUser Request: "
#         )
#         result = run_pipeline(
#             task_input,
#             index=None,
#             decompose_session=decompose_session,
#             execute_session=execute_session,
#             out_dir=out_dir,
#             dry_run=dry_run,
#             print_only=print_only,
#         )
#         if generate_py_files:
#             generate_python_template(
#                 subtask_data=result["executed_results"]["subtask_data"],
#                 output_dir=out_dir,
#             )
#         if not print_only:
#             with open(os.path.join(out_dir, "combined_results.json"), "w") as f:
#                 json.dump([result], f, indent=2)
#             print(
#                 f"\nSaved combined result to: {os.path.join(out_dir, 'combined_results.json')}"
#             )


# # Basic dry run, no file input
# m decompose --dry_run --print_only

# # Basic dry run, no file output
# m decompose --dry_run --print_only

# # Full run, only print to terminal
# m decompose --print_only

# # Normal full run with outputs
# m decompose --out_dir outputs/

# Run with generation of m programs based on the executed results
# m decompose --generate-py-file --out_dir output/
