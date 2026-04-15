# decompose/decompose.py
"""Implementation of the ``m decompose run`` CLI command.

Accepts a task prompt (from a text file or interactive input), calls the multi-step
LLM decomposition pipeline to produce a structured list of subtasks each with
constraints and inter-subtask dependencies, then validates and topologically reorders
the subtasks before writing a JSON result file and a rendered Python script to the
specified output directory.
"""

import json
import keyword
import re
import shutil
from enum import StrEnum
from graphlib import TopologicalSorter
from pathlib import Path
from typing import Annotated

import typer
from jinja2 import Environment, FileSystemLoader

from . import pipeline
from .logging import LogMode, configure_logging, get_logger, log_section
from .pipeline import DecompBackend, DecompPipelineResult, DecompSubtasksResult
from .utils import validate_filename


class DecompVersion(StrEnum):
    """Available template versions for generated decomposition programs.

    Newer concrete versions must be declared after older ones so that
    ``latest`` can resolve to the most recently declared template version.

    Attributes:
        latest: Sentinel value that resolves to the last declared concrete
            template version.
        v1: Version 1 of the decomposition program template.
        v2: Version 2 of the decomposition program template.
    """

    latest = "latest"
    v1 = "v1"
    v2 = "v2"
    v3 = "v3"


this_file_dir = Path(__file__).resolve().parent


def reorder_subtasks(
    subtasks: list[DecompSubtasksResult],
) -> list[DecompSubtasksResult]:
    """Topologically sort subtasks by their ``depends_on`` relationships.

    Args:
        subtasks: List of subtask dicts, each with a ``"tag"`` and optional
            ``"depends_on"`` field.

    Returns:
        list[DecompSubtasksResult]: The subtasks reordered so that dependencies
        come before dependents, with numbering prefixes updated.

    Raises:
        ValueError: If a circular dependency is detected.
    """
    subtask_map = {subtask["tag"].lower(): subtask for subtask in subtasks}

    graph = {}
    for tag, subtask in subtask_map.items():
        deps = subtask.get("depends_on", [])
        valid_deps = {dep.lower() for dep in deps if dep.lower() in subtask_map}
        graph[tag] = valid_deps

    try:
        ts = TopologicalSorter(graph)
        sorted_tags = list(ts.static_order())
    except ValueError as e:
        raise ValueError(
            "Circular dependency detected in subtasks. Cannot automatically reorder."
        ) from e

    reordered = [subtask_map[tag] for tag in sorted_tags]

    number_pattern = re.compile(r"^\d+\.\s+")
    for i, subtask in enumerate(reordered, start=1):
        if number_pattern.match(subtask["subtask"]):
            subtask["subtask"] = number_pattern.sub(f"{i}. ", subtask["subtask"])

    return reordered


def verify_user_variables(
    decomp_data: DecompPipelineResult, input_var: list[str] | None
) -> DecompPipelineResult:
    """Validate that all required input variables and dependencies exist.

    Args:
        decomp_data: The decomposition pipeline result containing subtasks.
        input_var: User-provided input variable names, or ``None`` for none.

    Returns:
        DecompPipelineResult: The (possibly reordered) decomposition data.

    Raises:
        ValueError: If a subtask requires an input variable that was not
            provided, or depends on a subtask tag that does not exist.
    """
    if input_var is None:
        input_var = []

    available_input_vars = {var.lower() for var in input_var}
    all_subtask_tags = {subtask["tag"].lower() for subtask in decomp_data["subtasks"]}

    for subtask in decomp_data["subtasks"]:
        subtask_tag = subtask["tag"].lower()

        for required_var in subtask.get("input_vars_required", []):
            var_lower = required_var.lower()
            if var_lower not in available_input_vars:
                raise ValueError(
                    f'Subtask "{subtask_tag}" requires input variable '
                    f'"{required_var}" which was not provided in --input-var. '
                    f"Available input variables: {sorted(available_input_vars) if available_input_vars else 'none'}"
                )

        for dep_var in subtask.get("depends_on", []):
            dep_lower = dep_var.lower()
            if dep_lower not in all_subtask_tags:
                raise ValueError(
                    f'Subtask "{subtask_tag}" depends on variable '
                    f'"{dep_var}" which does not exist in any subtask. '
                    f"Available subtask tags: {sorted(all_subtask_tags)}"
                )

    needs_reordering = False
    defined_subtask_tags = set()

    for subtask in decomp_data["subtasks"]:
        subtask_tag = subtask["tag"].lower()

        for dep_var in subtask.get("depends_on", []):
            dep_lower = dep_var.lower()
            if dep_lower not in defined_subtask_tags:
                needs_reordering = True
                break

        if needs_reordering:
            break

        defined_subtask_tags.add(subtask_tag)

    if needs_reordering:
        decomp_data["subtasks"] = reorder_subtasks(decomp_data["subtasks"])

    return decomp_data


def run(
    out_dir: Annotated[
        Path,
        typer.Option(help="Path to an existing directory to save the output files."),
    ],
    out_name: Annotated[
        str,
        typer.Option(help='Name for the output files. Defaults to "m_decomp_result".'),
    ] = "m_decomp_result",
    input_file: Annotated[
        str | None, typer.Option(help="Path to a text file containing user queries.")
    ] = None,
    model_id: Annotated[
        str,
        typer.Option(
            help=(
                "Model name/id used to run the decomposition pipeline. "
                'Defaults to "mistral-small3.2:latest", valid for the "ollama" backend.'
            )
        ),
    ] = "mistral-small3.2:latest",
    backend: Annotated[
        DecompBackend,
        typer.Option(
            help=('Backend used for inference. Options: "ollama", "openai".'),
            case_sensitive=False,
        ),
    ] = DecompBackend.ollama,
    backend_req_timeout: Annotated[
        int,
        typer.Option(
            help='Timeout in seconds for backend requests. Defaults to "300".'
        ),
    ] = 300,
    backend_endpoint: Annotated[
        str | None,
        typer.Option(help='Backend endpoint / base URL. Required for "openai".'),
    ] = None,
    backend_api_key: Annotated[
        str | None, typer.Option(help='Backend API key. Required for "openai".')
    ] = None,
    version: Annotated[
        DecompVersion,
        typer.Option(
            help="Version of the mellea program generator template to use.",
            case_sensitive=False,
        ),
    ] = DecompVersion.latest,
    input_var: Annotated[
        list[str] | None,
        typer.Option(
            help=(
                "Optional user input variable names. "
                "You may pass this option multiple times. "
                "Each value must be a valid Python identifier."
            )
        ),
    ] = None,
    log_mode: Annotated[
        LogMode,
        typer.Option(
            help='Readable logging mode. Options: "demo" or "debug".',
            case_sensitive=False,
        ),
    ] = LogMode.demo,
    enable_script_run: Annotated[
        bool,
        typer.Option(
            help=(
                "When true, generated scripts expose argparse runtime options "
                "for backend, model, endpoint, and API key overrides."
            )
        ),
    ] = False,
) -> None:
    """Runs the ``m decompose`` CLI workflow and writes generated outputs.

    Reads user queries from a file or interactive input, runs the decomposition
    pipeline for each task job, and writes one JSON file, one rendered Python
    program, and any generated validation modules under a per-job output
    directory.

    Args:
        out_dir: Existing directory under which per-job output directories are
            created.
        out_name: Base name used for the per-job output directory and generated
            files.
        input_file: Optional path to a text file containing one or more task
            prompts. Each non-empty line is processed as a separate task job.
            When omitted, the command prompts interactively for one task.
        model_id: Model identifier used for all decomposition pipeline stages.
        backend: Inference backend used to execute model calls.
        backend_req_timeout: Request timeout in seconds for backend inference calls.
        backend_endpoint: Endpoint URL or base URL required by remote backends.
        backend_api_key: API key required by remote backends.
        version: Template version used to render the generated Python program.
            ``latest`` resolves to the most recently declared concrete version.
        input_var: Optional user input variable names to expose in generated
            prompts and programs. Each name must be a valid non-keyword Python
            identifier.
        log_mode: Logging verbosity for CLI and pipeline execution.
        enable_script_run: Whether generated scripts should expose argparse
            runtime options. Defaults to ``False``.

    Raises:
        AssertionError: If ``out_name`` is invalid, ``out_dir`` does not name an
            existing directory, ``input_file`` does not name an existing file,
            or any declared ``input_var`` is not a valid Python identifier.
        ValueError: If ``input_file`` exists but contains no non-empty task
            lines.
        Exception: Propagates pipeline, rendering, parsing, or file-writing
            failures. Any output directories created earlier in the run are
            removed before the exception is re-raised.
    """
    created_dirs: list[Path] = []

    try:
        configure_logging(log_mode)
        logger = get_logger("m_decompose.cli")

        log_section(logger, "m_decompose cli")
        logger.info("out_dir        : %s", out_dir)
        logger.info("out_name       : %s", out_name)
        logger.info("backend        : %s", backend.value)
        logger.info("model_id       : %s", model_id)
        logger.info("version        : %s", version.value)
        logger.info("log_mode       : %s", log_mode.value)
        logger.info("script options : %s", enable_script_run)
        logger.info("input_vars     : %s", input_var or "[]")

        environment = Environment(
            loader=FileSystemLoader(this_file_dir), autoescape=False
        )

        ver = (
            list(DecompVersion)[-1].value
            if version == DecompVersion.latest
            else version.value
        )
        logger.info("resolved version: %s", ver)

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
                'One or more of the "input-var" are not valid. '
                "Each input variable name must be a valid Python identifier."
            )

        log_section(logger, "load task prompt")

        if input_file:
            input_path = Path(input_file)
            assert input_path.exists() and input_path.is_file(), (
                f'Path passed in "input-file" is not a file: {input_path.as_posix()}'
            )

            raw_lines = input_path.read_text(encoding="utf-8").splitlines()
            task_jobs = [line.strip() for line in raw_lines if line.strip()]
            user_input_variable = input_var

            logger.info("prompt source  : file")
            logger.info("input_file     : %s", input_path)
            logger.info("task jobs      : %d", len(task_jobs))

            if not task_jobs:
                raise ValueError("Input file contains no non-empty task lines.")
        else:
            task_prompt = typer.prompt(
                (
                    "\nThis mode doesn't support tasks that need input data."
                    + '\nInput must be provided in a single line. Use "\\n" for new lines.'
                    + "\n\nInsert the task prompt to decompose"
                ),
                type=str,
            )
            task_prompt = task_prompt.replace("\\n", "\n")
            task_jobs = [task_prompt]
            user_input_variable = None

            logger.info("prompt source  : interactive")
            logger.info("task jobs      : 1")

        for job_idx, task_prompt in enumerate(task_jobs, start=1):
            job_out_name = out_name if len(task_jobs) == 1 else f"{out_name}_{job_idx}"

            log_section(logger, f"run pipeline job {job_idx}/{len(task_jobs)}")
            logger.info("job out_name   : %s", job_out_name)
            logger.info("prompt length  : %d", len(task_prompt))
            logger.info("task prompt    : %s", task_prompt)

            decomp_data = pipeline.decompose(
                task_prompt=task_prompt,
                user_input_variable=user_input_variable,
                model_id=model_id,
                backend=backend,
                backend_req_timeout=backend_req_timeout,
                backend_endpoint=backend_endpoint,
                backend_api_key=backend_api_key,
                log_mode=log_mode,
            )

            # TODO: verify_user_variables
            # logger.info("verify_user_variables: skipped")

            log_section(logger, f"write outputs job {job_idx}/{len(task_jobs)}")

            decomp_dir = out_dir / job_out_name
            val_fn_dir = decomp_dir / "validations"

            logger.info("creating output dir: %s", decomp_dir)
            decomp_dir.mkdir(parents=True, exist_ok=False)
            created_dirs.append(decomp_dir)

            val_fn_dir.mkdir(exist_ok=True)
            (val_fn_dir / "__init__.py").touch()

            val_fn_count = 0
            for constraint in decomp_data["identified_constraints"]:
                if constraint["val_fn"] is not None:
                    val_fn_count += 1
                    with open(val_fn_dir / f"{constraint['val_fn_name']}.py", "w") as f:
                        f.write(constraint["val_fn"] + "\n")

            with open(decomp_dir / f"{job_out_name}.json", "w") as f:
                json.dump(decomp_data, f, indent=2)

            with open(decomp_dir / f"{job_out_name}.py", "w") as f:
                f.write(
                    m_template.render(
                        subtasks=decomp_data["subtasks"],
                        user_inputs=input_var,
                        identified_constraints=decomp_data["identified_constraints"],
                        model_id=model_id,
                        backend=backend.value,
                        backend_endpoint=backend_endpoint,
                        backend_api_key=backend_api_key,
                        enable_script_run=enable_script_run,
                    )
                    + "\n"
                )

            logger.info("json written    : %s", decomp_dir / f"{job_out_name}.json")
            logger.info("program written : %s", decomp_dir / f"{job_out_name}.py")
            logger.info("validation files: %d", val_fn_count)

        logger.info("")
        logger.info("m_decompose CLI completed successfully")

    except Exception:
        for decomp_dir in reversed(created_dirs):
            if decomp_dir.exists() and decomp_dir.is_dir():
                shutil.rmtree(decomp_dir)
        raise
