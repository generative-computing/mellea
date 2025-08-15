"""Utils for m decomposition pipeline."""

import json
import os
import re
from typing import List  # noqa: UP035

from jinja2 import Template

from mellea import MelleaSession
from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.model_ids import IBM_GRANITE_3_3_8B, MISTRALAI_MISTRAL_SMALL_24B
from mellea.backends.ollama import OllamaModelBackend
from mellea.backends.openai import OpenAIBackend

from .task_decomposer import build_subtasks, decompose_task
from .task_executor import execute_task

# Pattern allows alphanumeric characters, underscore, hyphen, period, and space.
# Anchors ^ and $ ensure the entire string matches the pattern.
FILENAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.][a-zA-Z0-9_.\- ]+$")


def validate_filename(candidate_str: str) -> bool:
    # Check if the "filename" matches the pattern and is within a reasonable length
    # (e.g., 1 to 250 characters, a common limit that considers 5 more character for extension)
    if re.fullmatch(FILENAME_PATTERN, candidate_str) and 1 <= len(candidate_str) <= 250:
        return True
    return False


def create_model(model_id=None, backend_type="huggingface", chat_history_on=False):  # type: ignore[no-untyped-def]
    """Setup the backend model session with a model_id."""
    # Import here to avoid circular import if any
    from mellea.backends.formatter import TemplateFormatter
    from mellea.stdlib.session import LinearContext

    chat_history = LinearContext() if chat_history_on else None
    if backend_type == "huggingface":
        if model_id is None:
            model_id = IBM_GRANITE_3_3_8B.hf_model_name
        backend = LocalHFBackend(
            model_id=model_id, formatter=TemplateFormatter(model_id=model_id)
        )
        m = MelleaSession(backend, ctx=chat_history)
    elif backend_type == "ollama":
        if model_id is None:
            model_id = MISTRALAI_MISTRAL_SMALL_24B
        backend = OllamaModelBackend(model_id=model_id)
        m = MelleaSession(backend, ctx=chat_history)
    elif backend_type == "openai":
        if model_id is None:
            model_id = "mistralai/Mistral-Large-Instruct-2411"
        backend = OpenAIBackend(model_id=model_id)
        m = MelleaSession(backend, ctx=chat_history)
    else:
        raise ValueError(f"backend type is not valid: {backend_type}")
    return m


def run_pipeline(  # type: ignore[no-untyped-def]
    task,
    index=None,
    out_dir=".",
    dry_run=False,
    print_only=False,
    decompose_session=None,
    execute_session=None,
):
    """Run the full m decompose pipeline."""
    print(f"\n--- Running task {index if index is not None else ''} ---")

    print("\nDecomposing task...")
    subtask_lines, steps_and_vars, input_vars = decompose_task(
        task, m_session=decompose_session
    )

    print("\nGenerating prompts for each subtask...")
    subtasks = build_subtasks(
        task, steps_and_vars, input_vars, m_session=execute_session
    )

    for step_obj in subtasks:
        print("\n========== STEP ==========")
        print(f"STEP: {step_obj['step']}")
        print("Prompt:\n", step_obj["instruction"])
        if step_obj["requirements"]:
            print("Requirements:")
            for req in step_obj["requirements"]:
                print(f" - {req}")
        else:
            print("Requirements: N/A")

    task_data = {"task": task, "input_data": input_vars, "subtask_data": subtasks}

    suffix = f"_{index}" if index is not None else ""

    if not print_only:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"task_data{suffix}.json"), "w") as f:
            json.dump(task_data, f, indent=2)
        print(f"\nSaved task decomposition to: task_data{suffix}.json")

    if dry_run:
        print("\n[DRY RUN] Skipping execution of subtasks.")
        executed_results = {
            "final_generated_answer": "[DRY RUN MODE: Execution Skipped]"
        }
    else:
        print("\nGenerating outputs for the subtasks and the task...")
        executed_results, status = execute_task(task_data, m_session=execute_session)

        if not print_only:
            with open(
                os.path.join(out_dir, f"executed_results{suffix}.json"), "w"
            ) as f:
                json.dump(executed_results, f, indent=2)
            print(f"Saved executed task results to: executed_results{suffix}.json")

    print("\n========== FINAL RESULT ==========")
    print(executed_results.get("final_generated_answer", "[NO OUTPUT]"))

    return {
        "task_input": task,
        "task_data": task_data,
        "executed_results": executed_results,
    }


def generate_python_template(  # type: ignore[no-untyped-def]
    subtask_data: list, output_dir: str, index: int | None = None
):
    """Helper function to generate a python M program using a Jinja template."""
    python_template = Template(
        r'''
    import mellea
    m = mellea.start_session()

    {% for task in tasks%}
    task_{{loop.index}} = m.instruct("""{{ task.instruction }}""", requirements = ["{{ task.requirements|join(', ') }}"])
    {% endfor %}

    '''
    )

    python_out = python_template.render(tasks=subtask_data)
    out_file = (
        os.path.join(output_dir, f"m_program_{index}.py")
        if index
        else os.path.join(output_dir, "m_program.py")
    )
    with open(out_file, "w") as f:
        f.write(python_out)
    print(f"\nSaved python M program to: {out_file}")
