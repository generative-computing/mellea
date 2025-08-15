import json
import keyword
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
    """Runs the decomposition pipeline."""
    try:
        from jinja2 import Environment, FileSystemLoader

        from . import pipeline
        from .utils import validate_filename

        environment = Environment(
            loader=FileSystemLoader(this_file_dir), autoescape=False
        )
        m_template = environment.get_template("m_decomp_result.py.jinja2")

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
