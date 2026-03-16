"""Typer sub-application for the `m decompose` command group.

Exposes a single `run` command that takes a task prompt (from a file or
interactively), calls the LLM-based decomposition pipeline to break it into
structured subtasks with constraints and dependency ordering, and writes the results
as a JSON data file and a ready-to-run Python script. Invoke via
`m decompose run --help` for full option documentation.
"""

import typer

# from .inference import app as inference_app
from .decompose import run

app = typer.Typer(
    name="decompose",
    no_args_is_help=True,
    help="Utility pipeline for decomposing task prompts.",
)

app.command(name="run", no_args_is_help=True)(run)
