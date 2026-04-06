"""Entrypoint for the ``m`` command-line tool.

Wires together all CLI sub-applications into a single Typer root command: ``m serve``
(start a model-serving endpoint), ``m alora`` (train and upload LoRA/aLoRA adapters),
``m decompose`` (LLM-driven task decomposition), and ``m eval`` (test-based model
evaluation). Run ``m --help`` to see all available sub-commands.
"""

try:
    import typer
except ImportError:
    raise SystemExit(
        "The 'm' CLI requires extra dependencies. "
        'Please install them with: pip install "mellea[cli]"'
    ) from None

from cli.alora.commands import alora_app
from cli.decompose import app as decompose_app
from cli.eval.commands import eval_app
from cli.fix import fix_app
from cli.serve.app import serve

cli = typer.Typer(name="m", no_args_is_help=True)


# Add a default callback for handling the default cli description.
@cli.callback()
def callback() -> None:
    """Mellea command-line tool for LLM-powered workflows.

    Provides sub-commands for serving models (``m serve``), training and uploading
    adapters (``m alora``), decomposing tasks into subtasks (``m decompose``), and
    running test-based evaluation pipelines (``m eval``).
    """


# Typer assumes that all commands are in the same file/module.
# Use this workaround to separate out functionality. Can still be called
# as if added with @cli.command() (ie `m serve` here).
cli.command(name="serve")(serve)


# Add new subcommand groups by importing and adding with `cli.add_typer()`
# as documented: https://typer.tiangolo.com/tutorial/subcommands/add-typer/#put-them-together.
cli.add_typer(alora_app)
cli.add_typer(decompose_app)

cli.add_typer(eval_app)
cli.add_typer(fix_app)
