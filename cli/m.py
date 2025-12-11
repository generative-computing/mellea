"""Entrypoint for the M CLI."""

import typer

from cli.alora.commands import alora_app
from cli.decompose import app as decompose_app
from cli.serve.app import serve
from cli.eval.commands import eval_app

cli = typer.Typer(name="m", no_args_is_help=True)


# Add a default callback for handling the default cli description.
@cli.callback()
def callback() -> None:
    """Perform M Tasks."""


# Typer assumes that all commands are in the same file/module.
# Use this workaround to separate out functionality. Can still be called
# as if added with @cli.command() (ie `m serve` here).
cli.command(name="serve")(serve)


# Add new subcommand groups by importing and adding with `cli.add_typer()`
# as documented: https://typer.tiangolo.com/tutorial/subcommands/add-typer/#put-them-together.
cli.add_typer(alora_app)
cli.add_typer(decompose_app)

cli.add_typer(eval_app)
