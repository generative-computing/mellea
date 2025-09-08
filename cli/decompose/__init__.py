import typer

# from .inference import app as inference_app
from .decompose import run

app = typer.Typer(
    name="decompose",
    no_args_is_help=True,
    help="Utility pipeline for decomposing task prompts.",
)

app.command(name="run", no_args_is_help=True)(run)
