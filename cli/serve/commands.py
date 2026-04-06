"""Typer command definition for ``m serve``.

Separates the CLI interface (typer annotations) from the server implementation
(FastAPI, uvicorn) so that ``m --help`` works without the ``server`` extra installed.
The heavy server dependencies are only imported when ``m serve`` is actually invoked.
"""

import typer


def serve(
    script_path: str = typer.Argument(
        default="docs/examples/m_serve/example.py",
        help="Path to the Python script to import and serve",
    ),
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8080, help="Port to bind to"),
):
    """Serve a FastAPI endpoint for a given script."""
    from cli.serve.app import run_server

    run_server(script_path=script_path, host=host, port=port)
