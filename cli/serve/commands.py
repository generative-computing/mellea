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
    """Serve a Mellea program as an OpenAI-compatible HTTP endpoint.

    Loads a Python file containing a ``serve`` function and exposes it
    via a FastAPI server implementing the OpenAI chat completions API. The server
    accepts ``POST /v1/chat/completions`` requests.

    Prerequisites:
        Mellea installed with server dependency group (``uv add 'mellea[server]'``).
        The python file being loaded must have a ``serve`` function.

    Output:
        Starts a long-running HTTP server on the specified host and port.
        The ``/v1/chat/completions`` endpoint accepts OpenAI-format chat
        completion requests and returns ``ChatCompletion`` JSON responses.

    Examples:
        m serve my_app.py --port 9000

    See Also:
        guide: integrations/m-serve
    """

    from cli.serve.app import run_server

    run_server(script_path=script_path, host=host, port=port)
