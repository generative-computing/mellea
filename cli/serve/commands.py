# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typer command definition for `m serve`.

Separates the CLI interface (typer annotations) from the server implementation
(FastAPI, uvicorn) so that `m --help` works without the `server` extra installed.
The heavy server dependencies are only imported when `m serve` is actually invoked.
"""

import typer


def serve(
    script_path: str = typer.Argument(
        default="docs/examples/m_serve/example.py",
        help="Path to the Python script to import and serve",
    ),
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8080, help="Port to bind to"),
    response_ttl: int = typer.Option(
        1800,
        help="Seconds before stored responses expire (default: 1800 = 30 minutes). "
        "Stored responses enable multi-turn sessions via previous_response_id.",
    ),
):
    """Serve a Mellea program as an OpenAI-compatible HTTP endpoint.

    Loads a Python file containing a `serve` function and exposes it
    via a FastAPI server implementing the OpenAI chat completions API. The server
    accepts `POST /v1/chat/completions` and `POST /v1/responses` requests.

    Multi-turn sessions are supported on the Responses API: each completed
    response is stored in memory and returned in its `id` field.  Pass that
    `id` as `previous_response_id` in the next request to continue the
    conversation without resending history.  Responses expire after
    `--response-ttl` seconds.

    Prerequisites:
        Mellea installed with server dependency group (`uv add 'mellea[server]'`).
        The python file being loaded must have a `serve` function.

    Output:
        Starts a long-running HTTP server on the specified host and port.
        The `/v1/chat/completions` endpoint accepts OpenAI-format chat
        completion requests and returns `ChatCompletion` JSON responses.
        The `/v1/responses` endpoint accepts Responses API requests and
        supports multi-turn sessions via `previous_response_id`.

    Examples:
        m serve my_app.py --port 9000
        m serve my_app.py --response-ttl 3600

    See Also:
        guide: integrations/m-serve
    """

    from cli.serve.app import run_server

    run_server(script_path=script_path, host=host, port=port, response_ttl=response_ttl)
