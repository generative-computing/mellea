# pytest: skip_always
"""Basic example of running the OpenAI-compatible server with mellea.

This example demonstrates how to start a server that wraps a mellea backend
and exposes it via OpenAI's API format.
"""

import socket

from mellea.integrations.openai_compat.server import ServerConfig, run_server


def find_unused_port(start_port: int = 8000, max_attempts: int = 100) -> int:
    """Find an unused port starting from start_port.

    Args:
        start_port: Port to start searching from.
        max_attempts: Maximum number of ports to try.

    Returns:
        An available port number.

    Raises:
        RuntimeError: If no available port is found.
    """
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue
    raise RuntimeError(
        f"Could not find an unused port in range {start_port}-{start_port + max_attempts}"
    )


def main():
    """Run the OpenAI-compatible server with Ollama backend."""
    # Configure the server to use Ollama with granite4:micro
    config = ServerConfig(
        backend_name="ollama",
        model_id="granite4:micro",
        default_model_options={"temperature": 0.7},
    )

    # Find an available port starting from 8000
    port = find_unused_port(start_port=8000)

    # Start the server
    print(f"Starting OpenAI-compatible server on http://localhost:{port}")
    print(f"API endpoint: http://localhost:{port}/v1/chat/completions")
    print(f"Models endpoint: http://localhost:{port}/v1/models")
    print(f"Health check: http://localhost:{port}/health")
    print(f"API docs: http://localhost:{port}/docs")

    run_server(host="0.0.0.0", port=port, config=config)


if __name__ == "__main__":
    main()
