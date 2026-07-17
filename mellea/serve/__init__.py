"""Public API for Mellea's OpenAI-compatible server.

Provides types for writing ``serve()`` functions and the ``run_server()``
entry point for starting the server programmatically.

``run_server`` requires the ``mellea[server]`` extra and is imported lazily:

Example:
    ```python
    from mellea.serve.app import run_server

    run_server("my_app.py", host="0.0.0.0", port=8080)
    ```
"""

from .models import ChatMessage, ImageUrlContent, MessageContent, TextContent

__all__ = ["ChatMessage", "ImageUrlContent", "MessageContent", "TextContent"]
