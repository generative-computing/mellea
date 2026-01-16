"""Server Type Helpers."""

from enum import Enum
from urllib.parse import urlparse


class _ServerType(Enum):
    """Different types of servers that might be relevant for a backend."""

    UNKNOWN = 0
    LOCALHOST = 1
    OPENAI = 2
    REMOTE_VLLM = 3
    """Must be set manually for now."""


def _server_type(url: str) -> _ServerType:
    """Find a server type based on the url."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
            return _ServerType.LOCALHOST
        elif hostname == "api.openai.com":
            return _ServerType.OPENAI
    except Exception as e:
        print(f"Error parsing URL: {e}")
    return _ServerType.UNKNOWN
