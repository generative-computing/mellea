"""Utilities for detecting and classifying the target inference server.

Defines the ``_ServerType`` enum (``LOCALHOST``, ``OPENAI``, ``REMOTE_VLLM``,
``UNKNOWN``) and ``_server_type``, which classifies a URL by hostname. Also provides
``is_vllm_server_with_structured_output``, which probes a server's ``/version``
endpoint to determine whether it supports the ``structured_outputs`` parameter
introduced in vLLM ≥ 0.12.0. Used by the OpenAI-compatible backend to choose between
``guided_json`` and ``structured_outputs`` request formats.
"""

import json
from collections.abc import Mapping
from enum import Enum
from typing import Any
from urllib.parse import urlparse

import requests
from packaging.version import Version


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


VLLM_VERSION_STRUCTURED_OUTPUTS = Version("v0.12.0")


def is_vllm_server_with_structured_output(
    base_url: str, headers: Mapping[str, Any]
) -> bool:
    """Attempts to determine if the backend is a vllm server with version >= v0.12.0. Defaults to false.

    v0.12.0 was the last version to support guided_json params. It's now under structured_outputs.

    Args:
        base_url: Base url for LLM API.
        headers: Additional headers to pass to the request.

    Returns:
        True if the server is vLLM >= v0.12.0, False otherwise.
    """
    # Not using the models endpoint for now. Assuming version is enough.
    # vllm_provider_endpoint = str(self._client.base_url).join("/models")
    # vllm_provider_response = requests.get(vllm_provider_endpoint)

    try:
        # There's some arbitrariness in vllm endpoints. Best we can do is ensure the `/v1` that is required
        # to come before the chat and completions endpoints is removed.
        version_endpoint = (
            base_url.removesuffix("/v1").removesuffix("/v1/") + "/version"
        )

        version_response = requests.get(version_endpoint, headers=headers)
        if not version_response.ok:
            return False

        # Should look something like: `{"version":"0.16.0rc1.dev172+gd88a1df69"}`.
        json_version = json.loads(version_response.text)
        return Version(json_version["version"]) >= VLLM_VERSION_STRUCTURED_OUTPUTS
    except Exception:
        return False
