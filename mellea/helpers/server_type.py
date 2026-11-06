# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for detecting and classifying the target inference server.

Defines the `_ServerType` enum (`LOCALHOST`, `OPENAI`, `REMOTE_VLLM`,
`UNKNOWN`) and `_server_type`, which classifies a URL by hostname. Also provides
`is_vllm_server_with_structured_output`, which probes a server's `/version`
endpoint to determine whether it supports the `structured_outputs` parameter
introduced in vLLM ≥ 0.12.0. Used by the OpenAI-compatible backend to choose between
`guided_json` and `structured_outputs` request formats. `vllm_server_version` and
`vllm_tokenize_single_tokens` support one-token decision readouts, which ask a vLLM
server for the logprobs of exact token ids (`logprob_token_ids`).
"""

import ipaddress
import json
import socket
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


def _is_link_local_host(hostname: str) -> bool:
    """Return True if `hostname` resolves to a link-local address.

    Link-local ranges (IPv4 `169.254.0.0/16`, IPv6 `fe80::/10`) include the
    cloud metadata endpoint `169.254.169.254`, which is a common SSRF target.
    Resolves the hostname first so that a name mapping to a link-local address
    is caught, not just literal IPs. Unresolvable hosts are treated as not
    link-local (the request will fail on its own).

    This is a resolve-then-fetch check: `requests` resolves the host again
    independently, so a hostname that rebinds between the two lookups (DNS
    rebinding / TOCTOU) is not covered. Acceptable for a developer-supplied
    `base_url`. Scoped to link-local per #1246 — provider-specific metadata
    addresses outside that range (e.g. Alibaba `100.100.100.200`) are not caught.

    Args:
        hostname: The hostname or IP literal to check.

    Returns:
        True if the host resolves to a link-local address, False otherwise.
    """
    try:
        addrinfos = socket.getaddrinfo(hostname, None)
    except (socket.gaierror, UnicodeError):
        return False

    for family, _, _, _, sockaddr in addrinfos:
        try:
            addr = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            continue
        if addr.is_link_local:
            return True
    return False


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
        # Deny requests to link-local / cloud metadata endpoints (e.g.
        # 169.254.169.254) before probing to avoid SSRF against them.
        hostname = urlparse(base_url).hostname
        if hostname is not None and _is_link_local_host(hostname):
            return False

        # There's some arbitrariness in vllm endpoints. Best we can do is ensure the `/v1` that is required
        # to come before the chat and completions endpoints is removed.
        version_endpoint = (
            base_url.removesuffix("/v1").removesuffix("/v1/") + "/version"
        )

        # Disallow redirects: a benign host could 302 to a link-local address
        # (e.g. 169.254.169.254), which requests would otherwise follow and
        # defeat the guard above. The /version probe should not redirect.
        version_response = requests.get(
            version_endpoint, headers=headers, timeout=5, allow_redirects=False
        )
        if not version_response.ok:
            return False

        # Should look something like: `{"version":"0.16.0rc1.dev172+gd88a1df69"}`.
        json_version = json.loads(version_response.text)
        return Version(json_version["version"]) >= VLLM_VERSION_STRUCTURED_OUTPUTS
    except Exception:
        return False


def vllm_server_version(base_url: str, headers: Mapping[str, Any]) -> Version | None:
    """Return a vLLM server's version from its `/version` endpoint, or None.

    Applies the same link-local and no-redirect guards as
    `is_vllm_server_with_structured_output`.

    Args:
        base_url: Base url for LLM API.
        headers: Additional headers to pass to the request.

    Returns:
        The parsed version, or None when the server is not vLLM, does not
        answer, or reports an unparsable version.
    """
    try:
        hostname = urlparse(base_url).hostname
        if hostname is not None and _is_link_local_host(hostname):
            return None
        version_endpoint = (
            base_url.removesuffix("/v1").removesuffix("/v1/") + "/version"
        )
        response = requests.get(
            version_endpoint, headers=headers, timeout=5, allow_redirects=False
        )
        if not response.ok:
            return None
        return Version(json.loads(response.text)["version"])
    except Exception:
        return None


def vllm_tokenize_single_tokens(
    base_url: str, headers: Mapping[str, Any], model: str, texts: list[str]
) -> list[int]:
    """Token ids of texts that each encode to exactly one token, via a vLLM server.

    Calls the server's `/tokenize` endpoint (a sibling of `/v1`, not under it)
    without special tokens, so the ids match what the server will score.

    Args:
        base_url: Base url for LLM API.
        headers: Additional headers to pass to the request.
        model: Served model name, as the server knows it.
        texts: Strings that must each be a single token (for example decision
            labels).

    Returns:
        One token id per text, in order.

    Raises:
        ValueError: If the base URL is link-local, the server rejects the
            request or answers in an unexpected shape, or a text does not
            encode to exactly one token.
    """
    hostname = urlparse(base_url).hostname
    if hostname is not None and _is_link_local_host(hostname):
        raise ValueError(f"Refusing to tokenize against link-local host {hostname!r}.")
    endpoint = base_url.removesuffix("/v1").removesuffix("/v1/") + "/tokenize"
    ids: list[int] = []
    for text in texts:
        response = requests.post(
            endpoint,
            headers=headers,
            json={"model": model, "prompt": text, "add_special_tokens": False},
            timeout=10,
            allow_redirects=False,
        )
        if not response.ok:
            raise ValueError(
                f"The server's /tokenize endpoint answered {response.status_code} "
                f"for {text!r}; cannot map decision labels to token ids."
            )
        tokens = response.json().get("tokens")
        if not isinstance(tokens, list) or len(tokens) != 1:
            raise ValueError(
                f"Decision label {text!r} encodes to {tokens!r} on the server; "
                "a one-token readout needs single-token labels."
            )
        ids.append(int(tokens[0]))
    return ids
