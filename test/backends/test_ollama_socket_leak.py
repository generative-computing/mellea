# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-socket regression guard for issue #349.

Targeted stand-in for globally re-enabling `-W always::ResourceWarning`: creates
a real `OllamaModelBackend`, drives both the sync and async paths, closes it,
and asserts no `ResourceWarning` mentioning the Ollama port survives a `gc.collect()`.
Scoped to this one backend so the assertion doesn't drown in third-party warning
noise the way a blanket `always::ResourceWarning` filter would.
"""

import gc
import warnings

import pytest

from mellea.backends.model_ids import IBM_GRANITE_4_2_3B
from mellea.backends.ollama import OllamaModelBackend
from mellea.helpers.event_loop_helper import _run_async_in_thread

pytestmark = [pytest.mark.ollama, pytest.mark.e2e]


def _resolved_port(backend: OllamaModelBackend) -> int:
    """The port the backend's client actually connected to.

    Read off the constructed client rather than assumed: `OllamaModelBackend` resolves
    its host through `OLLAMA_HOST`, so hard-coding 11434 would silently match nothing
    (and pass) wherever that env var points elsewhere.

    Args:
        backend: The backend whose resolved Ollama port is wanted.

    Returns:
        The port of the sync client's base URL.
    """
    base_url = backend._client._client.base_url
    port = base_url.port
    assert port is not None, f"no port to filter warnings on in base url {base_url}"
    return port


def test_close_leaves_no_socket_resource_warning():
    backend = OllamaModelBackend(model_id=IBM_GRANITE_4_2_3B)
    port = _resolved_port(backend)

    backend._client.ps()
    # Route through Mellea's own background loop, same as real call sites
    # (e.g. call_tools()), so the async client binds to a loop that stays
    # alive until backend.close() runs — not one that closes out from under it.
    _run_async_in_thread(backend._async_client.ps())

    backend.close()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        del backend
        gc.collect()

    port_warnings = [
        w
        for w in caught
        if issubclass(w.category, ResourceWarning) and str(port) in str(w.message)
    ]
    assert port_warnings == [], f"leaked Ollama socket(s): {port_warnings}"


if __name__ == "__main__":
    pytest.main([__file__])
