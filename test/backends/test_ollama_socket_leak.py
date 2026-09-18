# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-socket regression guard for issue #349.

Targeted stand-in for globally re-enabling `-W always::ResourceWarning`: creates a
real `OllamaModelBackend`, drives both the sync and async paths, closes it, and
asserts no `ResourceWarning` naming the Ollama port is emitted.

The scenario runs in a subprocess. A `ResourceWarning` only fires when the leaked
object is finalized, so recording warnings around an in-process `gc.collect()` also
catches whatever *other* tests left unreachable: `test_ollama.py`, which runs
immediately before this file, leaves async clients bound to closed `asyncio.run`
loops behind — the one case `close()` documents it cannot reclaim. Their sockets
point at the same port, so the in-process form of this test failed on leaks it does
not own, on whichever Python version happened to collect them inside the recording
window. In a child process the only Ollama client is the one under test, so a
warning naming the port is unambiguously a regression here.
"""

import subprocess
import sys

import pytest

pytestmark = [pytest.mark.ollama, pytest.mark.e2e]

PROBE_TIMEOUT: float = 300.0
"""Seconds allowed for the child process, which may pull the model on a cold host."""

_PROBE = """
import gc

from mellea.backends.model_ids import IBM_GRANITE_4_2_3B
from mellea.backends.ollama import OllamaModelBackend
from mellea.helpers.event_loop_helper import _run_async_in_thread

backend = OllamaModelBackend(model_id=IBM_GRANITE_4_2_3B)

# Report the port the client actually connected to rather than assuming 11434:
# `OllamaModelBackend` resolves its host through `OLLAMA_HOST`, so a hard-coded port
# would silently match nothing (and pass) wherever that env var points elsewhere.
base_url = backend._client._client.base_url
assert base_url.port is not None, f"no port to filter warnings on in base url {base_url}"
print(f"PORT {base_url.port}")

backend._client.ps()
# Route through Mellea's own background loop, same as real call sites
# (e.g. call_tools()), so the async client binds to a loop that stays
# alive until backend.close() runs — not one that closes out from under it.
_run_async_in_thread(backend._async_client.ps())

backend.close()

# Finalize the clients while the interpreter can still print warnings, so a socket
# `close()` failed to release is reported here rather than during shutdown.
del backend
gc.collect()
"""


def _probe_port(stdout: str) -> str:
    """The Ollama port the probe reported, as text for substring matching.

    Args:
        stdout: The probe's captured standard output.

    Returns:
        The port the probe's client connected to.
    """
    ports = [
        line.removeprefix("PORT ").strip()
        for line in stdout.splitlines()
        if line.startswith("PORT ")
    ]
    assert len(ports) == 1, f"probe did not report exactly one port:\n{stdout}"
    return ports[0]


def test_close_leaves_no_socket_resource_warning():
    probe = subprocess.run(
        [sys.executable, "-W", "always::ResourceWarning", "-c", _PROBE],
        capture_output=True,
        text=True,
        timeout=PROBE_TIMEOUT,
    )
    assert probe.returncode == 0, (
        f"leak probe exited {probe.returncode}:\n{probe.stdout}\n{probe.stderr}"
    )

    port = _probe_port(probe.stdout)
    # Warnings from the whole child run, interpreter shutdown included — a socket the
    # backend never closed shows up even if nothing finalized it before exit.
    leaks = [
        line
        for line in probe.stderr.splitlines()
        if "ResourceWarning" in line and port in line
    ]
    assert leaks == [], "leaked Ollama socket(s):\n" + "\n".join(leaks)


if __name__ == "__main__":
    pytest.main([__file__])
