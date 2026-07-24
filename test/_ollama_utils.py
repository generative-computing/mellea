# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Ollama eviction helpers for test and example conftest files.

The core evict-all-loaded-models logic is duplicated between `test/conftest.py`
and `docs/examples/conftest.py`; this module holds the single implementation.
Each conftest keeps its own teardown hook and logging style by passing
`on_info`/`on_warning` callbacks (see #808).
"""

from __future__ import annotations

import os
from collections.abc import Callable

import requests


def resolve_ollama_base_url() -> str:
    """Resolve the Ollama HTTP base URL from the environment.

    Reads `OLLAMA_HOST` (which may be `host`, `host:port`, or absent) and
    `OLLAMA_PORT`, normalizing the server bind address `0.0.0.0` to
    `127.0.0.1` so client connections succeed.

    Returns:
        The base URL, e.g. `http://127.0.0.1:11434`.
    """
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1")
    if ":" in host:
        host, port = host.rsplit(":", 1)
    else:
        port = os.environ.get("OLLAMA_PORT", "11434")

    if host == "0.0.0.0":
        host = "127.0.0.1"

    return f"http://{host}:{port}"


def evict_all_loaded_ollama_models(
    *,
    on_info: Callable[[str], None] | None = None,
    on_warning: Callable[[str], None] | None = None,
) -> None:
    """Evict all currently loaded Ollama models to free memory.

    Queries `/api/ps` to discover loaded models, then sends `keep_alive=0`
    to each via `/api/generate`. Prevents heavyweight models from starving
    subsequent tests of memory (see #798).

    Best-effort: errors are routed to `on_warning` but never raised. Callbacks
    default to no-ops so callers that don't care about logging can omit them.

    Args:
        on_info: Optional callback invoked with a message after each successful
            eviction.
        on_warning: Optional callback invoked with a message when the model
            query fails or an individual eviction fails.
    """
    info = on_info or (lambda _msg: None)
    warn = on_warning or (lambda _msg: None)

    base_url = resolve_ollama_base_url()

    try:
        resp = requests.get(f"{base_url}/api/ps", timeout=5)
        resp.raise_for_status()
        loaded = resp.json().get("models", [])
    except Exception as e:
        warn(f"ollama-evict: could not query loaded models: {e}")
        return

    for entry in loaded:
        model_name = entry.get("name") or entry.get("model", "unknown")
        try:
            requests.post(
                f"{base_url}/api/generate",
                json={"model": model_name, "keep_alive": 0},
                timeout=10,
            )
            info(f"ollama-evict: evicted {model_name}")
        except Exception as e:
            warn(f"ollama-evict: failed to evict {model_name}: {e}")
