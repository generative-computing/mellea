# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helper for asserting on the adapter-function hooks (Epic #929).

Single home for the hook-capture idiom, so the patching contract below is stated
once. Not a `conftest.py` fixture: callers need to wrap a *specific* block inside
a test (the integration tests capture only the `adapter_scope` section, not the
whole test), which a fixture cannot express.

Assertions here are on **hooks, not spans**. `adapter_scope` fires hooks and never
opens a span — #1464 documents that rule, #1466 adds the spans from a plugin.
"""

import contextlib
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

from mellea.plugins.hooks.adapter_function import (
    AdapterFunctionInvocationCompletePayload,
    AdapterFunctionInvocationStartPayload,
    AdapterFunctionPhaseCompletePayload,
    AdapterFunctionPhaseStartPayload,
)

_TARGET = "mellea.backends.adapters.adapter"


@contextlib.contextmanager
def capture_adapter_hooks() -> Iterator[MagicMock]:
    """Capture the hook payloads fired inside the block.

    Patches three things, each for a distinct reason:

    - **`has_plugins` pinned `True`.** It is already `True` under pytest —
      `test/conftest.py`'s `auto_register_acceptance_sets` is `autouse`,
      session-scoped, and registers a plugin for every `HookType`
      (`test/plugins/_acceptance_sets.py`). Pinning it removes the dependency on
      that ambient registration.
    - **`invoke_hook` replaced with `new_callable=MagicMock`.** Load-bearing:
      `invoke_hook` is an `async def`, so a bare `patch()` auto-creates an
      `AsyncMock`. Calling an `AsyncMock` returns a coroutine, and if a
      `side_effect` returns a coroutine of its own, *that* inner coroutine becomes
      the outer one's result and is never awaited — surfacing as
      `PytestUnraisableExceptionWarning: coroutine ... was never awaited`. Note
      `-W error::RuntimeWarning` does **not** catch it; use
      `-W error::pytest.PytestUnraisableExceptionWarning`. Forcing a sync
      `MagicMock` means no coroutine exists to leak.
    - **`_run_async_in_thread` patched out.** Real dispatch works fine; it is
      simply not needed to read the payloads, and skipping it keeps these tests
      off the shared event loop.

    Yields:
        The `invoke_hook` mock. Use `hook_payloads()` to read what it recorded.
    """
    with (
        patch(f"{_TARGET}.has_plugins", return_value=True),
        patch(f"{_TARGET}.invoke_hook", new_callable=MagicMock) as mock_invoke,
        patch(f"{_TARGET}._run_async_in_thread"),
    ):
        yield mock_invoke


def hook_payloads(mock_invoke: MagicMock) -> list:
    """Returns the payload argument of every recorded `invoke_hook` call, in order.

    Args:
        mock_invoke: The mock yielded by `capture_adapter_hooks`.

    Returns:
        Each call's payload, ordered as fired.
    """
    return [call.args[1] for call in mock_invoke.call_args_list]


def phase_start_payloads(mock_invoke: MagicMock) -> list:
    """Returns only the phase-start payloads.

    Args:
        mock_invoke: The mock yielded by `capture_adapter_hooks`.

    Returns:
        The recorded `AdapterFunctionPhaseStartPayload`s, ordered as fired.
    """
    return [
        p
        for p in hook_payloads(mock_invoke)
        if isinstance(p, AdapterFunctionPhaseStartPayload)
    ]


def phase_payloads(mock_invoke: MagicMock) -> list:
    """Returns only the phase-complete payloads.

    Args:
        mock_invoke: The mock yielded by `capture_adapter_hooks`.

    Returns:
        The recorded `AdapterFunctionPhaseCompletePayload`s, ordered as fired.
    """
    return [
        p
        for p in hook_payloads(mock_invoke)
        if isinstance(p, AdapterFunctionPhaseCompletePayload)
    ]


def invocation_start_payloads(mock_invoke: MagicMock) -> list:
    """Returns only the invocation-start payloads.

    Args:
        mock_invoke: The mock yielded by `capture_adapter_hooks`.

    Returns:
        The recorded `AdapterFunctionInvocationStartPayload`s, ordered as fired.
    """
    return [
        p
        for p in hook_payloads(mock_invoke)
        if isinstance(p, AdapterFunctionInvocationStartPayload)
    ]


def invocation_payloads(mock_invoke: MagicMock) -> list:
    """Returns only the invocation-complete payloads.

    Args:
        mock_invoke: The mock yielded by `capture_adapter_hooks`.

    Returns:
        The recorded `AdapterFunctionInvocationCompletePayload`s, ordered as fired.
    """
    return [
        p
        for p in hook_payloads(mock_invoke)
        if isinstance(p, AdapterFunctionInvocationCompletePayload)
    ]
