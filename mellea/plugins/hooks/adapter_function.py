# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter function invocation hook payloads."""

from __future__ import annotations

from typing import Any, Literal

from mellea.plugins.base import MelleaBasePayload

# Every lifecycle phase that fires (or is expected to fire in a future issue)
# ADAPTER_FUNCTION_PHASE_START / ADAPTER_FUNCTION_PHASE_COMPLETE. Firing sites,
# issue #1466:
#   prepare    -- metric-only completion in LocalFileBinding.prepare(); tracing
#                 is deferred because setup is not an adapter-function invocation
#   activate   -- AdapterMixin.adapter_scope()
#   deactivate -- AdapterMixin.adapter_scope()
#   generate   -- none yet; blocked on #1465 wiring generation through adapter_scope
#   parse      -- none yet; blocked on #1465 wiring parsing through adapter_scope
#   release    -- none yet, deliberately: WeightsBinding.release() runs outside
#                 adapter_scope and is not wrapped in an invocation, unlike
#                 prepare/activate/deactivate (see LocalFileBinding.release() and
#                 the pre-existing note this carries forward from Epic #929 Phase 1
#                 that "release" has no phase-duration metric in this contract).
#                 It keeps a Literal value so downstream consumers can name it, but
#                 has no firing site — tracked as remaining work, not silently
#                 dropped.
AdapterFunctionPhase = Literal[
    "prepare", "activate", "generate", "parse", "deactivate", "release"
]


class AdapterFunctionInvocationStartPayload(MelleaBasePayload):
    """Payload for `adapter_function_invocation_start` — before an adapter function invocation begins.

    Attributes:
        adapter_function_invocation_id: Correlation id shared with the matching
            `adapter_function_invocation_complete` event.
        name: Adapter function name (e.g. `"answerability"`).
        revision: Catalog revision of the adapter, or `None` if unpinned.
        binding_type: Weight-binding reality the adapter will run under (e.g.
            `"local_file"`, `"embedded"`, `"server_mediated"`).
        adapter_type: Adapter mechanism (e.g. `"lora"`, `"alora"`).
    """

    adapter_function_invocation_id: str
    name: str
    revision: str | None = None
    binding_type: str = "unknown"
    adapter_type: str = "unknown"


class AdapterFunctionInvocationCompletePayload(MelleaBasePayload):
    """Payload for `adapter_function_invocation_complete` — after an adapter function invocation finishes.

    Attributes:
        adapter_function_invocation_id: Correlation id shared with the `adapter_function_invocation_start`
            event that opened this invocation.
        name: Adapter function name (e.g. `"answerability"`).
        revision: Catalog revision of the adapter, or `None` if unpinned.
        binding_type: Weight-binding reality the adapter ran under (e.g.
            `"local_file"`, `"embedded"`, `"server_mediated"`).
        adapter_type: Adapter mechanism (e.g. `"lora"`, `"alora"`).
        outcome: `"success"`, `"schema_error"`, or `"error"`.
        error: The exception raised during invocation, or `None` on success.
    """

    adapter_function_invocation_id: str
    name: str
    revision: str | None = None
    binding_type: str = "unknown"
    adapter_type: str = "unknown"
    # Required, not defaulted: an invocation always has a determined outcome, and
    # defaulting to "success" would let a forgotten emit silently record success.
    outcome: Literal["success", "schema_error", "error"]
    # Carried for consumers that inspect the failure (e.g. structured logging);
    # the metrics plugin classifies on `outcome` and does not read this field.
    # Typed `Any` rather than `BaseException | None` because the payload base is
    # pydantic-backed and an exception type would need `arbitrary_types_allowed`.
    error: Any = None


class AdapterFunctionPhaseStartPayload(MelleaBasePayload):
    """Payload for `adapter_function_phase_start` — before one lifecycle phase begins.

    Attributes:
        adapter_function_invocation_id: Correlation id of the enclosing invocation (shared with the
            `adapter_function_invocation_start`/`_complete` events).
        name: Adapter function name (e.g. `"answerability"`).
        phase: Lifecycle phase about to run. See `AdapterFunctionPhase` for which
            values currently have a firing site.
        revision: Catalog revision of the adapter, or `None` if unpinned.
    """

    adapter_function_invocation_id: str
    name: str
    phase: AdapterFunctionPhase
    revision: str | None = None


class AdapterFunctionPhaseCompletePayload(MelleaBasePayload):
    """Payload for `adapter_function_phase_complete` — after one lifecycle phase finishes.

    Only fires when the phase itself succeeded — a phase that raised did not
    complete, and its failure is reported once, at invocation level, via
    `adapter_function_invocation_complete`'s `outcome`/`error`.

    Attributes:
        adapter_function_invocation_id: Correlation id of the enclosing invocation,
            or `None` for an existing metric-only completion with no span.
        name: Adapter function name (e.g. `"answerability"`).
        phase: Lifecycle phase that completed. See `AdapterFunctionPhase` for
            which values currently have a firing site.
        duration_ms: Wall-clock duration of the phase in milliseconds.
    """

    adapter_function_invocation_id: str | None = None
    name: str
    # Constrained to a Literal so a typo can't silently spawn a new metric-label
    # series (the phase becomes a metric dimension). Required, with no unset
    # sentinel: a phase-complete event always has a real phase. (The payload is a
    # pydantic model, so a required field after the base's defaulted ones is fine.)
    phase: AdapterFunctionPhase
    duration_ms: float
