# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter function invocation hook payloads."""

from __future__ import annotations

from typing import Any, Literal

from mellea.plugins.base import MelleaBasePayload


class AdapterFunctionInvocationCompletePayload(MelleaBasePayload):
    """Payload for `adapter_function_invocation_complete` — after an adapter function invocation finishes.

    Attributes:
        name: Adapter function name (e.g. `"answerability"`).
        revision: Catalog revision of the adapter, or `None` if unpinned.
        binding_type: Weight-binding reality the adapter ran under (e.g.
            `"local_file"`, `"embedded"`, `"server_mediated"`).
        adapter_type: Adapter mechanism (e.g. `"lora"`, `"alora"`).
        outcome: `"success"`, `"schema_error"`, or `"error"`.
        error: The exception raised during invocation, or `None` on success.

    Note:
        The span of work `outcome` covers differs by `binding_type`, so
        aggregating across bindings mixes events with different scope.
        `"local_file"` covers activate→generate→deactivate only
        (`AdapterMixin.adapter_scope` closes before parsing runs);
        `"embedded"` covers generate+JSON-parse
        (`_fire_embedded_invocation_complete` in
        `mellea.backends.adapters._core`). Neither covers the later
        contract-level `IOContract` validation in `call_intrinsic` — a
        contract-mismatch on already-valid JSON records `"success"` for
        either binding (tracked in #1611).
    """

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


class AdapterFunctionPhaseCompletePayload(MelleaBasePayload):
    """Payload for `adapter_function_phase_complete` — after one lifecycle phase finishes.

    Attributes:
        name: Adapter function name (e.g. `"answerability"`).
        phase: Lifecycle phase (`"prepare"`, `"activate"`, `"generate"`,
            `"parse"`, or `"deactivate"`).
        duration_ms: Wall-clock duration of the phase in milliseconds.
    """

    name: str
    # Constrained to a Literal so a typo can't silently spawn a new metric-label
    # series (the phase becomes a metric dimension). Required, with no unset
    # sentinel: a phase-complete event always has a real phase. (The payload is a
    # pydantic model, so a required field after the base's defaulted ones is fine.)
    phase: Literal["prepare", "activate", "generate", "parse", "deactivate"]
    duration_ms: float
