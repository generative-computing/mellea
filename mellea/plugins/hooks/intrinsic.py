# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter function (intrinsic) invocation hook payloads."""

from __future__ import annotations

from typing import Any

from mellea.plugins.base import MelleaBasePayload


class IntrinsicInvocationCompletePayload(MelleaBasePayload):
    """Payload for `intrinsic_invocation_complete` — after an adapter function invocation finishes.

    Attributes:
        name: Adapter function name (e.g. `"answerability"`).
        revision: Catalog revision of the adapter, or `None` if unpinned.
        binding_type: Weight-binding reality the adapter ran under (e.g.
            `"local_file"`, `"embedded"`, `"server_mediated"`).
        adapter_type: Adapter mechanism (e.g. `"lora"`, `"alora"`).
        outcome: `"success"`, `"schema_error"`, or `"error"`.
        error: The exception raised during invocation, or `None` on success.
    """

    name: str = ""
    revision: str | None = None
    binding_type: str = "unknown"
    adapter_type: str = "unknown"
    outcome: str = "success"
    error: Any = None


class IntrinsicPhaseCompletePayload(MelleaBasePayload):
    """Payload for `intrinsic_phase_complete` — after one lifecycle phase finishes.

    Attributes:
        name: Adapter function name (e.g. `"answerability"`).
        phase: Lifecycle phase (`"prepare"`, `"activate"`, `"generate"`,
            `"parse"`, or `"deactivate"`).
        duration_ms: Wall-clock duration of the phase in milliseconds.
    """

    name: str = ""
    phase: str = ""
    duration_ms: float = 0.0
