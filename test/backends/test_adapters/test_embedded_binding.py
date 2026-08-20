# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for EmbeddedBinding (Epic #929 Phase 2, issue #1142).

No real backend or network access. `apply_activation` is exercised directly
against `EmbeddedActivationRequest` instances built in-test.
"""

from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mellea.backends.adapters import (
    EmbeddedActivationRequest,
    EmbeddedBinding,
    Identity,
)
from mellea.plugins.types import HookType


def _identity(
    name: str = "answerability", adapter_type: Literal["lora", "alora"] = "alora"
) -> Identity:
    return Identity(name=name, adapter_type=adapter_type, capability=name)


async def test_apply_activation_sets_adapter_name():
    binding = EmbeddedBinding()
    request = EmbeddedActivationRequest(extra_body={}, api_params={})

    await binding.apply_activation(request, _identity("answerability"))

    assert request.extra_body["chat_template_kwargs"]["adapter_name"] == "answerability"


async def test_apply_activation_removes_model_param():
    # The rewriter config can set `model` to the adapter name; for an embedded
    # adapter the real model is the base model already being served, so this
    # must be dropped rather than sent to the API.
    binding = EmbeddedBinding()
    request = EmbeddedActivationRequest(
        extra_body={}, api_params={"model": "answerability_alora", "seed": 1}
    )

    await binding.apply_activation(request, _identity("answerability"))

    assert "model" not in request.api_params
    assert request.api_params["seed"] == 1


async def test_apply_activation_preserves_existing_chat_template_kwargs():
    binding = EmbeddedBinding()
    request = EmbeddedActivationRequest(
        extra_body={"chat_template_kwargs": {"enable_thinking": True}}, api_params={}
    )

    await binding.apply_activation(request, _identity("citations"))

    ctk = request.extra_body["chat_template_kwargs"]
    assert ctk["enable_thinking"] is True
    assert ctk["adapter_name"] == "citations"


async def test_apply_activation_handles_explicit_none_chat_template_kwargs():
    # The `or {}` branch: an explicit None value (not a missing key) is
    # normalised to a dict rather than crashing on assignment.
    binding = EmbeddedBinding()
    request = EmbeddedActivationRequest(
        extra_body={"chat_template_kwargs": None}, api_params={}
    )

    await binding.apply_activation(request, _identity("answerability"))

    assert request.extra_body["chat_template_kwargs"]["adapter_name"] == (
        "answerability"
    )


def test_no_weights_verbs_on_embedded_binding():
    binding = EmbeddedBinding()
    for verb in ("prepare", "activate", "deactivate", "release"):
        assert not hasattr(binding, verb), f"EmbeddedBinding must not have {verb!r}"


async def test_multi_call_isolation():
    # EmbeddedBinding is stateless across calls: activating one adapter must not
    # leak into the request built for the next call.
    binding = EmbeddedBinding()

    request_one = EmbeddedActivationRequest(extra_body={}, api_params={"model": "m"})
    await binding.apply_activation(request_one, _identity("answerability"))

    request_two = EmbeddedActivationRequest(extra_body={}, api_params={"model": "m"})
    await binding.apply_activation(
        request_two, _identity("citations", adapter_type="lora")
    )

    assert request_one.extra_body["chat_template_kwargs"]["adapter_name"] == (
        "answerability"
    )
    assert request_two.extra_body["chat_template_kwargs"]["adapter_name"] == (
        "citations"
    )


def test_from_base_model_records_backend_base_model_name():
    backend = MagicMock(base_model_name="granite-switch")
    binding = EmbeddedBinding.from_base_model(backend)
    assert binding.source == "granite-switch"


async def test_apply_activation_fires_phase_complete_metric():
    # AdapterFunctionMetricsPlugin (mellea/telemetry/metrics_plugins.py) hooks
    # into `adapter_function_phase_complete` to record the
    # `mellea.adapter_function.phase_duration` histogram. Pin that
    # apply_activation fires it correctly for the "activate" phase.
    pytest.importorskip("cpex", reason="cpex not installed — install mellea[hooks]")
    binding = EmbeddedBinding()
    request = EmbeddedActivationRequest(extra_body={}, api_params={})

    with (
        patch("mellea.backends.adapters._core.has_plugins", return_value=True),
        patch(
            "mellea.backends.adapters._core.invoke_hook", new_callable=AsyncMock
        ) as mock_invoke,
    ):
        await binding.apply_activation(request, _identity("answerability"))

    mock_invoke.assert_awaited_once()
    assert mock_invoke.call_args.args[0] is HookType.ADAPTER_FUNCTION_PHASE_COMPLETE
    payload = mock_invoke.call_args.args[1]
    assert payload.name == "answerability"
    assert payload.phase == "activate"


async def test_apply_activation_does_not_fire_invocation_complete():
    # apply_activation only edits the request — the real generate+parse
    # outcome isn't known yet at this point (OpenAIBackend resolves it later,
    # lazily, when the caller awaits the ModelOutputThunk). Firing
    # `adapter_function_invocation_complete` here would have to guess an
    # `outcome`, which would misreport failed calls as "success". Pin that it
    # doesn't, so nobody "fixes" this back to a hardcoded success outcome
    # without solving the underlying problem (see the docstring on
    # apply_activation for the follow-up this needs).
    pytest.importorskip("cpex", reason="cpex not installed — install mellea[hooks]")
    binding = EmbeddedBinding()
    request = EmbeddedActivationRequest(extra_body={}, api_params={})

    with (
        patch("mellea.backends.adapters._core.has_plugins", return_value=True),
        patch(
            "mellea.backends.adapters._core.invoke_hook", new_callable=AsyncMock
        ) as mock_invoke,
    ):
        await binding.apply_activation(request, _identity("answerability"))

    fired_hook_types = [call.args[0] for call in mock_invoke.call_args_list]
    assert HookType.ADAPTER_FUNCTION_INVOCATION_COMPLETE not in fired_hook_types
