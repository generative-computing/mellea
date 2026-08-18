# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for EmbeddedBinding (Epic #929 Phase 2, issue #1142).

No real backend or network access. `apply_activation` is exercised directly
against `EmbeddedActivationRequest` instances built in-test.
"""

from typing import Literal
from unittest.mock import MagicMock, patch

import pytest

from mellea.backends.adapters._core import (
    EmbeddedActivationRequest,
    EmbeddedBinding,
    Identity,
)


def _identity(
    name: str = "answerability", adapter_type: Literal["lora", "alora"] = "alora"
) -> Identity:
    return Identity(name=name, adapter_type=adapter_type, capability=name)


def test_apply_activation_sets_adapter_name():
    binding = EmbeddedBinding()
    request = EmbeddedActivationRequest(extra_body={}, api_params={})

    binding.apply_activation(request, _identity("answerability"))

    assert request.extra_body["chat_template_kwargs"]["adapter_name"] == "answerability"


def test_apply_activation_removes_model_param():
    # The rewriter config can set `model` to the adapter name; for an embedded
    # adapter the real model is the base model already being served, so this
    # must be dropped rather than sent to the API.
    binding = EmbeddedBinding()
    request = EmbeddedActivationRequest(
        extra_body={}, api_params={"model": "answerability_alora", "seed": 1}
    )

    binding.apply_activation(request, _identity("answerability"))

    assert "model" not in request.api_params
    assert request.api_params["seed"] == 1


def test_apply_activation_preserves_existing_chat_template_kwargs():
    binding = EmbeddedBinding()
    request = EmbeddedActivationRequest(
        extra_body={"chat_template_kwargs": {"enable_thinking": True}}, api_params={}
    )

    binding.apply_activation(request, _identity("citations"))

    ctk = request.extra_body["chat_template_kwargs"]
    assert ctk["enable_thinking"] is True
    assert ctk["adapter_name"] == "citations"


def test_no_weights_verbs_on_embedded_binding():
    binding = EmbeddedBinding()
    for verb in ("prepare", "activate", "deactivate", "release"):
        assert not hasattr(binding, verb), f"EmbeddedBinding must not have {verb!r}"


def test_multi_call_isolation():
    # EmbeddedBinding is stateless across calls: activating one adapter must not
    # leak into the request built for the next call.
    binding = EmbeddedBinding()

    request_one = EmbeddedActivationRequest(extra_body={}, api_params={"model": "m"})
    binding.apply_activation(request_one, _identity("answerability"))

    request_two = EmbeddedActivationRequest(extra_body={}, api_params={"model": "m"})
    binding.apply_activation(request_two, _identity("citations", adapter_type="lora"))

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


def test_metrics_invocation_counter_increments_for_embedded():
    # AdapterFunctionMetricsPlugin (mellea/telemetry/metrics_plugins.py) hooks
    # into `adapter_function_invocation_complete` to record the
    # `mellea.adapter_function.invocations` counter, keyed in part by
    # `binding_type`. Pin that apply_activation fires it correctly for the
    # "embedded" binding type.
    pytest.importorskip("cpex", reason="cpex not installed — install mellea[hooks]")
    binding = EmbeddedBinding()
    request = EmbeddedActivationRequest(extra_body={}, api_params={})

    with (
        patch("mellea.backends.adapters._core.has_plugins", return_value=True),
        patch(
            "mellea.backends.adapters._core.invoke_hook", new_callable=MagicMock
        ) as mock_invoke,
        patch("mellea.backends.adapters._core._run_async_in_thread"),
    ):
        binding.apply_activation(request, _identity("answerability"))

    payloads = [call.args[1] for call in mock_invoke.call_args_list]
    invocation_payloads = [p for p in payloads if hasattr(p, "outcome")]

    assert len(invocation_payloads) == 1
    payload = invocation_payloads[0]
    assert payload.name == "answerability"
    assert payload.binding_type == "embedded"
    assert payload.adapter_type == "alora"
    assert payload.outcome == "success"


def test_activate_span_binding_type_is_embedded():
    # Span emission is deferred to #1466 (blocked on the missing start hooks —
    # see this issue's OTel section). This pins the underlying
    # `adapter_function_phase_complete` payload's shape ahead of that: once
    # #1466 lands a tracing plugin, it opens the `adapter_function.activate`
    # span from this same phase-complete event and sets
    # `mellea.adapter_function.binding_type="embedded"` from it.
    pytest.importorskip("cpex", reason="cpex not installed — install mellea[hooks]")
    binding = EmbeddedBinding()
    request = EmbeddedActivationRequest(extra_body={}, api_params={})

    with (
        patch("mellea.backends.adapters._core.has_plugins", return_value=True),
        patch(
            "mellea.backends.adapters._core.invoke_hook", new_callable=MagicMock
        ) as mock_invoke,
        patch("mellea.backends.adapters._core._run_async_in_thread"),
    ):
        binding.apply_activation(request, _identity("answerability"))

    payloads = [call.args[1] for call in mock_invoke.call_args_list]
    phase_payloads = [p for p in payloads if hasattr(p, "phase")]
    invocation_payloads = [p for p in payloads if hasattr(p, "outcome")]

    assert len(phase_payloads) == 1
    assert phase_payloads[0].phase == "activate"
    assert phase_payloads[0].name == "answerability"
    assert invocation_payloads[0].binding_type == "embedded"
