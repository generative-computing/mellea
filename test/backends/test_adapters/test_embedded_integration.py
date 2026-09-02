# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test: OpenAIBackend activating embedded adapters through the
new EmbeddedBinding (Epic #929 Phase 2, issue #1142).

A real `OpenAIBackend` and its adapter registration/generation path are used;
only the outer network boundary (the OpenAI async client) is mocked, per
test/README.md's definition of `integration`. No vLLM server or Granite
Switch model is required — see `test/backends/test_openai_intrinsics.py` for
the GPU-backed e2e counterpart.
"""

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from mellea.backends.adapters import EmbeddedBinding, ServerMediatedBinding
from mellea.backends.adapters.adapter import EmbeddedIntrinsicAdapter
from mellea.backends.openai import OpenAIBackend
from mellea.formatters.granite import IntrinsicsResultProcessor
from mellea.plugins.types import HookType
from mellea.stdlib import functional as mfuncs
from mellea.stdlib.components import Intrinsic, Message
from mellea.stdlib.context import ChatContext

pytestmark = pytest.mark.integration

_SIMPLE_CONFIG = {
    "model": None,
    "response_format": None,
    "transformations": None,
    "instruction": None,
    "parameters": {"max_completion_tokens": 64},
    "sentence_boundaries": None,
}


def _chat_completion(content: str = '{"result": "ok"}') -> ChatCompletion:
    return ChatCompletion(
        id="test-embedded-integration",
        created=0,
        model="granite-switch",
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content=content),
            )
        ],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=4, total_tokens=14),
    )


def _backend_with_adapter(technology: str) -> OpenAIBackend:
    backend = OpenAIBackend(
        model_id="granite-switch",
        api_key="fake-key",
        base_url="http://localhost:9999/v1",
    )
    backend.add_adapter(
        EmbeddedIntrinsicAdapter(
            intrinsic_name="answerability", config=_SIMPLE_CONFIG, technology=technology
        )
    )
    return backend


@pytest.mark.parametrize("technology", ["lora", "alora"])
async def test_activation_goes_through_embedded_binding(technology):
    """The registered adapter's weights are a real EmbeddedBinding, and it is
    that binding's `apply_activation` — not an inline isinstance check — that
    ends up writing the request the API call receives."""
    backend = _backend_with_adapter(technology)
    adapter = backend._added_adapters[f"answerability_{technology}"]
    assert isinstance(adapter.weights, EmbeddedBinding)
    # add_adapter stamps the registration-time source (openai.py), distinct
    # from the from_base_model() classmethod covered in test_embedded_binding.py.
    assert adapter.weights.source == "granite-switch"

    mock_create = AsyncMock(return_value=_chat_completion())
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create

    original_apply_activation = EmbeddedBinding.apply_activation
    with (
        patch.object(
            OpenAIBackend,
            "_async_client",
            new_callable=PropertyMock,
            return_value=mock_client,
        ),
        patch.object(
            EmbeddedBinding,
            "apply_activation",
            autospec=True,
            side_effect=original_apply_activation,
        ) as mock_apply,
    ):
        ctx = ChatContext().add(Message("user", "What is the square root of 4?"))
        mot, _ = await mfuncs.aact(
            Intrinsic("answerability"), ctx, backend, strategy=None
        )
        await mot.avalue()

    mock_apply.assert_called_once()
    _, called_identity = mock_apply.call_args.args[1:]
    assert called_identity.name == "answerability"

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["extra_body"]["chat_template_kwargs"]["adapter_name"] == (
        "answerability"
    )
    assert call_kwargs["model"] == "granite-switch"


def _invocation_complete_payloads(mock_invoke: AsyncMock) -> list:
    return [
        call.args[1]
        for call in mock_invoke.call_args_list
        if call.args[0] is HookType.ADAPTER_FUNCTION_INVOCATION_COMPLETE
    ]


async def test_invocation_complete_fires_success_for_embedded_call():
    pytest.importorskip("cpex", reason="cpex not installed — install mellea[hooks]")
    backend = _backend_with_adapter("alora")
    mock_create = AsyncMock(return_value=_chat_completion())
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create

    with (
        patch.object(
            OpenAIBackend,
            "_async_client",
            new_callable=PropertyMock,
            return_value=mock_client,
        ),
        patch("mellea.backends.adapters._core.has_plugins", return_value=True),
        patch(
            "mellea.backends.adapters._core.invoke_hook", new_callable=AsyncMock
        ) as mock_invoke,
    ):
        ctx = ChatContext().add(Message("user", "What is the square root of 4?"))
        mot, _ = await mfuncs.aact(
            Intrinsic("answerability"), ctx, backend, strategy=None
        )
        await mot.avalue()

    payloads = _invocation_complete_payloads(mock_invoke)
    assert len(payloads) == 1
    assert payloads[0].name == "answerability"
    assert payloads[0].binding_type == "embedded"
    assert payloads[0].adapter_type == "alora"
    assert payloads[0].outcome == "success"
    assert payloads[0].error is None


async def test_invocation_complete_fires_schema_error_on_malformed_json():
    # Pins #1142/#1559: a non-JSON response must record schema_error, not success.
    pytest.importorskip("cpex", reason="cpex not installed — install mellea[hooks]")
    backend = _backend_with_adapter("alora")
    mock_create = AsyncMock(return_value=_chat_completion(content="not valid json"))
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create

    with (
        patch.object(
            OpenAIBackend,
            "_async_client",
            new_callable=PropertyMock,
            return_value=mock_client,
        ),
        patch("mellea.backends.adapters._core.has_plugins", return_value=True),
        patch(
            "mellea.backends.adapters._core.invoke_hook", new_callable=AsyncMock
        ) as mock_invoke,
    ):
        ctx = ChatContext().add(Message("user", "What is the square root of 4?"))
        mot, _ = await mfuncs.aact(
            Intrinsic("answerability"), ctx, backend, strategy=None
        )
        with pytest.raises(Exception, match="did not return a JSON"):
            await mot.avalue()

    payloads = _invocation_complete_payloads(mock_invoke)
    assert len(payloads) == 1
    assert payloads[0].outcome == "schema_error"
    assert payloads[0].error is not None


async def test_invocation_complete_fires_error_on_unrelated_exception():
    pytest.importorskip("cpex", reason="cpex not installed — install mellea[hooks]")
    backend = _backend_with_adapter("alora")
    mock_create = AsyncMock(return_value=_chat_completion())
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create

    with (
        patch.object(
            OpenAIBackend,
            "_async_client",
            new_callable=PropertyMock,
            return_value=mock_client,
        ),
        patch.object(
            IntrinsicsResultProcessor, "transform", side_effect=ValueError("boom")
        ),
        patch("mellea.backends.adapters._core.has_plugins", return_value=True),
        patch(
            "mellea.backends.adapters._core.invoke_hook", new_callable=AsyncMock
        ) as mock_invoke,
    ):
        ctx = ChatContext().add(Message("user", "What is the square root of 4?"))
        mot, _ = await mfuncs.aact(
            Intrinsic("answerability"), ctx, backend, strategy=None
        )
        with pytest.raises(ValueError, match="boom"):
            await mot.avalue()

    payloads = _invocation_complete_payloads(mock_invoke)
    assert len(payloads) == 1
    assert payloads[0].outcome == "error"
    assert payloads[0].error is not None


async def test_invocation_complete_fires_error_on_generation_failure():
    # A failure in the SDK call itself never reaches granite_formatters_processing —
    # avalue() raises it straight off the queue — so _await_embedded_generation
    # must fire outcome="error" instead.
    pytest.importorskip("cpex", reason="cpex not installed — install mellea[hooks]")
    backend = _backend_with_adapter("alora")
    mock_create = AsyncMock(side_effect=RuntimeError("simulated provider error"))
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create

    with (
        patch.object(
            OpenAIBackend,
            "_async_client",
            new_callable=PropertyMock,
            return_value=mock_client,
        ),
        patch("mellea.backends.adapters._core.has_plugins", return_value=True),
        patch(
            "mellea.backends.adapters._core.invoke_hook", new_callable=AsyncMock
        ) as mock_invoke,
    ):
        ctx = ChatContext().add(Message("user", "What is the square root of 4?"))
        mot, _ = await mfuncs.aact(
            Intrinsic("answerability"), ctx, backend, strategy=None
        )
        with pytest.raises(RuntimeError, match="simulated provider error"):
            await mot.avalue()

    payloads = _invocation_complete_payloads(mock_invoke)
    assert len(payloads) == 1
    assert payloads[0].outcome == "error"
    assert isinstance(payloads[0].error, RuntimeError)


async def test_invocation_complete_fires_error_on_empty_choices():
    # An empty-choices response (content filter, proxy error) raises IndexError
    # from self.processing() — must still fire outcome="error", not skip firing.
    pytest.importorskip("cpex", reason="cpex not installed — install mellea[hooks]")
    backend = _backend_with_adapter("alora")
    empty_choices_response = _chat_completion().model_copy(update={"choices": []})
    mock_create = AsyncMock(return_value=empty_choices_response)
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create

    with (
        patch.object(
            OpenAIBackend,
            "_async_client",
            new_callable=PropertyMock,
            return_value=mock_client,
        ),
        patch("mellea.backends.adapters._core.has_plugins", return_value=True),
        patch(
            "mellea.backends.adapters._core.invoke_hook", new_callable=AsyncMock
        ) as mock_invoke,
    ):
        ctx = ChatContext().add(Message("user", "What is the square root of 4?"))
        mot, _ = await mfuncs.aact(
            Intrinsic("answerability"), ctx, backend, strategy=None
        )
        with pytest.raises(IndexError):
            await mot.avalue()

    payloads = _invocation_complete_payloads(mock_invoke)
    assert len(payloads) == 1
    assert payloads[0].outcome == "error"
    assert isinstance(payloads[0].error, IndexError)


async def test_reassigned_weights_fail_loudly():
    """Reassigning `.weights` off the EmbeddedBinding must fail at generation.

    The shim permits attribute mutation, so a caller reassigning `.weights`
    after construction must hit the explicit TypeError rather than silently
    skipping activation and sending an unactivated request (issue #1142).
    """
    backend = _backend_with_adapter("alora")
    adapter = backend._added_adapters["answerability_alora"]
    adapter.weights = ServerMediatedBinding()

    ctx = ChatContext().add(Message("user", "What is the square root of 4?"))
    with pytest.raises(TypeError, match=r"weights must be an EmbeddedBinding"):
        await mfuncs.aact(Intrinsic("answerability"), ctx, backend, strategy=None)
