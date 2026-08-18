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

from mellea.backends.adapters._core import EmbeddedBinding
from mellea.backends.adapters.adapter import EmbeddedIntrinsicAdapter
from mellea.backends.openai import OpenAIBackend
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
