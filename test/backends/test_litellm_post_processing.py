# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LiteLLMBackend.post_processing — normalized response shape.

Verifies that the streaming path stores a top-level envelope in mot.raw.response
(`{"choices": [...], "usage": ...}`) identical in shape to the non-streaming path.
No API calls are made.
"""

import pytest

pytest.importorskip("litellm", reason="litellm not installed — install mellea[litellm]")

from litellm import Choices, Message, ModelResponse

from mellea.backends.litellm import LiteLLMBackend
from mellea.core.base import CBlock, ModelOutputThunk


def _make_backend() -> "LiteLLMBackend":
    """Return a LiteLLMBackend with a fake base URL."""
    return LiteLLMBackend(
        model_id="hosted_vllm/qwen3", base_url="http://localhost:9997"
    )


def _streaming_chunks() -> list[dict]:
    """Minimal choice-level delta chunks as stored by LiteLLMBackend.processing()."""
    return [
        {"delta": {"role": "assistant", "content": "hello"}, "finish_reason": None},
        {"delta": {"content": " world"}, "finish_reason": "stop"},
    ]


def _non_streaming_response() -> dict:
    """Minimal non-streaming LiteLLM response dict."""
    msg = Message(content="hello world", role="assistant")
    choice = Choices(finish_reason="stop", index=0, message=msg)
    response = ModelResponse(
        id="test",
        choices=[choice],
        created=0,
        model="hosted_vllm/qwen3",
        object="chat.completion",
        usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    )
    return dict(response)


class TestPostProcessingStreamingShape:
    @pytest.mark.asyncio
    async def test_streaming_response_has_top_level_choices_envelope(self):
        """Streaming path stores a top-level envelope with a 'choices' list."""
        backend = _make_backend()
        mot = ModelOutputThunk(value="hello world")
        mot._call.action = CBlock("q")
        mot._call.model_options = {}
        mot.raw.streamed_chunks = _streaming_chunks()

        await backend.post_processing(
            mot, conversation=[], tools={}, thinking=None, _format=None
        )

        response = mot.raw.response
        assert isinstance(response, dict)
        assert "choices" in response
        assert isinstance(response["choices"], list)
        assert len(response["choices"]) == 1
        assert response["choices"][0]["message"]["content"] == "hello world"

    @pytest.mark.asyncio
    async def test_streaming_response_includes_usage(self):
        """Streaming envelope carries usage from mot.generation.usage."""
        backend = _make_backend()
        mot = ModelOutputThunk(value="hello world")
        mot._call.action = CBlock("q")
        mot._call.model_options = {}
        mot.raw.streamed_chunks = _streaming_chunks()
        mot.generation.usage = {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }

        await backend.post_processing(
            mot, conversation=[], tools={}, thinking=None, _format=None
        )

        response = mot.raw.response
        assert isinstance(response, dict)
        assert response["usage"] == {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }

    @pytest.mark.asyncio
    async def test_non_streaming_response_shape_unchanged(self):
        """Non-streaming path still stores a top-level shape with 'choices' (regression guard)."""
        backend = _make_backend()
        mot = ModelOutputThunk(value="hello world")
        mot._call.action = CBlock("q")
        mot._call.model_options = {}
        mot.raw.response = _non_streaming_response()

        await backend.post_processing(
            mot, conversation=[], tools={}, thinking=None, _format=None
        )

        response = mot.raw.response
        assert isinstance(response, dict)
        assert "choices" in response
        assert response["choices"][0]["message"]["content"] == "hello world"
