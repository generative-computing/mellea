# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OpenAIBackend.post_processing — normalized response shape.

Verifies that the streaming path stores a top-level envelope in mot.raw.response
(``{"choices": [...], "usage": ...}``) identical in shape to the non-streaming path.
No API calls are made.
"""

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from mellea.backends.openai import OpenAIBackend
from mellea.core.base import CBlock, ModelOutputThunk


def _make_backend() -> OpenAIBackend:
    """Return an OpenAIBackend with a fake API key."""
    return OpenAIBackend(
        model_id="gpt-4o", api_key="fake-key", base_url="http://localhost:9999/v1"
    )


def _streaming_chunks() -> list[dict]:
    """Minimal choice-level delta chunks as stored by processing()."""
    return [
        {"delta": {"role": "assistant", "content": "hello"}, "finish_reason": None},
        {"delta": {"content": " world"}, "finish_reason": "stop"},
    ]


def _non_streaming_response() -> ChatCompletion:
    """Minimal real ChatCompletion for the non-streaming path."""
    return ChatCompletion(
        id="test",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content="hello world"),
            )
        ],
        created=0,
        model="gpt-4o",
        object="chat.completion",
    )


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
            mot, tools={}, conversation=[], thinking=None, seed=None, _format=None
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
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "total_tokens": 7,
        }

        await backend.post_processing(
            mot, tools={}, conversation=[], thinking=None, seed=None, _format=None
        )

        response = mot.raw.response
        assert isinstance(response, dict)
        assert response["usage"] == {
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "total_tokens": 7,
        }

    @pytest.mark.asyncio
    async def test_non_streaming_response_shape_unchanged(self):
        """Non-streaming path still stores a top-level shape with 'choices' (regression guard)."""
        backend = _make_backend()
        mot = ModelOutputThunk(value="hello world")
        mot._call.action = CBlock("q")
        mot._call.model_options = {}
        mot.raw.response = _non_streaming_response().model_dump()

        await backend.post_processing(
            mot, tools={}, conversation=[], thinking=None, seed=None, _format=None
        )

        response = mot.raw.response
        assert isinstance(response, dict)
        assert "choices" in response
        assert response["choices"][0]["message"]["content"] == "hello world"
