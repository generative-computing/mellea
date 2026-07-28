# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for WatsonxAIBackend.post_processing — normalized response shape.

Verifies that the streaming path stores a top-level envelope in mot.raw.response
(``{"choices": [...], "usage": ...}``).  Also explicitly documents that Watsonx
streaming usage is ``None`` (known limitation, canary test).
No live credentials or API calls are made.
"""

import pytest

pytest.importorskip(
    "ibm_watsonx_ai", reason="ibm_watsonx_ai not installed — install mellea[watsonx]"
)

from unittest.mock import patch

from mellea.backends.watsonx import WatsonxAIBackend
from mellea.core.base import CBlock, ModelOutputThunk


def _make_backend(monkeypatch: pytest.MonkeyPatch) -> "WatsonxAIBackend":
    """Build a WatsonxAIBackend with SDK internals mocked out."""
    monkeypatch.delenv("WATSONX_API_KEY", raising=False)
    monkeypatch.delenv("WATSONX_URL", raising=False)
    monkeypatch.delenv("WATSONX_PROJECT_ID", raising=False)
    with (
        patch("mellea.backends.watsonx.Credentials"),
        patch("mellea.backends.watsonx.APIClient"),
        patch("mellea.backends.watsonx.ModelInference"),
    ):
        return WatsonxAIBackend(
            model_id="ibm/granite-4-h-small",
            base_url="https://example.com",
            project_id="test-project",
            api_key="fake-key",
        )


def _streaming_chunks() -> list[dict]:
    """Minimal choice-level delta chunks as stored by WatsonxAIBackend.processing().

    Note: Watsonx's processing() appends `chunk["choices"][0]` (the choice dict),
    not the full top-level chunk, so streamed_chunks holds choice-level dicts.
    `chat_completion_delta_merge` then merges these into a single choice dict.
    """
    return [
        {"delta": {"role": "assistant", "content": "hello"}, "finish_reason": None},
        {"delta": {"content": " world"}, "finish_reason": "stop"},
    ]


class TestPostProcessingStreamingShape:
    @pytest.mark.asyncio
    async def test_streaming_response_has_top_level_choices_envelope(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Streaming path stores a top-level envelope with a 'choices' list."""
        backend = _make_backend(monkeypatch)
        mot = ModelOutputThunk(value="hello world")
        mot._call.action = CBlock("q")
        mot._call.model_options = {}
        mot.raw.streamed_chunks = _streaming_chunks()

        await backend.post_processing(
            mot, conversation=[], tools={}, seed=None, _format=None
        )

        response = mot.raw.response
        assert isinstance(response, dict)
        assert "choices" in response
        assert isinstance(response["choices"], list)
        assert len(response["choices"]) == 1

    @pytest.mark.asyncio
    async def test_streaming_response_usage_is_none(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Watsonx streaming drops usage — mot.raw.response['usage'] is None.

        This test acts as a canary: if the behaviour changes unexpectedly, it will fail.
        """
        backend = _make_backend(monkeypatch)
        mot = ModelOutputThunk(value="hello world")
        mot._call.action = CBlock("q")
        mot._call.model_options = {}
        mot.raw.streamed_chunks = _streaming_chunks()
        # generation.usage is not pre-populated for Watsonx streaming.

        await backend.post_processing(
            mot, conversation=[], tools={}, seed=None, _format=None
        )

        response = mot.raw.response
        assert isinstance(response, dict)
        assert response["usage"] is None
