"""Unit tests for LiteLLMBackend mot._thinking population.

Covers the vLLM case where the wire key is ``"reasoning"`` instead of
``"reasoning_content"``, and the case where LiteLLM has already normalised
it to ``"reasoning_content"`` (so both keys are exercised).
"""

import pytest

pytest.importorskip("litellm", reason="litellm not installed — install mellea[litellm]")

from litellm.types.utils import (
    Choices,
    Delta,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)

from mellea.backends.litellm import LiteLLMBackend
from mellea.core import ModelOutputThunk


def _make_non_streaming_chunk(
    content: str, reasoning_key: str, reasoning_value: str
) -> ModelResponse:
    """Build a minimal non-streaming ModelResponse with a custom reasoning key."""
    msg = Message(content=content, role="assistant")
    msg[reasoning_key] = reasoning_value
    choice = Choices(finish_reason="stop", index=0, message=msg)
    return ModelResponse(
        id="test",
        choices=[choice],
        created=0,
        model="openai/qwen3",
        object="chat.completion",
    )


def _make_streaming_chunk(
    content: str, reasoning_key: str, reasoning_value: str
) -> ModelResponseStream:
    """Build a minimal streaming delta chunk with a custom reasoning key."""
    delta = Delta(content=content)
    delta[reasoning_key] = reasoning_value
    chunk_choice = StreamingChoices(finish_reason=None, index=0, delta=delta)
    return ModelResponseStream(
        id="test", choices=[chunk_choice], created=0, model="openai/qwen3"
    )


@pytest.fixture()
def backend() -> LiteLLMBackend:
    return LiteLLMBackend(model_id="openai/qwen3", base_url="http://localhost:8000/v1")


def _fresh_mot() -> ModelOutputThunk:
    mot: ModelOutputThunk = ModelOutputThunk(None)
    mot._thinking = ""
    mot._underlying_value = ""
    mot._meta = {}
    return mot


# ---------------------------------------------------------------------------
# Non-streaming path
# ---------------------------------------------------------------------------


async def test_processing_non_streaming_reasoning_content_key(backend: LiteLLMBackend):
    """reasoning_content (normalised key) is captured correctly."""
    mot = _fresh_mot()
    chunk = _make_non_streaming_chunk(
        content="Paris",
        reasoning_key="reasoning_content",
        reasoning_value="France has its capital in Paris.",
    )
    await backend.processing(mot, chunk)
    assert mot._thinking == "France has its capital in Paris."
    assert mot._underlying_value == "Paris"


async def test_processing_non_streaming_reasoning_raw_key(backend: LiteLLMBackend):
    """Fallback: vLLM 'reasoning' key (not normalised by older LiteLLM) is captured."""
    mot = _fresh_mot()
    chunk = _make_non_streaming_chunk(
        content="Paris",
        reasoning_key="reasoning",
        reasoning_value="France has its capital in Paris.",
    )
    await backend.processing(mot, chunk)
    assert mot._thinking == "France has its capital in Paris."
    assert mot._underlying_value == "Paris"


async def test_processing_non_streaming_reasoning_content_wins_over_reasoning(
    backend: LiteLLMBackend,
):
    """reasoning_content takes priority when both keys are present."""
    mot = _fresh_mot()
    msg = Message(content="Paris", role="assistant")
    msg["reasoning_content"] = "from_reasoning_content"
    msg["reasoning"] = "from_reasoning"
    choice = Choices(finish_reason="stop", index=0, message=msg)
    chunk = ModelResponse(
        id="test",
        choices=[choice],
        created=0,
        model="openai/qwen3",
        object="chat.completion",
    )
    await backend.processing(mot, chunk)
    assert mot._thinking == "from_reasoning_content"


async def test_processing_non_streaming_no_reasoning(backend: LiteLLMBackend):
    """No reasoning key — thinking stays empty string, content is captured."""
    mot = _fresh_mot()
    chunk = _make_non_streaming_chunk(
        content="Paris",
        reasoning_key="unrelated_key",
        reasoning_value="should be ignored",
    )
    await backend.processing(mot, chunk)
    assert mot._thinking == ""
    assert mot._underlying_value == "Paris"


# ---------------------------------------------------------------------------
# Streaming path
# ---------------------------------------------------------------------------


async def test_processing_streaming_reasoning_content_key(backend: LiteLLMBackend):
    """Streaming: reasoning_content key is accumulated across chunks."""
    mot = _fresh_mot()
    for text in ("chunk1 ", "chunk2"):
        stream_chunk = _make_streaming_chunk(
            content="", reasoning_key="reasoning_content", reasoning_value=text
        )
        await backend.processing(mot, stream_chunk)
    assert mot._thinking == "chunk1 chunk2"


async def test_processing_streaming_reasoning_raw_key(backend: LiteLLMBackend):
    """Streaming fallback: vLLM 'reasoning' key is accumulated across chunks."""
    mot = _fresh_mot()
    for text in ("chunk1 ", "chunk2"):
        stream_chunk = _make_streaming_chunk(
            content="", reasoning_key="reasoning", reasoning_value=text
        )
        await backend.processing(mot, stream_chunk)
    assert mot._thinking == "chunk1 chunk2"


async def test_processing_streaming_reasoning_content_wins_over_reasoning(
    backend: LiteLLMBackend,
):
    """Streaming: reasoning_content takes priority when both keys are present."""
    mot = _fresh_mot()
    delta = Delta(content="")
    delta["reasoning_content"] = "from_reasoning_content"
    delta["reasoning"] = "from_reasoning"
    chunk_choice = StreamingChoices(finish_reason=None, index=0, delta=delta)
    stream_chunk = ModelResponseStream(
        id="test", choices=[chunk_choice], created=0, model="openai/qwen3"
    )
    await backend.processing(mot, stream_chunk)
    assert mot._thinking == "from_reasoning_content"


async def test_processing_streaming_no_reasoning(backend: LiteLLMBackend):
    """Streaming: no reasoning key — thinking stays empty string."""
    mot = _fresh_mot()
    stream_chunk = _make_streaming_chunk(
        content="Paris", reasoning_key="unrelated_key", reasoning_value="ignored"
    )
    await backend.processing(mot, stream_chunk)
    assert mot._thinking == ""
    assert mot._underlying_value == "Paris"
