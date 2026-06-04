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


async def test_processing_non_streaming_empty_reasoning_content_does_not_fall_back(
    backend: LiteLLMBackend,
):
    """Empty-string reasoning_content wins — does not fall back to reasoning key.

    Validates that the is-None guard (not ``or``) is used: an empty-string
    ``reasoning_content`` chunk is preserved as-is, not silently replaced by the
    fallback ``reasoning`` value.
    """
    mot = _fresh_mot()
    msg = Message(content="Paris", role="assistant")
    msg["reasoning_content"] = ""
    msg["reasoning"] = "should not appear"
    choice = Choices(finish_reason="stop", index=0, message=msg)
    chunk = ModelResponse(
        id="test",
        choices=[choice],
        created=0,
        model="openai/qwen3",
        object="chat.completion",
    )
    await backend.processing(mot, chunk)
    assert mot._thinking == ""


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


async def test_processing_streaming_empty_reasoning_content_does_not_fall_back(
    backend: LiteLLMBackend,
):
    """Streaming: empty-string reasoning_content wins — does not fall back to reasoning key.

    Validates that the is-None guard (not ``or``) is used in the streaming branch too.
    """
    mot = _fresh_mot()
    delta = Delta(content="")
    delta["reasoning_content"] = ""
    delta["reasoning"] = "should not appear"
    chunk_choice = StreamingChoices(finish_reason=None, index=0, delta=delta)
    stream_chunk = ModelResponseStream(
        id="test", choices=[chunk_choice], created=0, model="openai/qwen3"
    )
    await backend.processing(mot, stream_chunk)
    assert mot._thinking == ""


# ---------------------------------------------------------------------------
# Parameter-passing tests (mock litellm.acompletion — no server required)
# ---------------------------------------------------------------------------


def _make_mock_response() -> ModelResponse:
    """Minimal non-streaming ModelResponse that survives processing() and post_processing()."""
    msg = Message(content="ok", role="assistant")
    choice = Choices(finish_reason="stop", index=0, message=msg)
    return ModelResponse(
        id="test",
        choices=[choice],
        created=0,
        model="hosted_vllm/qwen3",
        object="chat.completion",
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


@pytest.fixture()
def chat_backend() -> LiteLLMBackend:
    return LiteLLMBackend(
        model_id="hosted_vllm/qwen3", base_url="http://localhost:9997"
    )


async def _call_and_capture(backend: LiteLLMBackend, model_options: dict) -> dict:
    """Call _generate_from_chat_context_standard with mocked litellm.acompletion.

    Returns the kwargs dict that litellm.acompletion was called with.
    """
    from unittest.mock import AsyncMock, patch

    from mellea.core import CBlock
    from mellea.stdlib.components import Message as MelleaMessage
    from mellea.stdlib.context import ChatContext

    ctx = ChatContext().add(MelleaMessage("user", "Hello"))
    action = CBlock(value="Test")
    mock_response = _make_mock_response()

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acomplete:
        mock_acomplete.return_value = mock_response
        mot = await backend._generate_from_chat_context_standard(
            action, ctx, model_options=model_options
        )
        await mot.avalue()

    assert mock_acomplete.called, "litellm.acompletion was never called"
    return mock_acomplete.call_args.kwargs


async def test_thinking_true_sets_reasoning_effort_and_enable_thinking(
    chat_backend: LiteLLMBackend,
) -> None:
    """THINKING=True sets reasoning_effort='medium' AND extra_body.chat_template_kwargs.enable_thinking=True."""
    from mellea.backends import ModelOption

    kwargs = await _call_and_capture(chat_backend, {ModelOption.THINKING: True})

    assert kwargs.get("reasoning_effort") == "medium", (
        "reasoning_effort should be 'medium' for THINKING=True"
    )
    assert (
        kwargs.get("extra_body", {})
        .get("chat_template_kwargs", {})
        .get("enable_thinking")
        is True
    ), "extra_body.chat_template_kwargs.enable_thinking should be True"


async def test_thinking_false_omits_reasoning_effort_and_sets_disable(
    chat_backend: LiteLLMBackend,
) -> None:
    """THINKING=False: reasoning_effort absent, extra_body.chat_template_kwargs.enable_thinking=False."""
    from mellea.backends import ModelOption

    kwargs = await _call_and_capture(chat_backend, {ModelOption.THINKING: False})

    assert "reasoning_effort" not in kwargs, (
        "reasoning_effort must not be sent for THINKING=False (invalid value)"
    )
    assert (
        kwargs.get("extra_body", {})
        .get("chat_template_kwargs", {})
        .get("enable_thinking")
        is False
    ), "extra_body.chat_template_kwargs.enable_thinking should be False"


async def test_thinking_string_sets_only_reasoning_effort(
    chat_backend: LiteLLMBackend,
) -> None:
    """THINKING='high' sets reasoning_effort='high' but does NOT set extra_body.enable_thinking."""
    from mellea.backends import ModelOption

    kwargs = await _call_and_capture(chat_backend, {ModelOption.THINKING: "high"})

    assert kwargs.get("reasoning_effort") == "high", (
        "reasoning_effort should be 'high' for THINKING='high'"
    )
    assert "enable_thinking" not in kwargs.get("extra_body", {}).get(
        "chat_template_kwargs", {}
    ), "enable_thinking must not be set for string THINKING values"


async def test_thinking_unset_sends_neither(chat_backend: LiteLLMBackend) -> None:
    """No THINKING option: neither reasoning_effort nor extra_body.enable_thinking is sent."""
    kwargs = await _call_and_capture(chat_backend, {})

    assert "reasoning_effort" not in kwargs, (
        "reasoning_effort should not be present when THINKING is not set"
    )
    assert "enable_thinking" not in kwargs.get("extra_body", {}).get(
        "chat_template_kwargs", {}
    ), "enable_thinking should not be present when THINKING is not set"


async def test_api_base_passed_to_litellm(chat_backend: LiteLLMBackend) -> None:
    """api_base is forwarded to litellm.acompletion matching the backend's base_url."""
    kwargs = await _call_and_capture(chat_backend, {})

    assert kwargs.get("api_base") == "http://localhost:9997", (
        "api_base must equal the backend base_url"
    )


async def test_thinking_true_with_user_extra_body_merged(
    chat_backend: LiteLLMBackend,
) -> None:
    """THINKING=True + user extra_body: enable_thinking and user keys both survive in extra_body."""
    from mellea.backends import ModelOption

    kwargs = await _call_and_capture(
        chat_backend,
        {ModelOption.THINKING: True, "extra_body": {"guided_json": {"type": "string"}}},
    )

    eb = kwargs.get("extra_body", {})
    assert eb.get("chat_template_kwargs", {}).get("enable_thinking") is True, (
        "enable_thinking must survive the merge with user-supplied extra_body"
    )
    assert eb.get("guided_json") == {"type": "string"}, (
        "user-supplied extra_body keys must be preserved alongside enable_thinking"
    )
    assert kwargs.get("reasoning_effort") == "medium"


async def test_thinking_true_with_user_chat_template_kwargs_deep_merged(
    chat_backend: LiteLLMBackend,
) -> None:
    """THINKING=True + user extra_body.chat_template_kwargs: both enable_thinking and user CTK keys survive."""
    from mellea.backends import ModelOption

    kwargs = await _call_and_capture(
        chat_backend,
        {
            ModelOption.THINKING: True,
            "extra_body": {"chat_template_kwargs": {"adapter_name": "my-adapter"}},
        },
    )

    ctk = kwargs.get("extra_body", {}).get("chat_template_kwargs", {})
    assert ctk.get("enable_thinking") is True, (
        "enable_thinking must survive when user also supplies chat_template_kwargs"
    )
    assert ctk.get("adapter_name") == "my-adapter", (
        "user-supplied chat_template_kwargs keys must be preserved"
    )


async def test_user_api_base_in_model_options_takes_precedence(
    chat_backend: LiteLLMBackend,
) -> None:
    """api_base in model_options must win over backend base_url and not cause a TypeError."""
    kwargs = await _call_and_capture(
        chat_backend, {"api_base": "http://custom:1234/v1"}
    )

    assert kwargs.get("api_base") == "http://custom:1234/v1", (
        "user-supplied api_base in model_options must take precedence over backend base_url"
    )


async def test_no_api_base_forwarded_when_base_url_not_set() -> None:
    """Backend constructed without base_url must not forward api_base (cloud-provider safety)."""
    backend = LiteLLMBackend(model_id="anthropic/claude-3-5-sonnet-20241022")
    kwargs = await _call_and_capture(backend, {})

    assert kwargs.get("api_base") is None, (
        "api_base must not be forwarded for cloud providers with default base_url — "
        "LiteLLM must infer the endpoint from the model prefix"
    )


async def test_extra_body_mutation_does_not_corrupt_caller_dict(
    chat_backend: LiteLLMBackend,
) -> None:
    """Reusing a model_options dict across two calls must not lose chat_template_kwargs."""
    from mellea.backends import ModelOption

    model_opts = {
        ModelOption.THINKING: True,
        "extra_body": {"chat_template_kwargs": {"adapter_name": "persistent"}},
    }

    kwargs1 = await _call_and_capture(chat_backend, model_opts)
    kwargs2 = await _call_and_capture(chat_backend, model_opts)

    for call_n, kwargs in enumerate((kwargs1, kwargs2), start=1):
        ctk = kwargs.get("extra_body", {}).get("chat_template_kwargs", {})
        assert ctk.get("adapter_name") == "persistent", (
            f"call {call_n}: adapter_name silently lost — caller dict was mutated"
        )
        assert ctk.get("enable_thinking") is True, (
            f"call {call_n}: enable_thinking missing"
        )
