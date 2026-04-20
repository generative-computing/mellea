"""Unit tests for the OpenAI backend intrinsic generation path — no server required.

Mocks the OpenAI async client to verify that ``_generate_from_intrinsic`` correctly:
- injects ``intrinsic_name`` into ``chat_template_kwargs``
- applies the ``IntrinsicsResultProcessor`` to the raw response
- forwards ``temperature`` and ``seed`` from model options
- raises when no adapter is registered or streaming is requested
"""

import json
import pathlib
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
import yaml
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice, ChoiceLogprobs
from openai.types.chat.chat_completion_token_logprob import (
    ChatCompletionTokenLogprob,
    TopLogprob,
)
from openai.types.completion_usage import CompletionUsage

from mellea.backends import ModelOption
from mellea.backends.adapters.adapter import EmbeddedIntrinsicAdapter
from mellea.backends.openai import OpenAIBackend
from mellea.stdlib import functional as mfuncs
from mellea.stdlib.components import Intrinsic, Message
from mellea.stdlib.context import ChatContext

_TEST_DIR = pathlib.Path(__file__).parent
_INTRINSICS_DATA = _TEST_DIR / "test_adapters" / "intrinsics-data"

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

# Minimal config: no transformations, no logprobs.  Good enough for most tests
# that only inspect the API call (chat_template_kwargs, temp/seed forwarding).
_SIMPLE_CONFIG = {
    "model": None,
    "response_format": None,
    "transformations": None,
    "instruction": None,
    "parameters": {"max_completion_tokens": 64},
    "sentence_boundaries": None,
}

# Full answerability config with likelihood + nest transformations.
_ANSWERABILITY_CONFIG = yaml.safe_load(
    (_INTRINSICS_DATA / "answerability.yaml").read_text()
)

# ---------------------------------------------------------------------------
# Canned responses
# ---------------------------------------------------------------------------


def _simple_chat_completion(content: str = '{"result": "ok"}') -> ChatCompletion:
    """Build a minimal ChatCompletion with no logprobs."""
    return ChatCompletion(
        id="test-simple",
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


def _answerability_chat_completion() -> ChatCompletion:
    """Build a ChatCompletion that the answerability result processor can parse.

    The likelihood transformation reads top_logprobs to compute an expected value.
    """
    return ChatCompletion(
        id="test-ans",
        created=0,
        model="granite-switch",
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content='"answerable"'),
                logprobs=ChoiceLogprobs(
                    content=[
                        ChatCompletionTokenLogprob(
                            token='"',
                            logprob=-0.01,
                            top_logprobs=[TopLogprob(token='"', logprob=-0.01)],
                            bytes=None,
                        ),
                        ChatCompletionTokenLogprob(
                            token="answerable",
                            logprob=-0.05,
                            top_logprobs=[
                                TopLogprob(token="answerable", logprob=-0.05),
                                TopLogprob(token="unanswerable", logprob=-5.0),
                            ],
                            bytes=None,
                        ),
                        ChatCompletionTokenLogprob(
                            token='"',
                            logprob=-0.001,
                            top_logprobs=[TopLogprob(token='"', logprob=-0.001)],
                            bytes=None,
                        ),
                    ]
                ),
            )
        ],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=3, total_tokens=13),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_backend_with_adapter(
    config: dict, *, model_options: dict | None = None
) -> OpenAIBackend:
    """Return an OpenAIBackend with a registered embedded answerability adapter."""
    backend = OpenAIBackend(
        model_id="granite-switch",
        api_key="fake-key",
        base_url="http://localhost:9999/v1",
        model_options=model_options,
    )
    adapter = EmbeddedIntrinsicAdapter(
        intrinsic_name="answerability", config=config, technology="alora"
    )
    backend.add_adapter(adapter)
    return backend


def _make_context() -> ChatContext:
    """Return a simple two-turn chat context."""
    return ChatContext().add(Message("user", "What is the square root of 4?"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_chat_template_kwargs_set():
    """_generate_from_intrinsic injects intrinsic_name into chat_template_kwargs."""
    backend = _make_backend_with_adapter(_SIMPLE_CONFIG)
    ctx = _make_context()
    mock_create = AsyncMock(return_value=_simple_chat_completion())

    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create

    with patch.object(
        OpenAIBackend,
        "_async_client",
        new_callable=PropertyMock,
        return_value=mock_client,
    ):
        mot, _ = await mfuncs.aact(
            Intrinsic("answerability"), ctx, backend, strategy=None
        )
        await mot.avalue()

    mock_create.assert_called_once()
    call_kwargs = mock_create.call_args
    extra_body = call_kwargs.kwargs.get("extra_body", {})

    assert "chat_template_kwargs" in extra_body
    assert extra_body["chat_template_kwargs"]["intrinsic_name"] == "answerability"


async def test_result_processor_applied():
    """Full answerability config: likelihood + nest transforms produce the expected JSON."""
    backend = _make_backend_with_adapter(_ANSWERABILITY_CONFIG)
    ctx = _make_context()
    mock_create = AsyncMock(return_value=_answerability_chat_completion())

    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create

    with patch.object(
        OpenAIBackend,
        "_async_client",
        new_callable=PropertyMock,
        return_value=mock_client,
    ):
        mot, _ = await mfuncs.aact(
            Intrinsic("answerability"), ctx, backend, strategy=None
        )
        await mot.avalue()

    # The result processor should have applied likelihood (logprobs → float)
    # and nest (wrap in {"answerability_likelihood": <float>}).
    parsed = json.loads(mot.value)
    assert "answerability_likelihood" in parsed
    score = parsed["answerability_likelihood"]
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


async def test_temperature_and_seed_forwarded():
    """temperature and seed from model_options are forwarded to the API call."""
    backend = _make_backend_with_adapter(_SIMPLE_CONFIG)
    ctx = _make_context()
    mock_create = AsyncMock(return_value=_simple_chat_completion())

    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create

    with patch.object(
        OpenAIBackend,
        "_async_client",
        new_callable=PropertyMock,
        return_value=mock_client,
    ):
        mot, _ = await mfuncs.aact(
            Intrinsic("answerability"),
            ctx,
            backend,
            strategy=None,
            model_options={ModelOption.TEMPERATURE: 0.5, ModelOption.SEED: 42},
        )
        await mot.avalue()

    call_kwargs = mock_create.call_args
    assert call_kwargs.kwargs.get("temperature") == 0.5
    assert call_kwargs.kwargs.get("seed") == 42


async def test_no_adapter_raises_valueerror():
    """Calling an intrinsic with no registered adapter raises ValueError."""
    backend = OpenAIBackend(
        model_id="granite-switch",
        api_key="fake-key",
        base_url="http://localhost:9999/v1",
    )
    ctx = _make_context()

    with pytest.raises(ValueError, match="has no adapter"):
        await mfuncs.aact(Intrinsic("answerability"), ctx, backend, strategy=None)


async def test_streaming_raises():
    """Intrinsics do not support streaming — should raise NotImplementedError."""
    backend = _make_backend_with_adapter(_SIMPLE_CONFIG)
    ctx = _make_context()

    with pytest.raises(NotImplementedError, match="do not support streaming"):
        await mfuncs.aact(
            Intrinsic("answerability"),
            ctx,
            backend,
            strategy=None,
            model_options={ModelOption.STREAM: True},
        )
