"""Unit tests verifying that documents on Messages render correctly through each backend.

Each test mocks the backend's API call, sends a Message with documents through
generate_from_context, and asserts the rendered conversation contains the expected
document format.
"""

import warnings
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import litellm
import ollama
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from mellea.backends.litellm import LiteLLMBackend
from mellea.backends.ollama import OllamaModelBackend
from mellea.backends.openai import OpenAIBackend
from mellea.stdlib.components import Message
from mellea.stdlib.components.docs.document import Document
from mellea.stdlib.context import ChatContext

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DOC = Document("The answer is 42.", title="Guide", doc_id="1")
USER_MSG = Message("user", "What is the answer?", documents=[DOC])


def _make_context() -> ChatContext:
    return ChatContext().add(USER_MSG)


def _openai_chat_completion() -> ChatCompletion:
    return ChatCompletion(
        id="test",
        created=0,
        model="test-model",
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content="42"),
            )
        ],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=1, total_tokens=11),
    )


def _ollama_chat_response() -> ollama.ChatResponse:
    return ollama.ChatResponse(
        model="granite3.3:8b",
        created_at="2024-01-01T00:00:00Z",
        message=ollama.Message(role="assistant", content="42"),
        done=True,
        done_reason="stop",
    )


def _litellm_model_response() -> litellm.ModelResponse:
    return litellm.ModelResponse(
        id="test",
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "42"},
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        model="openai/gpt-4o",
    )


def _assert_document_in_messages(messages: list[dict]) -> None:
    """Assert that at least one message contains the rendered document."""
    all_content = " ".join(m.get("content", "") for m in messages)
    assert "[Document 1]" in all_content, f"Document header not found in: {all_content}"
    assert "Guide:" in all_content, f"Document title not found in: {all_content}"
    assert "The answer is 42." in all_content, (
        f"Document text not found in: {all_content}"
    )


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_renders_documents_in_prompt():
    """OpenAI backend includes rendered documents in the messages sent to the API."""
    backend = OpenAIBackend(
        model_id="gpt-4o", api_key="fake-key", base_url="http://localhost:9999/v1"
    )

    mock_create = AsyncMock(return_value=_openai_chat_completion())
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create

    with patch.object(
        OpenAIBackend,
        "_async_client",
        new_callable=PropertyMock,
        return_value=mock_client,
    ):
        ctx = _make_context()
        mot, _ = await backend._generate_from_context(
            Message("user", "Follow up question"), ctx
        )
        await mot.avalue()

    mock_create.assert_called_once()
    messages = mock_create.call_args.kwargs["messages"]
    _assert_document_in_messages(messages)


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


def _make_ollama_backend() -> OllamaModelBackend:
    """Return an OllamaModelBackend with all network calls patched."""
    with (
        patch.object(OllamaModelBackend, "_check_ollama_server", return_value=True),
        patch.object(OllamaModelBackend, "_pull_ollama_model", return_value=True),
        patch("mellea.backends.ollama.ollama.Client", return_value=MagicMock()),
        patch("mellea.backends.ollama.ollama.AsyncClient", return_value=MagicMock()),
    ):
        return OllamaModelBackend(model_id="granite3.3:8b")


@pytest.mark.asyncio
async def test_ollama_renders_documents_in_prompt():
    """Ollama backend includes rendered documents in the messages sent to the API."""
    backend = _make_ollama_backend()

    mock_chat = AsyncMock(return_value=_ollama_chat_response())

    with patch.object(
        OllamaModelBackend, "_async_client", new_callable=PropertyMock
    ) as mock_client_prop:
        mock_async_client = MagicMock()
        mock_async_client.chat = mock_chat
        mock_client_prop.return_value = mock_async_client

        ctx = _make_context()
        mot, _ = await backend._generate_from_context(
            Message("user", "Follow up question"), ctx
        )
        await mot.avalue()

    mock_chat.assert_called_once()
    messages = mock_chat.call_args.kwargs["messages"]
    _assert_document_in_messages(messages)


# ---------------------------------------------------------------------------
# LiteLLM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_litellm_renders_documents_in_prompt():
    """LiteLLM backend includes rendered documents in the messages sent to the API."""
    backend = LiteLLMBackend(
        model_id="openai/gpt-4o", base_url="http://localhost:9999/v1"
    )

    with patch(
        "mellea.backends.litellm.litellm.acompletion", new_callable=AsyncMock
    ) as mock_acompletion:
        mock_acompletion.return_value = _litellm_model_response()

        ctx = _make_context()
        mot, _ = await backend._generate_from_context(
            Message("user", "Follow up question"), ctx
        )
        await mot.avalue()

    mock_acompletion.assert_called_once()
    messages = mock_acompletion.call_args.kwargs["messages"]
    _assert_document_in_messages(messages)


# ---------------------------------------------------------------------------
# Watsonx
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_watsonx_renders_documents_in_prompt():
    """Watsonx backend includes rendered documents in the messages sent to the API."""
    from mellea.backends.watsonx import WatsonxAIBackend

    mock_achat_response = {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "42"},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with (
            patch("mellea.backends.watsonx.Credentials"),
            patch("mellea.backends.watsonx.ModelInference") as mock_model_cls,
        ):
            mock_model_instance = MagicMock()
            mock_model_cls.return_value = mock_model_instance
            # achat must return a non-AsyncIterator (dict is fine)
            mock_model_instance.achat = AsyncMock(return_value=mock_achat_response)

            with patch.object(
                WatsonxAIBackend,
                "_model",
                new_callable=PropertyMock,
                return_value=mock_model_instance,
            ):
                backend = WatsonxAIBackend(
                    model_id="ibm/granite-3-8b-instruct",
                    api_key="fake-key",
                    base_url="https://fake.cloud.ibm.com",
                    project_id="fake-project",
                )

                ctx = _make_context()
                mot, _ = await backend._generate_from_context(
                    Message("user", "Follow up question"), ctx
                )
                await mot.avalue()

    mock_model_instance.achat.assert_called_once()
    messages = mock_model_instance.achat.call_args.kwargs["messages"]
    _assert_document_in_messages(messages)


# ---------------------------------------------------------------------------
# HuggingFace
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_huggingface_renders_documents_in_prompt():
    """HuggingFace backend includes rendered documents in the chat template input."""
    import torch
    from transformers.generation.utils import GenerateDecoderOnlyOutput

    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_device = torch.device("cpu")

    # apply_chat_template returns input_ids tensor
    fake_input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    mock_tokenizer.apply_chat_template = MagicMock(return_value=fake_input_ids)

    # model.generate returns a real GenerateDecoderOnlyOutput (not a MagicMock)
    # to avoid AsyncIterator detection in send_to_queue
    real_output = GenerateDecoderOnlyOutput(
        sequences=torch.tensor([[1, 2, 3, 4, 5, 6, 7]]),
        scores=None,
        logits=None,
        attentions=None,
        hidden_states=None,
        past_key_values=None,
    )
    mock_model.generate = MagicMock(return_value=real_output)

    # decode returns the "answer"
    mock_tokenizer.decode = MagicMock(return_value="42")

    with (
        patch("mellea.backends.huggingface.llguidance") as mock_llg,
        patch("mellea.backends.huggingface.set_seed"),
    ):
        mock_llg.hf.from_tokenizer.return_value = MagicMock(vocab_size=32000)
        mock_tokenizer._tokenizer = MagicMock()
        mock_tokenizer._tokenizer.get_vocab_size.return_value = 32000

        from mellea.backends.huggingface import LocalHFBackend

        backend = LocalHFBackend(
            model_id="ibm-granite/granite-3.3-8b-instruct",
            custom_config=(mock_tokenizer, mock_model, mock_device),
        )

        ctx = _make_context()
        mot, _ = await backend._generate_from_context(
            Message("user", "Follow up question"), ctx
        )
        await mot.avalue()

    mock_tokenizer.apply_chat_template.assert_called_once()
    chat_list = mock_tokenizer.apply_chat_template.call_args[0][0]
    # chat_list is a list of {"role": ..., "content": ...} dicts
    all_content = " ".join(entry.get("content", "") for entry in chat_list)
    assert "[Document 1]" in all_content, f"Document header not found in: {all_content}"
    assert "Guide:" in all_content, f"Document title not found in: {all_content}"
    assert "The answer is 42." in all_content, (
        f"Document text not found in: {all_content}"
    )
