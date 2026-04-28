"""Unit tests verifying that documents on Messages reach the API call.

Covers OpenAIBackend _generate_from_chat_context_standard to ensure that
Message.documents are extracted via messages_to_docs() and forwarded as
extra_body={"documents": [...]}.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from mellea.backends.openai import OpenAIBackend
from mellea.stdlib.components import Message
from mellea.stdlib.components.docs import Document
from mellea.stdlib.context import ChatContext


def _make_openai_backend() -> OpenAIBackend:
    return OpenAIBackend(
        model_id="gpt-4o", api_key="fake-key", base_url="http://localhost:9999/v1"
    )


def _build_context_with_docs(docs: list[Document] | None = None) -> ChatContext:
    ctx = ChatContext()
    ctx = ctx.add(Message("user", "What is in the document?", documents=docs))
    return ctx


def _fake_openai_response() -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = "ok"
    resp.choices[0].message.tool_calls = None
    resp.choices[0].finish_reason = "stop"
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    resp.usage.prompt_tokens_details = None
    return resp


@pytest.mark.integration
class TestOpenAIDocumentPassthrough:
    def test_documents_passed_as_extra_body(self):
        backend = _make_openai_backend()
        docs = [
            Document(text="The sky is blue.", title="Facts", doc_id="d1"),
            Document(text="Water is wet."),
        ]
        ctx = _build_context_with_docs(docs)

        captured_kwargs: dict = {}

        async def fake_create(**kwargs):
            captured_kwargs.update(kwargs)
            return _fake_openai_response()

        mock_client = MagicMock()
        mock_client.chat.completions.create = fake_create

        with patch.object(
            type(backend),
            "_async_client",
            new_callable=lambda: property(lambda self: mock_client),
        ):
            action = Message("user", "Summarise the documents.")
            asyncio.get_event_loop().run_until_complete(
                backend._generate_from_chat_context_standard(action, ctx)
            )

        assert "extra_body" in captured_kwargs
        assert captured_kwargs["extra_body"] == {
            "documents": [
                {"text": "The sky is blue.", "title": "Facts", "doc_id": "d1"},
                {"text": "Water is wet."},
            ]
        }

    def test_no_documents_no_extra_body(self):
        backend = _make_openai_backend()
        ctx = _build_context_with_docs(None)

        captured_kwargs: dict = {}

        async def fake_create(**kwargs):
            captured_kwargs.update(kwargs)
            return _fake_openai_response()

        mock_client = MagicMock()
        mock_client.chat.completions.create = fake_create

        with patch.object(
            type(backend),
            "_async_client",
            new_callable=lambda: property(lambda self: mock_client),
        ):
            action = Message("user", "Hello.")
            asyncio.get_event_loop().run_until_complete(
                backend._generate_from_chat_context_standard(action, ctx)
            )

        assert captured_kwargs.get("extra_body") is None
