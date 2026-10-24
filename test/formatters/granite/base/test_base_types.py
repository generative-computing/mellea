# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from mellea.formatters.granite.base.types import Document, GraniteChatCompletion

_DOCS = [{"text": "The square root of 4 is 2.", "doc_id": "0"}]


def _request(top_level, in_kwargs) -> dict:
    extra_body: dict = {"chat_template_kwargs": {"documents": in_kwargs}}
    if top_level is not None:
        extra_body["documents"] = top_level
    return {
        "messages": [{"role": "user", "content": "What is the square root of 4?"}],
        "extra_body": extra_body,
    }


class TestDocumentsInChatTemplateKwargs:
    """Documents may also ride in chat_template_kwargs (llama-server reads them there)."""

    def test_identical_copies_accepted(self):
        # OpenAIBackend sends documents both top-level and in
        # chat_template_kwargs; the request it builds must validate.
        completion = GraniteChatCompletion.model_validate(_request(_DOCS, _DOCS))

        assert completion.extra_body.documents == [
            Document.model_validate(d) for d in _DOCS
        ]
        assert not hasattr(completion.extra_body.chat_template_kwargs, "documents")

    def test_differing_copies_rejected(self):
        other = [{"text": "Something else entirely.", "doc_id": "0"}]
        with pytest.raises(ValueError, match="Conflicting values of documents"):
            GraniteChatCompletion.model_validate(_request(_DOCS, other))

    def test_non_list_rejected(self):
        with pytest.raises(ValueError, match="is not a list"):
            GraniteChatCompletion.model_validate(_request(_DOCS, "not a list"))
