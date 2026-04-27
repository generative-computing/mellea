"""Tests for _coerce_documents, _coerce_document, _resolve_question, and _resolve_response."""

import warnings

import pytest

from mellea.core import CBlock, ModelOutputThunk
from mellea.stdlib.components import Document, Instruction, Message
from mellea.stdlib.components.intrinsic._util import (
    _coerce_document,
    _coerce_documents,
    _resolve_question,
    _resolve_response,
)
from mellea.stdlib.context import ChatContext


class TestCoerceDocuments:
    def test_all_strings(self):
        result = _coerce_documents(["foo", "bar"])
        assert len(result) == 2
        assert result[0].text == "foo"
        assert result[0].doc_id is None
        assert result[1].text == "bar"
        assert result[1].doc_id is None

    def test_all_documents(self):
        d1 = Document("a", doc_id="x")
        d2 = Document("b", doc_id="y")
        result = _coerce_documents([d1, d2])
        assert result[0] is d1
        assert result[1] is d2

    def test_mixed(self):
        doc = Document("existing", doc_id="x")
        result = _coerce_documents(["new", doc])
        assert result[0].text == "new"
        assert result[0].doc_id is None
        assert result[1] is doc

    def test_auto_doc_id_strings(self):
        result = _coerce_documents(["a", "b", "c"], auto_doc_id=True)
        assert [d.doc_id for d in result] == ["0", "1", "2"]
        assert [d.text for d in result] == ["a", "b", "c"]

    def test_auto_doc_id_preserves_existing(self):
        doc = Document("a", doc_id="mine")
        result = _coerce_documents([doc, "b"], auto_doc_id=True)
        assert result[0].doc_id == "mine"
        assert result[1].doc_id == "1"

    def test_auto_doc_id_warns_on_missing_doc_id(self):
        doc = Document("no id")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _coerce_documents([doc], auto_doc_id=True)
            assert len(w) == 1
            assert "no doc_id" in str(w[0].message)
        assert result[0] is doc

    def test_auto_doc_id_no_warn_when_doc_id_present(self):
        doc = Document("has id", doc_id="0")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _coerce_documents([doc], auto_doc_id=True)
            assert len(w) == 0

    def test_empty(self):
        assert _coerce_documents([]) == []


class TestCoerceDocument:
    def test_string(self):
        result = _coerce_document("hello")
        assert isinstance(result, Document)
        assert result.text == "hello"
        assert result.doc_id is None

    def test_passthrough(self):
        doc = Document("existing", doc_id="1")
        assert _coerce_document(doc) is doc


class TestResolveQuestion:
    def test_explicit_string(self):
        ctx = ChatContext()
        text, returned_ctx = _resolve_question("hello", ctx)
        assert text == "hello"
        assert returned_ctx is ctx

    def test_from_context(self):
        ctx = ChatContext().add(Message("user", "What is 2+2?"))
        text, rewound = _resolve_question(None, ctx)
        assert text == "What is 2+2?"
        assert rewound.is_root_node  # type: ignore[union-attr]

    def test_context_with_prior_messages(self):
        ctx = (
            ChatContext()
            .add(Message("user", "first"))
            .add(Message("assistant", "reply"))
            .add(Message("user", "second"))
        )
        text, rewound = _resolve_question(None, ctx)
        assert text == "second"
        # Rewound context should end with the assistant reply
        last = rewound.last_turn()  # type: ignore[union-attr]
        assert last is not None
        assert isinstance(last.model_input, Message)
        assert last.model_input.content == "reply"

    def test_empty_context_raises(self):
        ctx = ChatContext()
        with pytest.raises(ValueError, match="no last turn"):
            _resolve_question(None, ctx)

    def test_from_cblock(self):
        ctx = ChatContext().add(CBlock("raw question"))
        text, rewound = _resolve_question(None, ctx)
        assert text == "raw question"
        assert rewound.is_root_node  # type: ignore[union-attr]

    def test_cblock_none_value_raises(self):
        ctx = ChatContext().add(CBlock(None))
        with pytest.raises(ValueError, match="no value"):
            _resolve_question(None, ctx)

    def test_from_component(self):
        ctx = ChatContext().add(Document("some document text"))
        text, rewound = _resolve_question(None, ctx)
        assert "some document text" in text
        assert rewound.is_root_node  # type: ignore[union-attr]

    def test_from_instruction_component(self):
        ctx = ChatContext().add(Instruction("Summarize the article"))
        text, rewound = _resolve_question(None, ctx)
        assert "Summarize the article" in text
        assert rewound.is_root_node  # type: ignore[union-attr]


class TestResolveResponse:
    def test_explicit_string(self):
        ctx = ChatContext()
        text, returned_ctx = _resolve_response("answer", ctx)
        assert text == "answer"
        assert returned_ctx is ctx

    def test_from_context(self):
        ctx = (
            ChatContext()
            .add(Message("user", "question"))
            .add(ModelOutputThunk(value="The answer is 4."))
        )
        text, rewound = _resolve_response(None, ctx)
        assert text == "The answer is 4."
        # Rewound context should still have the user question
        last = rewound.last_turn()  # type: ignore[union-attr]
        assert last is not None
        assert isinstance(last.model_input, Message)
        assert last.model_input.content == "question"

    def test_empty_context_raises(self):
        ctx = ChatContext()
        with pytest.raises(ValueError, match="no last turn"):
            _resolve_response(None, ctx)

    def test_none_value_raises(self):
        ctx = ChatContext().add(ModelOutputThunk(value=None))
        with pytest.raises(ValueError, match="no value"):
            _resolve_response(None, ctx)
