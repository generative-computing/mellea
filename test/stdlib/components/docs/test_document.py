"""Tests for Document and _coerce_documents/_coerce_document."""

import warnings

from mellea.stdlib.components.docs.document import (
    Document,
    _coerce_document,
    _coerce_documents,
)


def test_document_parts_returns_empty_list():
    doc = Document("some text", title="Test", doc_id="1")
    assert doc.parts() == [], "Document.parts() should return an empty list"


def test_document_format_for_llm():
    doc = Document("hello world", title="Greeting", doc_id="abc")
    result = doc.format_for_llm()
    assert "abc" in result
    assert "Greeting" in result
    assert "hello world" in result


def test_document_format_for_llm_no_title():
    doc = Document("just text")
    assert doc.format_for_llm() == "[Document]\njust text"


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
