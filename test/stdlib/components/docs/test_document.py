from mellea.stdlib.components.docs.document import Document


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
    assert doc.format_for_llm() == "just text"
