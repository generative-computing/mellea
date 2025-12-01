import pytest
from mellea.backends.openai import OpenAIBackend
from mellea.stdlib.base import Document
from mellea.stdlib.chat import Message


def test_message_with_docs():
    doc = Document("I'm text!", "Im a title!")
    msg = Message("user", "hello", documents=[doc])

    assert msg._docs is not None
    assert doc in msg._docs

    docs = OpenAIBackend.messages_to_docs([msg])
    assert len(docs) == 1
    assert docs[0]["text"] == doc.text
    assert docs[0]["title"] == doc.title

    assert "Im a titl..." in str(msg)

    tr = msg.format_for_llm()
    assert tr.args["documents"]


if __name__ == "__main__":
    pytest.main([__file__])
