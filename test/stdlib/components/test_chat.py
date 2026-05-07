import logging

import pytest

from mellea.core import CBlock, ModelOutputThunk, TemplateRepresentation
from mellea.formatters.template_formatter import TemplateFormatter
from mellea.helpers import message_to_openai_message, messages_to_docs
from mellea.stdlib.components import Document, Message
from mellea.stdlib.components.chat import (
    ToolMessage,
    as_chat_history,
    as_generic_chat_history,
)
from mellea.stdlib.context import ChatContext


def test_message_with_docs():
    doc = Document("I'm text!", "Im a title!")
    msg = Message("user", "hello", documents=[doc])

    assert msg._docs is not None
    assert doc in msg._docs

    docs = messages_to_docs([msg])
    assert len(docs) == 1
    assert docs[0]["text"] == doc.text
    assert docs[0]["title"] == doc.title

    assert "[Document] Im a titl..." in str(msg)

    tr = msg.format_for_llm()
    assert tr.args["documents"]


# --- Message init ---


def test_message_basic_fields():
    msg = Message("user", "hello")
    assert msg.role == "user"
    assert msg.content == "hello"
    assert msg._images is None
    assert msg._docs is None


def test_message_content_block_created():
    msg = Message("assistant", "response")
    assert isinstance(msg._content_cblock, CBlock)
    assert msg._content_cblock.value == "response"


def test_message_repr():
    msg = Message("user", "hi there")
    r = repr(msg)
    assert 'role="user"' in r
    assert 'content="hi there"' in r


# --- Message images property ---


def test_message_images_none():
    msg = Message("user", "text")
    assert msg.images is None


# --- Message parts() ---


def test_message_parts_no_docs_no_images():
    msg = Message("user", "text")
    parts = msg.parts()
    assert len(parts) == 1
    assert parts[0] is msg._content_cblock


def test_message_parts_with_docs():
    doc = Document("text", "title")
    msg = Message("user", "hi", documents=[doc])
    parts = msg.parts()
    assert doc in parts


# --- Message format_for_llm ---


def test_message_format_for_llm_structure():
    msg = Message("user", "hello")
    tr = msg.format_for_llm()
    assert isinstance(tr, TemplateRepresentation)
    assert tr.args["content"] is msg._content_cblock
    assert tr.args["documents"] is None


def test_message_documents_string_coercion():
    msg = Message("user", "hello", documents=["doc one", "doc two"])
    assert msg._docs is not None
    assert len(msg._docs) == 2
    assert all(isinstance(d, Document) for d in msg._docs)
    assert msg._docs[0].text == "doc one"
    assert msg._docs[1].text == "doc two"


def test_message_documents_mixed_coercion():
    doc = Document("existing", doc_id="x")
    msg = Message("user", "hello", documents=["new text", doc])
    assert msg._docs is not None
    assert len(msg._docs) == 2
    assert msg._docs[0].text == "new text"
    assert msg._docs[1] is doc


# --- Message._parse — no tool calls ---


def test_parse_plain_value_no_meta():
    msg = Message("user", "original")
    mot = ModelOutputThunk(value="model response")
    result = msg._parse(mot)
    assert isinstance(result, Message)
    assert result.role == "assistant"
    assert result.content == "model response"


def test_parse_ollama_chat_response():
    msg = Message("user", "q")
    mot = ModelOutputThunk(value="v")
    fake_response = type(
        "Resp",
        (),
        {
            "message": type(
                "Msg", (), {"role": "assistant", "content": "ollama answer"}
            )()
        },
    )()
    mot._meta["chat_response"] = fake_response
    result = msg._parse(mot)
    assert result.role == "assistant"
    assert result.content == "ollama answer"


def test_parse_openai_chat_response():
    msg = Message("user", "q")
    mot = ModelOutputThunk(value="v")
    mot._meta["oai_chat_response"] = {
        "choices": [{"message": {"role": "assistant", "content": "openai answer"}}]
    }
    result = msg._parse(mot)
    assert result.role == "assistant"
    assert result.content == "openai answer"


# --- Message._parse — with tool calls ---


def test_parse_tool_calls_ollama():
    msg = Message("user", "q")
    mot = ModelOutputThunk(value="v", tool_calls={"some_fn": None})
    fake_calls = [{"name": "some_fn"}]
    fake_response = type(
        "Resp",
        (),
        {"message": type("Msg", (), {"role": "assistant", "tool_calls": fake_calls})()},
    )()
    mot._meta["chat_response"] = fake_response
    result = msg._parse(mot)
    assert result.role == "assistant"
    assert "some_fn" in result.content


def test_parse_tool_calls_openai():
    msg = Message("user", "q")
    mot = ModelOutputThunk(value="v", tool_calls={"fn": None})
    mot._meta["oai_chat_response"] = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [{"function": {"name": "fn"}}],
                }
            }
        ]
    }
    result = msg._parse(mot)
    assert result.role == "assistant"


def test_parse_tool_calls_fallback_uses_value():
    """No chat_response or oai_chat_response — falls back to computed.value."""
    msg = Message("user", "q")
    mot = ModelOutputThunk(value="<tool_call>fn()</tool_call>", tool_calls={"fn": None})
    result = msg._parse(mot)
    assert result.role == "assistant"
    assert result.content == "<tool_call>fn()</tool_call>"


# --- ToolMessage ---


def test_tool_message_fields():
    from mellea.core import ModelToolCall

    fake_tool = type("T", (), {"as_json_tool": {}})()
    mtc = ModelToolCall("my_tool", fake_tool, {"x": 1})
    tm = ToolMessage(
        role="tool",
        content='{"result": 42}',
        tool_output=42,
        name="my_tool",
        args={"x": 1},
        tool=mtc,
    )
    assert tm.role == "tool"
    assert tm.name == "my_tool"
    assert tm.arguments == {"x": 1}


def test_tool_message_repr():
    from mellea.core import ModelToolCall

    fake_tool = type("T", (), {"as_json_tool": {}})()
    mtc = ModelToolCall("fn", fake_tool, {})
    tm = ToolMessage("tool", "out", "out", "fn", {}, mtc)
    r = repr(tm)
    assert 'name="fn"' in r


# --- as_chat_history ---


def test_as_chat_history_messages_only():
    ctx = ChatContext()
    ctx = ctx.add(Message("user", "hello"))
    ctx = ctx.add(Message("assistant", "hi"))
    history = as_chat_history(ctx)
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[1].role == "assistant"


def test_as_chat_history_empty():
    ctx = ChatContext()
    history = as_chat_history(ctx)
    assert history == []


def test_as_chat_history_with_parsed_mot():
    ctx = ChatContext()
    ctx = ctx.add(Message("user", "hello"))
    mot = ModelOutputThunk(value="reply")
    mot.parsed_repr = Message("assistant", "reply")
    ctx = ctx.add(mot)
    history = as_chat_history(ctx)
    assert len(history) == 2
    assert history[1].content == "reply"


# --- as_generic_chat_history ---


def test_as_generic_chat_history_messages_only():
    ctx = ChatContext()
    ctx = ctx.add(Message("user", "hello"))
    ctx = ctx.add(Message("assistant", "hi"))
    history = as_generic_chat_history(ctx)
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "hello"
    assert history[1].role == "assistant"
    assert history[1].content == "hi"


def test_as_generic_chat_history_empty():
    ctx = ChatContext()
    history = as_generic_chat_history(ctx)
    assert history == []


def test_as_generic_chat_history_with_parsed_mot():
    ctx = ChatContext()
    ctx = ctx.add(Message("user", "hello"))
    mot = ModelOutputThunk(value="reply")
    mot.parsed_repr = Message("assistant", "reply")
    ctx = ctx.add(mot)
    history = as_generic_chat_history(ctx)
    assert len(history) == 2
    assert history[1].role == "assistant"
    assert history[1].content == "reply"


def test_as_generic_chat_history_with_unparsed_mot():
    """Unresolved ModelOutputThunk gets converted to string."""
    ctx = ChatContext()
    ctx = ctx.add(Message("user", "hello"))
    mot = ModelOutputThunk(value="raw output")
    ctx = ctx.add(mot)
    history = as_generic_chat_history(ctx)
    assert len(history) == 2
    assert history[1].role == "assistant"
    assert "raw output" in history[1].content


def test_as_generic_chat_history_with_string_parsed_repr():
    """ModelOutputThunk with string parsed_repr (e.g., from CBlock action)."""
    ctx = ChatContext()
    ctx = ctx.add(Message("user", "hello"))
    # Simulate a ModelOutputThunk with a string parsed_repr,
    # as would result from a CBlock action completing
    mot = ModelOutputThunk(value="reply text", parsed_repr="reply text")
    ctx = ctx.add(mot)
    history = as_generic_chat_history(ctx)
    assert len(history) == 2
    assert history[1].role == "assistant"
    assert history[1].content == "reply text"


def test_as_generic_chat_history_with_non_message_parsed_repr():
    """ModelOutputThunk with non-Message, non-string parsed_repr uses formatter."""

    def custom_formatter(obj: object) -> str:
        if isinstance(obj, dict):
            return f"dict:{obj}"
        return str(obj)

    ctx = ChatContext()
    ctx = ctx.add(Message("user", "hello"))
    # parsed_repr is a dict (could be structured data from a model)
    mot = ModelOutputThunk(value="raw", parsed_repr={"key": "value"})
    ctx = ctx.add(mot)
    history = as_generic_chat_history(ctx, formatter=custom_formatter)
    assert len(history) == 2
    assert history[1].role == "assistant"
    assert "dict:" in history[1].content


def test_as_generic_chat_history_with_cblock():
    """CBlocks are converted to Messages with 'user' role."""
    ctx = ChatContext()
    ctx = ctx.add(CBlock("inline content"))
    ctx = ctx.add(Message("assistant", "response"))
    history = as_generic_chat_history(ctx)
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "inline content"


def test_as_generic_chat_history_custom_formatter():
    """Custom formatter handles unknown types."""

    def custom_formatter(obj: object) -> str:
        return f"<custom:{type(obj).__name__}>"

    class CustomComponent:
        def __str__(self):
            return "original"

    ctx = ChatContext()
    ctx = ctx.add(Message("user", "hello"))
    ctx = ctx.add(CustomComponent())
    history = as_generic_chat_history(ctx, formatter=custom_formatter)
    assert len(history) == 2
    assert "<custom:CustomComponent>" in history[1].content


def test_as_generic_chat_history_default_formatter_logs_warning(caplog):
    """Default formatter logs a warning for unknown types."""

    class UnknownComponent:
        pass

    ctx = ChatContext()
    ctx = ctx.add(Message("user", "hello"))
    ctx = ctx.add(UnknownComponent())

    with caplog.at_level(logging.WARNING):
        history = as_generic_chat_history(ctx)

    assert len(history) == 2
    assert any("Unknown component type" in record.message for record in caplog.records)


# --- Formatter rendering of Message documents ---


class TestMessageDocumentRendering:
    """Tests that documents on Messages are rendered through the formatter."""

    @pytest.fixture
    def formatter(self):
        return TemplateFormatter(model_id="test-model")

    def test_print_message_without_docs(self, formatter):
        msg = Message("user", "hello")
        result = formatter.print(msg)
        assert result == "hello"

    def test_print_message_with_docs(self, formatter):
        doc = Document("The answer is 42.", title="Guide", doc_id="1")
        msg = Message("user", "What is the answer?", documents=[doc])
        result = formatter.print(msg)
        assert "What is the answer?" in result
        assert "[Document 1]" in result
        assert "Guide:" in result
        assert "The answer is 42." in result

    def test_print_message_with_multiple_docs(self, formatter):
        docs = [
            Document("First doc content.", doc_id="0"),
            Document("Second doc content.", doc_id="1"),
        ]
        msg = Message("user", "Summarize these.", documents=docs)
        result = formatter.print(msg)
        assert "Summarize these." in result
        assert "First doc content." in result
        assert "Second doc content." in result

    def test_print_message_with_string_docs(self, formatter):
        msg = Message("user", "question", documents=["raw doc text"])
        result = formatter.print(msg)
        assert "question" in result
        assert "raw doc text" in result

    def test_to_chat_messages_preserves_docs_for_print(self, formatter):
        """Messages with docs survive to_chat_messages() and can be printed."""
        doc = Document("grounding info", title="Ref")
        msg = Message("user", "query", documents=[doc])

        messages = formatter.to_chat_messages([msg])
        assert len(messages) == 1

        returned_msg = messages[0]
        # Role is still accessible as a separate field
        assert returned_msg.role == "user"
        # Documents are preserved
        assert returned_msg._docs is not None
        assert len(returned_msg._docs) == 1
        # Formatter print renders docs into content
        rendered = formatter.print(returned_msg)
        assert "query" in rendered
        assert "grounding info" in rendered
        assert "Ref:" in rendered

    def test_message_to_openai_message_with_formatter(self, formatter):
        doc = Document("supporting text", doc_id="d1")
        msg = Message("user", "main content", documents=[doc])
        result = message_to_openai_message(msg, formatter)
        assert result["role"] == "user"
        assert "main content" in result["content"]
        assert "supporting text" in result["content"]

    def test_message_to_openai_message_without_formatter_drops_docs(self):
        doc = Document("supporting text", doc_id="d1")
        msg = Message("user", "main content", documents=[doc])
        result = message_to_openai_message(msg)
        assert result["role"] == "user"
        assert result["content"] == "main content"
        assert "supporting text" not in result["content"]

    def test_print_message_with_docs_renders_document_format(self, formatter):
        """Verify exact rendered format of documents within a Message."""
        doc = Document("The capital of France is Paris.", title="Geography", doc_id="7")
        msg = Message("user", "What is the capital of France?", documents=[doc])
        result = formatter.print(msg)
        assert "What is the capital of France?" in result
        assert "[Document 7]" in result
        assert "Geography: The capital of France is Paris." in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
