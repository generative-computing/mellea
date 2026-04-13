"""Unit tests for mellea.helpers.openai_compatible_helpers."""

import base64
import json

import pytest

from mellea.backends.tools import MelleaTool
from mellea.core.base import ImageBlock
from mellea.helpers.openai_compatible_helpers import (
    chat_completion_delta_merge,
    extract_model_tool_requests,
    message_to_openai_message,
    messages_to_docs,
)
from mellea.stdlib.components import Document, Message

# Minimal valid 1x1 white PNG
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MINIMAL_PNG = (
    _PNG_SIGNATURE
    + b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
    + b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx"
    + b"\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
    + b"\x00\x00\x00\x00IEND\xaeB`\x82"
)
_B64_PNG = base64.b64encode(_MINIMAL_PNG).decode()

# --- helpers ---


def _make_tool(name: str = "get_weather") -> MelleaTool:
    """Create a simple MelleaTool for testing."""

    def get_weather(location: str) -> str:
        return f"sunny in {location}"

    return MelleaTool.from_callable(get_weather, name)


def _response_with_tool_calls(calls: list[dict]) -> dict:
    """Build a minimal OpenAI-like response dict with tool_calls."""
    return {"message": {"tool_calls": calls}}


def _tool_call(name: str, arguments: str | None) -> dict:
    return {"function": {"name": name, "arguments": arguments}}


# --- extract_model_tool_requests ---


class TestExtractModelToolRequests:
    def test_single_tool_call(self):
        tool = _make_tool("get_weather")
        tools = {"get_weather": tool}
        response = _response_with_tool_calls(
            [_tool_call("get_weather", json.dumps({"location": "Dallas"}))]
        )
        result = extract_model_tool_requests(tools, response)
        assert result is not None
        assert "get_weather" in result
        assert result["get_weather"].name == "get_weather"
        assert result["get_weather"].args["location"] == "Dallas"

    def test_no_tool_calls_returns_none(self):
        tools = {"get_weather": _make_tool()}
        response = {"message": {}}
        assert extract_model_tool_requests(tools, response) is None

    def test_empty_tool_calls_returns_none(self):
        tools = {"get_weather": _make_tool()}
        response = {"message": {"tool_calls": []}}
        assert extract_model_tool_requests(tools, response) is None

    def test_tool_calls_none_returns_none(self):
        tools = {"get_weather": _make_tool()}
        response = {"message": {"tool_calls": None}}
        assert extract_model_tool_requests(tools, response) is None

    def test_unknown_tool_skipped(self):
        tools = {"get_weather": _make_tool()}
        response = _response_with_tool_calls(
            [_tool_call("nonexistent", json.dumps({"x": 1}))]
        )
        assert extract_model_tool_requests(tools, response) is None

    def test_multiple_tools(self):
        def search(query: str) -> str:
            return query

        tool_a = _make_tool("get_weather")
        tool_b = MelleaTool.from_callable(search, "search")
        tools = {"get_weather": tool_a, "search": tool_b}
        response = _response_with_tool_calls(
            [
                _tool_call("get_weather", json.dumps({"location": "NYC"})),
                _tool_call("search", json.dumps({"query": "news"})),
            ]
        )
        result = extract_model_tool_requests(tools, response)
        assert result is not None
        assert len(result) == 2
        assert "get_weather" in result
        assert "search" in result

    def test_null_arguments(self):
        """Tool call with None arguments produces empty args dict."""

        def no_args() -> str:
            return "ok"

        tool = MelleaTool.from_callable(no_args, "ping")
        tools = {"ping": tool}
        response = _response_with_tool_calls([_tool_call("ping", None)])
        result = extract_model_tool_requests(tools, response)
        assert result is not None
        assert result["ping"].args == {}

    def test_mixed_known_and_unknown(self):
        """Known tool is extracted; unknown tool is silently skipped."""
        tool = _make_tool("get_weather")
        tools = {"get_weather": tool}
        response = _response_with_tool_calls(
            [
                _tool_call("get_weather", json.dumps({"location": "LA"})),
                _tool_call("hallucinated_tool", json.dumps({"a": 1})),
            ]
        )
        result = extract_model_tool_requests(tools, response)
        assert result is not None
        assert len(result) == 1
        assert "get_weather" in result

    def test_malformed_json_arguments_raises(self):
        """Malformed JSON from the model propagates as JSONDecodeError."""
        tool = _make_tool("get_weather")
        tools = {"get_weather": tool}
        response = _response_with_tool_calls([_tool_call("get_weather", '{"broken')])
        with pytest.raises(json.JSONDecodeError):
            extract_model_tool_requests(tools, response)


# --- chat_completion_delta_merge ---


def _delta_chunk(
    *,
    content: str | None = None,
    role: str | None = None,
    tool_calls: list[dict] | None = None,
    finish_reason: str | None = None,
    reasoning_content: str | None = None,
) -> dict:
    delta: dict = {}
    if content is not None:
        delta["content"] = content
    if role is not None:
        delta["role"] = role
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content
    return {"delta": delta, "finish_reason": finish_reason}


class TestChatCompletionDeltaMerge:
    def test_empty_chunks(self):
        result = chat_completion_delta_merge([])
        assert result["message"]["content"] == ""
        assert result["message"]["role"] is None
        assert result["message"]["tool_calls"] == []
        assert result["finish_reason"] is None

    def test_text_content_merging(self):
        chunks = [
            _delta_chunk(role="assistant", content="Hello"),
            _delta_chunk(content=" world"),
            _delta_chunk(content="!", finish_reason="stop"),
        ]
        result = chat_completion_delta_merge(chunks)
        assert result["message"]["content"] == "Hello world!"
        assert result["message"]["role"] == "assistant"
        assert result["finish_reason"] == "stop"

    def test_reasoning_content_merging(self):
        chunks = [
            _delta_chunk(role="assistant", reasoning_content="Let me think"),
            _delta_chunk(reasoning_content="..."),
            _delta_chunk(content="answer"),
        ]
        result = chat_completion_delta_merge(chunks)
        assert result["message"]["reasoning_content"] == "Let me think..."
        assert result["message"]["content"] == "answer"

    def test_tool_call_assembly(self):
        """Tool call spread across multiple chunks is merged by index."""
        chunks = [
            _delta_chunk(
                role="assistant",
                tool_calls=[
                    {"index": 0, "function": {"name": "get_weather", "arguments": None}}
                ],
            ),
            _delta_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "function": {"name": None, "arguments": '{"location": "'},
                    }
                ]
            ),
            _delta_chunk(
                tool_calls=[
                    {"index": 0, "function": {"name": None, "arguments": 'Dallas"}'}}
                ],
                finish_reason="tool_calls",
            ),
        ]
        result = chat_completion_delta_merge(chunks)
        tc = result["message"]["tool_calls"]
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "get_weather"
        assert json.loads(tc[0]["function"]["arguments"]) == {"location": "Dallas"}

    def test_multiple_tool_calls_by_index(self):
        chunks = [
            _delta_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "function": {"name": "tool_a", "arguments": '{"x": 1}'},
                    }
                ]
            ),
            _delta_chunk(
                tool_calls=[
                    {
                        "index": 1,
                        "function": {"name": "tool_b", "arguments": '{"y": 2}'},
                    }
                ]
            ),
        ]
        result = chat_completion_delta_merge(chunks)
        tc = result["message"]["tool_calls"]
        assert len(tc) == 2
        assert tc[0]["function"]["name"] == "tool_a"
        assert tc[1]["function"]["name"] == "tool_b"

    def test_force_all_tool_calls_separate(self):
        """When force_all_tool_calls_separate=True, same-index calls become separate entries."""
        chunks = [
            _delta_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "function": {"name": "tool_a", "arguments": '{"a": 1}'},
                    }
                ]
            ),
            _delta_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "function": {"name": "tool_b", "arguments": '{"b": 2}'},
                    }
                ]
            ),
        ]
        result = chat_completion_delta_merge(chunks, force_all_tool_calls_separate=True)
        tc = result["message"]["tool_calls"]
        assert len(tc) == 2
        assert tc[0]["function"]["name"] == "tool_a"
        assert tc[1]["function"]["name"] == "tool_b"

    def test_stop_reason_captured(self):
        chunks = [
            _delta_chunk(content="hi"),
            {"delta": {}, "finish_reason": None, "stop_reason": "end_turn"},
        ]
        result = chat_completion_delta_merge(chunks)
        assert result["stop_reason"] == "end_turn"


# --- message_to_openai_message ---


class TestMessageToOpenaiMessage:
    def test_text_only(self):
        msg = Message(role="user", content="hello")
        result = message_to_openai_message(msg)
        assert result == {"role": "user", "content": "hello"}

    def test_with_images(self):
        img = ImageBlock(_B64_PNG)
        msg = Message(role="user", content="describe this", images=[img])
        result = message_to_openai_message(msg)
        assert result["role"] == "user"
        assert isinstance(result["content"], list)
        assert result["content"][0] == {"type": "text", "text": "describe this"}
        assert result["content"][1]["type"] == "image_url"
        assert result["content"][1]["image_url"]["url"].startswith(
            "data:image/png;base64,"
        )

    def test_with_multiple_images(self):
        imgs = [ImageBlock(_B64_PNG), ImageBlock(_B64_PNG)]
        msg = Message(role="user", content="compare", images=imgs)
        result = message_to_openai_message(msg)
        assert len(result["content"]) == 3  # 1 text + 2 images

    def test_with_empty_images_list(self):
        """Empty images list triggers multimodal content format (unlike None)."""
        msg = Message(role="user", content="hello", images=[])
        result = message_to_openai_message(msg)
        # images=[] is not None, so the list-content branch is taken
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 1
        assert result["content"][0] == {"type": "text", "text": "hello"}


# --- messages_to_docs ---


class TestMessagesToDocs:
    def test_no_docs(self):
        msgs = [Message(role="user", content="hello")]
        assert messages_to_docs(msgs) == []

    def test_single_doc(self):
        doc = Document(text="passage", title="Title", doc_id="d1")
        msg = Message(role="user", content="q", documents=[doc])
        result = messages_to_docs([msg])
        assert len(result) == 1
        assert result[0]["text"] == "passage"
        assert result[0]["title"] == "Title"
        assert result[0]["doc_id"] == "d1"

    def test_doc_without_optional_fields(self):
        doc = Document(text="passage only")
        msg = Message(role="user", content="q", documents=[doc])
        result = messages_to_docs([msg])
        assert result[0] == {"text": "passage only"}
        assert "title" not in result[0]
        assert "doc_id" not in result[0]

    def test_docs_across_messages(self):
        d1 = Document(text="a", title="A")
        d2 = Document(text="b", doc_id="id2")
        msgs = [
            Message(role="user", content="q1", documents=[d1]),
            Message(role="assistant", content="a1"),
            Message(role="user", content="q2", documents=[d2]),
        ]
        result = messages_to_docs(msgs)
        assert len(result) == 2
        assert result[0]["text"] == "a"
        assert result[1]["text"] == "b"


if __name__ == "__main__":
    pytest.main([__file__])
