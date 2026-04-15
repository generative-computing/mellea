# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Granite 3.2 input processor."""

import pytest

from mellea.formatters.granite.base.types import (
    AssistantMessage,
    Document,
    SystemMessage,
    ToolDefinition,
    UserMessage,
    VLLMExtraBody,
)
from mellea.formatters.granite.granite3.constants import (
    NO_TOOLS_AND_DOCS_SYSTEM_MESSAGE_PART,
    NO_TOOLS_NO_DOCS_NO_THINKING_SYSTEM_MESSAGE_PART,
)
from mellea.formatters.granite.granite3.granite32.constants import (
    DOCS_AND_CITATIONS_SYSTEM_MESSAGE_PART,
    DOCS_AND_HALLUCINATIONS_SYSTEM_MESSAGE_PART,
    NO_TOOLS_AND_NO_DOCS_AND_THINKING_SYSTEM_MESSAGE_PART,
    TOOLS_AND_DOCS_SYSTEM_MESSAGE_PART,
    TOOLS_AND_NO_DOCS_SYSTEM_MESSAGE_PART,
)
from mellea.formatters.granite.granite3.granite32.input import Granite32InputProcessor
from mellea.formatters.granite.granite3.granite32.types import Granite32ChatCompletion
from mellea.formatters.granite.granite3.types import Granite3Controls, Granite3Kwargs


def _make_completion(**kwargs) -> Granite32ChatCompletion:
    """Helper to build a Granite32ChatCompletion with sensible defaults."""
    if "messages" not in kwargs:
        kwargs["messages"] = [UserMessage(content="Hello")]
    return Granite32ChatCompletion(**kwargs)


# ---------------------------------------------------------------------------
# _build_default_system_message
# ---------------------------------------------------------------------------


class TestBuildDefaultSystemMessage:
    def setup_method(self):
        self.proc = Granite32InputProcessor()

    def test_no_tools_no_docs_no_thinking(self):
        cc = _make_completion()
        msg = self.proc._build_default_system_message(cc)
        assert "<|start_of_role|>system<|end_of_role|>" in msg
        assert NO_TOOLS_NO_DOCS_NO_THINKING_SYSTEM_MESSAGE_PART in msg
        assert msg.endswith("<|end_of_text|>\n")

    def test_tools_only(self):
        cc = _make_completion(tools=[ToolDefinition(name="my_tool")])
        msg = self.proc._build_default_system_message(cc)
        assert TOOLS_AND_NO_DOCS_SYSTEM_MESSAGE_PART in msg

    def test_docs_only(self):
        cc = _make_completion(
            extra_body=VLLMExtraBody(documents=[Document(text="Some document.")])
        )
        msg = self.proc._build_default_system_message(cc)
        assert NO_TOOLS_AND_DOCS_SYSTEM_MESSAGE_PART in msg

    def test_tools_and_docs(self):
        cc = _make_completion(
            tools=[ToolDefinition(name="my_tool")],
            extra_body=VLLMExtraBody(documents=[Document(text="Some document.")]),
        )
        msg = self.proc._build_default_system_message(cc)
        assert TOOLS_AND_DOCS_SYSTEM_MESSAGE_PART in msg

    def test_thinking_only(self):
        cc = _make_completion(
            extra_body=VLLMExtraBody(chat_template_kwargs=Granite3Kwargs(thinking=True))
        )
        msg = self.proc._build_default_system_message(cc)
        assert NO_TOOLS_AND_NO_DOCS_AND_THINKING_SYSTEM_MESSAGE_PART in msg

    def test_docs_and_citations(self):
        cc = _make_completion(
            extra_body=VLLMExtraBody(
                documents=[Document(text="doc")],
                chat_template_kwargs=Granite3Kwargs(
                    controls=Granite3Controls(citations=True)
                ),
            )
        )
        msg = self.proc._build_default_system_message(cc)
        assert DOCS_AND_CITATIONS_SYSTEM_MESSAGE_PART in msg

    def test_docs_citations_and_hallucinations(self):
        cc = _make_completion(
            extra_body=VLLMExtraBody(
                documents=[Document(text="doc")],
                chat_template_kwargs=Granite3Kwargs(
                    controls=Granite3Controls(citations=True, hallucinations=True)
                ),
            )
        )
        msg = self.proc._build_default_system_message(cc)
        assert DOCS_AND_CITATIONS_SYSTEM_MESSAGE_PART in msg
        assert DOCS_AND_HALLUCINATIONS_SYSTEM_MESSAGE_PART in msg

    def test_thinking_with_docs_raises(self):
        cc = _make_completion(
            extra_body=VLLMExtraBody(
                documents=[Document(text="doc")],
                chat_template_kwargs=Granite3Kwargs(thinking=True),
            )
        )
        with pytest.raises(ValueError, match="thinking"):
            self.proc._build_default_system_message(cc)

    def test_thinking_with_tools_raises(self):
        cc = _make_completion(
            tools=[ToolDefinition(name="tool")],
            extra_body=VLLMExtraBody(
                chat_template_kwargs=Granite3Kwargs(thinking=True)
            ),
        )
        with pytest.raises(ValueError, match="thinking"):
            self.proc._build_default_system_message(cc)

    def test_hallucinations_without_docs_raises(self):
        cc = _make_completion(
            extra_body=VLLMExtraBody(
                chat_template_kwargs=Granite3Kwargs(
                    controls=Granite3Controls(hallucinations=True)
                )
            )
        )
        with pytest.raises(ValueError, match="hallucinations"):
            self.proc._build_default_system_message(cc)

    def test_citations_without_docs_no_error(self):
        """Citations without docs is silently skipped per the Jinja template."""
        cc = _make_completion(
            extra_body=VLLMExtraBody(
                chat_template_kwargs=Granite3Kwargs(
                    controls=Granite3Controls(citations=True)
                )
            )
        )
        msg = self.proc._build_default_system_message(cc)
        assert DOCS_AND_CITATIONS_SYSTEM_MESSAGE_PART not in msg


# ---------------------------------------------------------------------------
# _remove_special_tokens
# ---------------------------------------------------------------------------


class TestRemoveSpecialTokens32:
    def test_removes_role_markers(self):
        text = "<|start_of_role|>system<|end_of_role|>content<|end_of_text|>"
        result = Granite32InputProcessor._remove_special_tokens(text)
        assert result == ""

    def test_removes_tool_call_marker(self):
        text = '<|tool_call|>{"name":"func"}'
        result = Granite32InputProcessor._remove_special_tokens(text)
        assert result == ""

    def test_removes_stray_special_tokens(self):
        text = "Hello <|end_of_text|> world <fim_prefix> end"
        result = Granite32InputProcessor._remove_special_tokens(text)
        assert "<|end_of_text|>" not in result
        assert "<fim_prefix>" not in result
        assert "Hello" in result

    def test_clean_text_unchanged(self):
        text = "Just normal text."
        assert Granite32InputProcessor._remove_special_tokens(text) == text


# ---------------------------------------------------------------------------
# sanitize
# ---------------------------------------------------------------------------


class TestSanitize32:
    def test_sanitizes_messages(self):
        cc = _make_completion(
            messages=[UserMessage(content="Hello <|end_of_text|> world")]
        )
        sanitized = Granite32InputProcessor.sanitize(cc, parts="messages")
        assert "<|end_of_text|>" not in sanitized.messages[0].content

    def test_sanitizes_all_by_default(self):
        cc = _make_completion(
            messages=[UserMessage(content="msg <|end_of_text|>")],
            tools=[ToolDefinition(name="tool <|end_of_text|>")],
        )
        sanitized = Granite32InputProcessor.sanitize(cc)
        assert "<|end_of_text|>" not in sanitized.messages[0].content
        assert "<|end_of_text|>" not in sanitized.tools[0].name


# ---------------------------------------------------------------------------
# transform
# ---------------------------------------------------------------------------


class TestTransform32:
    def setup_method(self):
        self.proc = Granite32InputProcessor()

    def test_basic_user_message(self):
        cc = _make_completion()
        result = self.proc.transform(cc)
        assert "<|start_of_role|>system<|end_of_role|>" in result
        assert "<|start_of_role|>user<|end_of_role|>Hello<|end_of_text|>" in result
        assert result.endswith("<|start_of_role|>assistant<|end_of_role|>")

    def test_custom_system_message(self):
        cc = _make_completion(
            messages=[
                SystemMessage(content="You are a pirate."),
                UserMessage(content="Ahoy!"),
            ]
        )
        result = self.proc.transform(cc)
        assert "You are a pirate." in result
        assert "Ahoy!" in result

    def test_custom_system_with_thinking_raises(self):
        cc = _make_completion(
            messages=[
                SystemMessage(content="Custom system."),
                UserMessage(content="Q"),
            ],
            extra_body=VLLMExtraBody(
                chat_template_kwargs=Granite3Kwargs(thinking=True)
            ),
        )
        with pytest.raises(ValueError, match="thinking"):
            self.proc.transform(cc)

    def test_custom_system_with_docs_raises(self):
        cc = _make_completion(
            messages=[
                SystemMessage(content="Custom system."),
                UserMessage(content="Q"),
            ],
            extra_body=VLLMExtraBody(documents=[Document(text="doc")]),
        )
        with pytest.raises(ValueError, match="documents"):
            self.proc.transform(cc)

    def test_custom_system_with_citations_raises(self):
        cc = _make_completion(
            messages=[
                SystemMessage(content="Custom system."),
                UserMessage(content="Q"),
            ],
            extra_body=VLLMExtraBody(
                chat_template_kwargs=Granite3Kwargs(
                    controls=Granite3Controls(citations=True)
                )
            ),
        )
        with pytest.raises(ValueError, match="citations"):
            self.proc.transform(cc)

    def test_custom_system_with_hallucinations_raises(self):
        cc = _make_completion(
            messages=[
                SystemMessage(content="Custom system."),
                UserMessage(content="Q"),
            ],
            extra_body=VLLMExtraBody(
                chat_template_kwargs=Granite3Kwargs(
                    controls=Granite3Controls(hallucinations=True)
                )
            ),
        )
        with pytest.raises(ValueError, match="hallucinations"):
            self.proc.transform(cc)

    def test_tools_section(self):
        tool = ToolDefinition(name="get_weather", description="Get weather info")
        cc = _make_completion(tools=[tool])
        result = self.proc.transform(cc)
        assert "<|start_of_role|>tools<|end_of_role|>" in result
        assert "get_weather" in result

    def test_documents_section(self):
        cc = _make_completion(
            extra_body=VLLMExtraBody(
                documents=[Document(text="First doc."), Document(text="Second doc.")]
            )
        )
        result = self.proc.transform(cc)
        assert "<|start_of_role|>documents<|end_of_role|>" in result
        assert "Document 0\nFirst doc." in result
        assert "Document 1\nSecond doc." in result

    def test_controls_in_assistant_role(self):
        cc = _make_completion(
            extra_body=VLLMExtraBody(
                documents=[Document(text="doc")],
                chat_template_kwargs=Granite3Kwargs(
                    controls=Granite3Controls(citations=True)
                ),
            )
        )
        result = self.proc.transform(cc)
        assert '<|start_of_role|>assistant {"citations": true}<|end_of_role|>' in result

    def test_no_generation_prompt(self):
        cc = _make_completion()
        result = self.proc.transform(cc, add_generation_prompt=False)
        assert not result.endswith("<|end_of_role|>")

    def test_multi_turn_conversation(self):
        cc = _make_completion(
            messages=[
                UserMessage(content="Hi"),
                AssistantMessage(content="Hello!"),
                UserMessage(content="How are you?"),
            ]
        )
        result = self.proc.transform(cc)
        assert result.count("<|start_of_role|>user<|end_of_role|>") == 2
        assert "<|start_of_role|>assistant<|end_of_role|>Hello!" in result
