# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Granite 3.2 output processor."""

import json

from mellea.formatters.granite.base.types import (
    Document,
    ToolDefinition,
    UserMessage,
    VLLMExtraBody,
)
from mellea.formatters.granite.granite3.granite32.constants import (
    CITATION_START,
    HALLUCINATION_START,
)
from mellea.formatters.granite.granite3.granite32.output import (
    Granite32OutputProcessor,
    _get_docs_from_citations,
    _parse_citations_text,
    _remove_citations_from_response_text,
    _split_model_output_into_parts,
    _update_docs_text_with_input_docs,
    _validate_response,
)
from mellea.formatters.granite.granite3.granite32.types import Granite32ChatCompletion
from mellea.formatters.granite.granite3.types import (
    Granite3AssistantMessage,
    Granite3Controls,
    Granite3Kwargs,
)
from test.predicates import require_nltk_data

# ---------------------------------------------------------------------------
# _parse_citations_text
# ---------------------------------------------------------------------------


class TestParseCitationsText:
    def test_single_citation(self):
        text = '<co>1</co> Document 2: "RAG is retrieval-augmented generation."'
        result = _parse_citations_text(text)
        assert len(result) == 1
        assert result[0]["citation_id"] == "1"
        assert result[0]["doc_id"] == "2"
        assert result[0]["context_text"] == "RAG is retrieval-augmented generation."

    def test_multiple_citations(self):
        text = (
            '<co>1</co> Document 0: "First cited text."\n'
            '<co>2</co> Document 1: "Second cited text."'
        )
        result = _parse_citations_text(text)
        assert len(result) == 2
        assert result[0]["citation_id"] == "1"
        assert result[0]["doc_id"] == "0"
        assert result[1]["citation_id"] == "2"
        assert result[1]["doc_id"] == "1"

    def test_empty_returns_empty(self):
        result = _parse_citations_text("")
        assert result == []

    def test_missing_closing_quote(self):
        text = '<co>1</co> Document 0: "text without closing quote'
        result = _parse_citations_text(text)
        assert len(result) == 1
        assert result[0]["context_text"] == "text without closing quote"


# ---------------------------------------------------------------------------
# _remove_citations_from_response_text
# ---------------------------------------------------------------------------


class TestRemoveCitationsFromResponseText:
    def test_removes_inline_tags(self):
        text = "Hello <co>1</co> world <co>2</co> end."
        result = _remove_citations_from_response_text(text)
        assert "<co>" not in result
        assert "</co>" not in result
        assert "Hello" in result
        assert "world" in result

    def test_no_tags_unchanged(self):
        text = "Plain text without citations."
        assert _remove_citations_from_response_text(text) == text


# ---------------------------------------------------------------------------
# _split_model_output_into_parts
# ---------------------------------------------------------------------------


class TestSplitModelOutputIntoParts:
    def test_response_only(self):
        output = "Just a plain response."
        response, citations, hallucinations = _split_model_output_into_parts(output)
        assert response == "Just a plain response."
        assert citations == ""
        assert hallucinations == ""

    def test_response_and_citations(self):
        output = f'My response.\n{CITATION_START}\n<co>1</co> Document 0: "cited"'
        response, citations, hallucinations = _split_model_output_into_parts(output)
        assert response == "My response."
        assert "<co>1</co>" in citations
        assert hallucinations == ""

    def test_response_and_hallucinations(self):
        output = f"My response.\n{HALLUCINATION_START}\n1. Risk high: statement"
        response, citations, hallucinations = _split_model_output_into_parts(output)
        assert response == "My response."
        assert citations == ""
        assert "Risk high" in hallucinations

    def test_response_citations_and_hallucinations(self):
        output = (
            f"My response.\n{CITATION_START}\ncitation text\n"
            f"{HALLUCINATION_START}\nhallucination text"
        )
        response, citations, hallucinations = _split_model_output_into_parts(output)
        assert response == "My response."
        assert citations.strip() == "citation text"
        assert hallucinations.strip() == "hallucination text"

    def test_hallucinations_before_citations(self):
        output = f"Response.\n{HALLUCINATION_START}\nhalluc\n{CITATION_START}\ncites"
        response, citations, hallucinations = _split_model_output_into_parts(output)
        assert response.strip() == "Response."
        assert "halluc" in hallucinations
        assert "cites" in citations


# ---------------------------------------------------------------------------
# _validate_response
# ---------------------------------------------------------------------------


class TestValidateResponse:
    def test_balanced_tags_no_warnings(self, caplog):
        text = "Hello <co>1</co> world <co>2</co>."
        citations = [{"id": "1"}, {"id": "2"}]
        _validate_response(text, citations)
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 0

    def test_nested_tags_warns(self, caplog):
        text = "Hello <co>1 <co>2</co> </co> end."
        _validate_response(text, [{"id": "1"}])
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any("nested" in r.message.lower() for r in warnings)

    def test_mismatched_opening_closing_warns(self, caplog):
        text = "Hello <co>1</co> <co>2 end."
        _validate_response(text, [{"id": "1"}, {"id": "2"}])
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any("different number" in r.message.lower() for r in warnings)

    def test_count_mismatch_warns(self, caplog):
        text = "Hello <co>1</co> <co>2</co>."
        _validate_response(text, [{"id": "1"}])  # only 1 citation but 2 tags
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any("different number" in r.message.lower() for r in warnings)


# ---------------------------------------------------------------------------
# _get_docs_from_citations
# ---------------------------------------------------------------------------


class TestGetDocsFromCitations:
    def test_normal_docs(self):
        text = (
            '<co>1</co> Document 0: "First document text."\n'
            '<co>2</co> Document 1: "Second document text."'
        )
        result = _get_docs_from_citations(text)
        assert len(result) == 2
        assert result[0]["doc_id"] == "0"
        assert result[0]["citation_id"] == "1"
        assert result[0]["text"] == "First document text."

    def test_empty_input(self):
        assert _get_docs_from_citations("") == []
        assert _get_docs_from_citations(None) == []

    def test_whitespace_only(self):
        assert _get_docs_from_citations("   \n  \n  ") == []

    def test_non_numeric_doc_id_skipped(self, caplog):
        text = '<co>1</co> Document abc: "text"'
        result = _get_docs_from_citations(text)
        assert len(result) == 0

    def test_missing_colon_skipped(self):
        text = "<co>1</co> Document 0 no colon here"
        result = _get_docs_from_citations(text)
        assert result == []


# ---------------------------------------------------------------------------
# _update_docs_text_with_input_docs
# ---------------------------------------------------------------------------


class TestUpdateDocsTextWithInputDocs:
    def test_updates_subset_text(self):
        input_docs = [
            Document(text="RAG is retrieval-augmented generation. It is cool.")
        ]
        citation_docs = [
            {
                "citation_id": "1",
                "doc_id": "0",
                "text": "RAG is retrieval-augmented generation.",
            }
        ]
        result = _update_docs_text_with_input_docs(input_docs, citation_docs)
        assert result[0]["text"] == "RAG is retrieval-augmented generation. It is cool."

    def test_no_match_unchanged(self):
        input_docs = [Document(text="Completely different text.")]
        citation_docs = [
            {"citation_id": "1", "doc_id": "0", "text": "Original citation doc text."}
        ]
        result = _update_docs_text_with_input_docs(input_docs, citation_docs)
        assert result[0]["text"] == "Original citation doc text."

    def test_empty_input_docs(self):
        citation_docs = [{"citation_id": "1", "doc_id": "0", "text": "some text"}]
        result = _update_docs_text_with_input_docs([], citation_docs)
        assert result[0]["text"] == "some text"


# ---------------------------------------------------------------------------
# Granite32OutputProcessor.transform
# ---------------------------------------------------------------------------


class TestGranite32OutputProcessorTransform:
    @staticmethod
    def _minimal_cc(**kwargs) -> Granite32ChatCompletion:
        kwargs.setdefault("messages", [UserMessage(content="q")])
        return Granite32ChatCompletion(**kwargs)

    def test_plain_text_no_controls(self):
        proc = Granite32OutputProcessor()
        cc = self._minimal_cc()
        result = proc.transform("Hello, world!", cc)
        assert isinstance(result, Granite3AssistantMessage)
        assert result.content == "Hello, world!"
        assert result.citations is None
        assert result.hallucinations is None
        assert result.reasoning_content is None
        assert result.tool_calls == []

    def test_cot_reasoning_extraction(self):
        proc = Granite32OutputProcessor()
        model_output = (
            "Here is my thought process:\nI need to think carefully.\n"
            "Here is my response:\nThe answer is 42."
        )
        cc = self._minimal_cc(
            extra_body=VLLMExtraBody(chat_template_kwargs=Granite3Kwargs(thinking=True))
        )
        result = proc.transform(model_output, cc)
        assert result.reasoning_content == "I need to think carefully."
        assert result.content == "The answer is 42."

    def test_tool_call_parsing(self):
        proc = Granite32OutputProcessor()
        tool_json = [{"name": "get_weather", "arguments": {"city": "NYC"}}]
        model_output = f"<tool_call>{json.dumps(tool_json)}"
        cc = self._minimal_cc(tools=[ToolDefinition(name="get_weather")])
        result = proc.transform(model_output, cc)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"

    def test_invalid_tool_call_falls_through(self):
        proc = Granite32OutputProcessor()
        model_output = "<tool_call>not valid json"
        cc = self._minimal_cc(tools=[ToolDefinition(name="my_tool")])
        result = proc.transform(model_output, cc)
        # Should fall through to text parsing without crashing
        assert result.tool_calls == []
        assert isinstance(result.content, str)

    @require_nltk_data()
    def test_citations_and_hallucinations_pipeline(self):
        proc = Granite32OutputProcessor()
        model_output = (
            "RAG <co>1</co> is powerful.\n"
            f"{CITATION_START}\n"
            '<co>1</co> Document 0: "RAG is retrieval-augmented generation."\n'
            f"{HALLUCINATION_START}\n"
            "1. Risk low: RAG is powerful."
        )
        cc = self._minimal_cc(
            extra_body=VLLMExtraBody(
                documents=[Document(text="RAG is retrieval-augmented generation.")],
                chat_template_kwargs=Granite3Kwargs(
                    controls=Granite3Controls(citations=True, hallucinations=True)
                ),
            )
        )
        result = proc.transform(model_output, cc)
        assert result.citations is not None
        assert result.hallucinations is not None
        assert "RAG" in result.content
        assert result.raw_content is not None

    def test_plain_text_has_no_raw_content(self):
        proc = Granite32OutputProcessor()
        cc = self._minimal_cc()
        result = proc.transform("Simple answer.", cc)
        assert result.raw_content is None
