# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Granite 3.3 output processor."""

import json

from mellea.formatters.granite.base.types import (
    Document,
    ToolDefinition,
    UserMessage,
    VLLMExtraBody,
)
from mellea.formatters.granite.granite3.granite33.constants import (
    CITATIONS_START,
    CITE_END,
    CITE_START,
    HALLUCINATIONS_START,
)
from mellea.formatters.granite.granite3.granite33.output import (
    Granite33OutputProcessor,
    _add_citation_response_spans,
    _get_docs_from_citations,
    _parse_citations_text,
    _remove_citations_from_response_text,
    _remove_controls_output_from_response_text,
    _split_model_output_into_parts,
    _validate_response,
)
from mellea.formatters.granite.granite3.granite33.types import Granite33ChatCompletion
from mellea.formatters.granite.granite3.types import (
    Granite3AssistantMessage,
    Granite3Controls,
    Granite3Kwargs,
)

# ---------------------------------------------------------------------------
# _parse_citations_text
# ---------------------------------------------------------------------------


class TestParseCitationsText33:
    def test_single_citation(self):
        text = '1: "RAG is retrieval-augmented generation."'
        result = _parse_citations_text(text)
        assert len(result) == 1
        assert result[0]["doc_id"] == "1"
        assert result[0]["context_text"] == "RAG is retrieval-augmented generation."

    def test_multiple_citations(self):
        text = '1: "First text."\n2: "Second text."'
        result = _parse_citations_text(text)
        assert len(result) == 2
        assert result[0]["doc_id"] == "1"
        assert result[1]["doc_id"] == "2"

    def test_empty_returns_empty(self):
        result = _parse_citations_text("")
        assert result == []


# ---------------------------------------------------------------------------
# _remove_citations_from_response_text
# ---------------------------------------------------------------------------


class TestRemoveCitationsFromResponseText33:
    def test_removes_cite_tags(self):
        text = f'Hello {CITE_START}{{"document_id": "1"}}{CITE_END} world.'
        result = _remove_citations_from_response_text(text)
        assert CITE_START not in result
        assert CITE_END not in result
        assert "Hello" in result
        assert "world" in result

    def test_removes_plugin_markers(self):
        text = "Some <|start_of_plugin|>plugin<|end_of_plugin|> text."
        result = _remove_citations_from_response_text(text)
        assert "<|start_of_plugin|>" not in result
        assert "<|end_of_plugin|>" not in result

    def test_no_tags_unchanged(self):
        text = "Plain text without citations."
        result = _remove_citations_from_response_text(text)
        assert result == text


# ---------------------------------------------------------------------------
# _remove_controls_output_from_response_text
# ---------------------------------------------------------------------------


class TestRemoveControlsOutputFromResponseText:
    def test_no_controls_present(self):
        text = "Just a normal response."
        assert _remove_controls_output_from_response_text(text) == text

    def test_removes_citation_in_text_pattern(self):
        text = 'The answer is yes {"document_id": "1"} and also no.'
        result = _remove_controls_output_from_response_text(text)
        assert '{"document_id"' not in result

    def test_removes_control_responses_list(self):
        text = 'My response here. {"id": "citation"} some more text'
        result = _remove_controls_output_from_response_text(text)
        assert '{"id": "citation"}' not in result
        assert "My response here." in result

    def test_removes_hallucination_control(self):
        text = 'Answer. {"id": "hallucination"} rest'
        result = _remove_controls_output_from_response_text(text)
        assert '{"id": "hallucination"}' not in result


# ---------------------------------------------------------------------------
# _split_model_output_into_parts
# ---------------------------------------------------------------------------


class TestSplitModelOutputIntoParts33:
    def test_response_only(self):
        response, citations, hallucinations = _split_model_output_into_parts(
            "Just a response."
        )
        assert response == "Just a response."
        assert citations == ""
        assert hallucinations == ""

    def test_response_and_citations(self):
        output = f'My response.\n{CITATIONS_START}\n1: "cited text"'
        response, citations, hallucinations = _split_model_output_into_parts(output)
        assert response == "My response."
        assert "cited text" in citations
        assert hallucinations == ""

    def test_response_and_hallucinations(self):
        output = f"My response.\n{HALLUCINATIONS_START}\n1. Risk high: statement"
        response, citations, hallucinations = _split_model_output_into_parts(output)
        assert response == "My response."
        assert citations == ""
        assert "Risk high" in hallucinations

    def test_response_citations_and_hallucinations(self):
        output = (
            f"My response.\n{CITATIONS_START}\ncite text\n"
            f"{HALLUCINATIONS_START}\nhalluc text"
        )
        response, citations, hallucinations = _split_model_output_into_parts(output)
        assert response == "My response."
        assert "cite text" in citations
        assert "halluc text" in hallucinations

    def test_plugin_tokens_stripped(self):
        output = "Hello <|start_of_plugin|>X<|end_of_plugin|> world."
        response, _, _ = _split_model_output_into_parts(output)
        assert "<|start_of_plugin|>" not in response
        assert "<|end_of_plugin|>" not in response


# ---------------------------------------------------------------------------
# _validate_response
# ---------------------------------------------------------------------------


class TestValidateResponse33:
    def test_balanced_tags_no_warnings(self, caplog):
        text = (
            f'Hello {CITE_START}{{"document_id": "1"}}{CITE_END} '
            f'world {CITE_START}{{"document_id": "2"}}{CITE_END}.'
        )
        _validate_response(text, [{"id": "1"}, {"id": "2"}])
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 0

    def test_mismatched_tags_warns(self, caplog):
        text = f'Hello {CITE_START}{{"document_id": "1"}} world.'
        _validate_response(text, [{"id": "1"}])
        assert "different number" in caplog.text.lower()


# ---------------------------------------------------------------------------
# _get_docs_from_citations
# ---------------------------------------------------------------------------


class TestGetDocsFromCitations33:
    def test_normal_docs(self):
        text = '1: "First document."\n2: "Second document."'
        result = _get_docs_from_citations(text)
        assert len(result) == 2
        assert result[0]["doc_id"] == "1"
        assert result[0]["text"] == "First document."

    def test_empty_input(self):
        assert _get_docs_from_citations("") == []

    def test_non_numeric_skipped(self, caplog):
        text = 'abc: "text"'
        result = _get_docs_from_citations(text)
        assert len(result) == 0

    def test_special_token_lines_ignored(self):
        text = '<|something|>\n1: "Real doc."'
        result = _get_docs_from_citations(text)
        # The special token line has no colon-separated numeric id
        found_real = [d for d in result if d["text"] == "Real doc."]
        assert len(found_real) == 1


# ---------------------------------------------------------------------------
# _add_citation_response_spans
# ---------------------------------------------------------------------------


class TestAddCitationResponseSpans:
    """Tests for _add_citation_response_spans (issue #843 regression guard)."""

    def _make_citation(self, doc_id: str = "1") -> dict:
        return {"doc_id": doc_id, "context_text": "some context"}

    def test_response_end_uses_sentence_length_not_full_response(self):
        """Regression: response_end must be index + len(sentence), not index + len(full_response).

        Before the fix, _add_citation_response_spans used len(response_text_without_citations)
        — the full response length — instead of len(response_text) — the cited sentence length.
        This caused response_end to overshoot for any sentence that is not the last one.
        """
        sent1 = "Short sentence."
        sent2 = "This is the second sentence, which is longer."
        cite_tag = f'{CITE_START}{{"document_id": "1"}}{CITE_END}'
        response_with_citations = f"{sent1} {cite_tag} {sent2}"
        response_without_citations = f"{sent1} {sent2}"

        result = _add_citation_response_spans(
            [self._make_citation("1")],
            response_with_citations,
            response_without_citations,
        )

        assert len(result) == 1
        citation = result[0]
        begin = citation["response_begin"]
        end = citation["response_end"]
        text = citation["response_text"]

        # The span must exactly bracket the cited sentence in the clean response
        assert response_without_citations[begin:end] == text
        # Span length must equal the sentence length, not the full response length
        assert end - begin == len(text)
        assert end <= len(response_without_citations)

    def test_multiple_citations_each_span_correct(self):
        """Each citation span must cover only its own sentence."""
        sent1 = "First sentence."
        sent2 = "Second sentence."
        cite1 = f'{CITE_START}{{"document_id": "1"}}{CITE_END}'
        cite2 = f'{CITE_START}{{"document_id": "2"}}{CITE_END}'
        response_with = f"{sent1} {cite1} {sent2} {cite2}"
        response_without = f"{sent1} {sent2}"

        result = _add_citation_response_spans(
            [self._make_citation("1"), self._make_citation("2")],
            response_with,
            response_without,
        )

        assert len(result) == 2
        for citation in result:
            begin = citation["response_begin"]
            end = citation["response_end"]
            text = citation["response_text"]
            assert response_without[begin:end] == text
            assert end - begin == len(text)
            assert end <= len(response_without)

        # The two spans must not overlap
        spans = sorted((c["response_begin"], c["response_end"]) for c in result)
        assert spans[0][1] <= spans[1][0]

    def test_single_sentence_response(self):
        """Single-sentence response: span must cover the full clean response."""
        sent = "The only sentence."
        cite_tag = f'{CITE_START}{{"document_id": "1"}}{CITE_END}'
        response_with = f"{sent} {cite_tag}"
        response_without = sent

        result = _add_citation_response_spans(
            [self._make_citation("1")], response_with, response_without
        )

        assert len(result) == 1
        citation = result[0]
        begin = citation["response_begin"]
        end = citation["response_end"]
        assert response_without[begin:end] == citation["response_text"]


# ---------------------------------------------------------------------------
# Granite33OutputProcessor.transform
# ---------------------------------------------------------------------------


class TestGranite33OutputProcessorTransform:
    @staticmethod
    def _minimal_cc(**kwargs) -> Granite33ChatCompletion:
        kwargs.setdefault("messages", [UserMessage(content="q")])
        return Granite33ChatCompletion(**kwargs)

    def test_plain_text_no_controls(self):
        proc = Granite33OutputProcessor()
        cc = self._minimal_cc()
        result = proc.transform("Hello, world!", cc)
        assert isinstance(result, Granite3AssistantMessage)
        assert result.content == "Hello, world!"
        assert result.citations is None
        assert result.hallucinations is None
        assert result.tool_calls == []

    def test_think_response_extraction(self):
        proc = Granite33OutputProcessor()
        model_output = (
            "<think>\nLet me reason about this.\n</think>\n"
            "<response>\nThe answer is 42.\n</response>"
        )
        cc = self._minimal_cc(
            extra_body=VLLMExtraBody(chat_template_kwargs=Granite3Kwargs(thinking=True))
        )
        result = proc.transform(model_output, cc)
        assert result.reasoning_content == "Let me reason about this."
        assert result.content == "The answer is 42."

    def test_controls_none_cleans_old_format(self):
        """Issue #173: controls=None should clean old citation/hallucination format."""
        proc = Granite33OutputProcessor()
        model_output = 'The answer. {"id": "citation"} extra stuff'
        cc = self._minimal_cc()
        result = proc.transform(model_output, cc)
        assert '{"id": "citation"}' not in result.content

    def test_tool_call_parsing(self):
        proc = Granite33OutputProcessor()
        tool_json = [{"name": "search", "arguments": {"q": "test"}}]
        model_output = f"<tool_call>{json.dumps(tool_json)}"
        cc = self._minimal_cc(tools=[ToolDefinition(name="search")])
        result = proc.transform(model_output, cc)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"

    def test_raw_content_set_when_different(self):
        proc = Granite33OutputProcessor()
        model_output = (
            f'Response text {CITE_START}{{"document_id": "1"}}{CITE_END}.\n'
            f'{CITATIONS_START}\n1: "cited"'
        )
        cc = self._minimal_cc(
            extra_body=VLLMExtraBody(
                documents=[Document(text="cited stuff", doc_id="1")],
                chat_template_kwargs=Granite3Kwargs(
                    controls=Granite3Controls(citations=True)
                ),
            )
        )
        result = proc.transform(model_output, cc)
        assert result.raw_content is not None
