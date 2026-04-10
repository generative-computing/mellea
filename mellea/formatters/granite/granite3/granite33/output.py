# SPDX-License-Identifier: Apache-2.0

"""Parser which receives Granite 3.3 model output and returns the constituents of the output.

The output from the lowest level of the parser is a dictionary as follows:

    * "citations": List of citations
    * "docs": List of document references
    * "hallucinations": List of hallucinations
    * "response": Model response text without the above constituents

This dict is further refined into dataclasses before being returned as an extended
``AssistantMessage``.
"""

# Standard
import copy
import json
import logging
import re
import sys

# Third Party
import pydantic

# First Party
from ...base.io import OutputProcessor
from ...base.optional import nltk_check
from ...base.types import AssistantMessage, ChatCompletion, ToolCall
from ...base.util import random_uuid
from ...granite3.output import (
    add_citation_context_spans,
    add_hallucination_response_spans,
    create_dict,
    parse_hallucinations_text,
)
from ...granite3.types import Granite3AssistantMessage

# Local
from .constants import (
    CITATIONS_START,
    CITE_END,
    CITE_START,
    COT_END_ALTERNATIVES,
    COT_START_ALTERNATIVES,
    HALLUCINATIONS_START,
    RESPONSE_END,
    RESPONSE_START,
)
from .types import Granite33ChatCompletion

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(
    logging.Formatter("%(levelname)s %(asctime)s %(message)s", datefmt="%H:%M:%S")
)
logger.addHandler(handler)


def _parse_citations_text(citations_text: str) -> list[dict]:
    """Parse citations text from model output.

    Given the citations text output by model under the "# Citations:" section, extract
    the citation info as an array of the form:
    ```
    [
        {
            "citation_id": "Citation ID output by model",
            "doc_id": "ID of doc where the cited text is drawn from",
            "context_text": "The cited text from the context"
        },
        ...
    ]
    ```
    """
    citations: list[dict] = []

    # Find citations in the response
    pattern = r"(\d+):(.+)"
    matches_iter = re.finditer(pattern, citations_text)
    matches: list[dict] = []
    for match in matches_iter:
        matches.append({"match_begin": match.start()})

    if len(matches) == 0:
        logger.warning(
            "Error in extracting citation info. Expected citations but found none."
        )
        return citations

    # For each citation, extract its components (citation ID, doc ID, context text)
    for i in range(len(matches)):  # pylint: disable=consider-using-enumerate
        cur_match = matches[i]

        # Select text corresponding to citation (which is the text from the beginning
        # of the citation until the beginning of the next citation or the end of the
        # text; whichever comes first)
        if i + 1 < len(matches):
            next_match_begin = matches[i + 1]["match_begin"] - 1
        else:
            next_match_begin = len(citations_text)
        citation_str = citations_text[cur_match["match_begin"] : next_match_begin]

        # Within the citation text, extract the citation components
        # (citation ID, doc ID, context text)
        # Use ?s flag to include newlines in match
        pattern = r"(\d+):(.+)"
        matches_iter = re.finditer(pattern, citation_str)
        idx = 0
        for match in matches_iter:
            context_text = match.group(2).strip().strip('"')

            cur_citation = {
                "citation_id": str(idx),
                "doc_id": match.group(1),
                "context_text": context_text,
            }
            citations.append(cur_citation)

            idx += 1

        if idx == 0:
            logger.warning(
                "Error in finding components of citation: Expected single RegEx match but found none."
            )
        if idx > 1:
            logger.warning(
                "Error in finding components of citation: Expected single RegEx match but found several."
            )

    return citations


def _add_citation_response_spans(
    citation_info: list[dict],
    response_text_with_citations: str,
    response_text_without_citations: str,
) -> list[dict]:
    """Given the response text in two forms (the original with citation tags and the processed without) and an array of parsed citations of the form.

    [
        {
            "citation_id": "Citation ID output by model",
            "doc_id": "ID of doc where the cited text is drawn from",
            "context_text": "The cited text from the context",
            "context_begin": "The begin index of "context_text" within document with
                                ID doc_id" (Optional)
            "context_end": "The end index of "context_text" within document with
                                ID doc_id" (Optional)
        },
        ...
    ]

    add to each citation in the array the following two attributes:
        "response_text": "The substring of the response for which the citation is
                            provided"
        "response_begin": "The begin index of "response_text" within the response text"
        "response_end": "The end index of "response_text" within the response text"
    """
    with nltk_check("Granite 3.3 citation support"):
        # Third Party
        import nltk

    augmented_citation_info = copy.deepcopy(citation_info)

    # Split response into sentences
    response_sentences = nltk.sent_tokenize(response_text_with_citations)

    # Create dictionary of the response sentence (cleaned from citations) corresponding
    # to each citation ID
    response_sents_by_citation_id = {}
    citation_idx = 0
    for sent_idx, sent in enumerate(response_sentences):
        # pylint: disable=anomalous-backslash-in-string
        pattern = (
            f'{re.escape(CITE_START)}{{"document_id": "(\\d+)"}}{re.escape(CITE_END)}'
        )
        matches_iter = re.finditer(pattern, sent)
        for match in matches_iter:
            citation_id = str(citation_idx)
            if citation_idx > len(augmented_citation_info) - 1:
                augmented_citation_info.append({})  # Too many, add extra
            augmented_citation_info[citation_idx]["citation_id"] = citation_id
            augmented_citation_info[citation_idx]["document_id"] = match.group(1)
            citation_idx = citation_idx + 1
            if citation_id not in response_sents_by_citation_id:
                sent_without_citations = _remove_citations_from_response_text(sent)
                # Fixes for sentence splitting issue: Associate citation with previous
                # sentence if either of the following holds:
                # - The citation ID appears at the beginning of a sentence
                # - The found sentence is empty after removing citations
                if (match.start() == 0) or (len(sent_without_citations) == 0):
                    if sent_idx > 0:
                        sent_without_citations = _remove_citations_from_response_text(
                            response_sentences[sent_idx - 1]
                        )
                    else:
                        logger.warning(
                            "Error in extracting the response sentence of a citation: Found empty sentence"
                        )
                        response_sents_by_citation_id[citation_id] = ""
                        continue
                response_sents_by_citation_id[citation_id] = sent_without_citations
            else:
                logger.warning(
                    "Error in extracting the response sentence of a "
                    "citation: Citation ID appears in more than one "
                    "response sentences"
                )
                continue

    # For each citation bring the response sentence to which it refers and its
    # begin/end spans
    for i, citation in enumerate(augmented_citation_info):
        response_text = response_sents_by_citation_id.get(str(i), "")
        index = response_text_without_citations.find(response_text)
        if index < 0:
            logger.warning(
                "Error in extracting the response sentence of a citation: Unexpected error."
            )
            continue

        citation["response_text"] = response_text
        citation["response_begin"] = index
        citation["response_end"] = index + len(response_text_without_citations)

    return augmented_citation_info


def _get_docs_from_citations(docs: str) -> list[dict]:
    """Extract documents from citations.

    Given a multi-line string with document information per line, extract
    and add to dictionary list with "doc_id" and "text" fields.

    Document line format:
    1: "<text>"
    2: "<text>"
    <|something to ignore|>
    """
    doc_dicts: list[dict] = []
    if not docs or docs.isspace():
        return doc_dicts
    for i, line in enumerate(docs.splitlines()):
        if not line or line.isspace():
            continue
        line_split = line.split(":", maxsplit=1)
        if len(line_split) <= 1:
            logger.debug(f"Unable to retrieve doc text from: '{line}'.")
            continue
        doc_id = line_split[0].strip()
        if not doc_id.isdigit():
            logger.warning(f"Unable to retrieve doc id from: '{line}'.")
            continue
        text = line_split[1].strip().strip('"')

        # Using line index as citation_id
        doc_dicts.append({"citation_id": str(i), "doc_id": doc_id, "text": text})
    return doc_dicts


def _remove_citations_from_response_text(response_text: str) -> str:
    """Remove citation tags from response text.

    Given a response text (potentially containing inline <co>...</co> tags),
    return the response text cleaned up from the <co>...</co> tags.
    """
    # TODO:  Need a better way to turn on and then remove special tokens
    sop = "<|start_of_plugin|>"
    ret = response_text.replace(sop, "")
    eop = "<|end_of_plugin|>"
    ret = ret.replace(eop, "")
    pattern = f'{re.escape(CITE_START)}{{"document_id": "\\d+"}}{re.escape(CITE_END)}'
    return re.sub(pattern, "", ret).strip()


def _remove_controls_output_from_response_text(response_text: str) -> str:
    """Issue #173, no controls were specified but sometimes appear in the output.

    Clean the response text of any controls output and return it.
    """
    regex_citation_in_text = r" \{\"document_id\": \"\d+\"\}"
    regex_control_responses_list = r"\{\"id\": \"(citation|hallucination)\"\}"

    cleaned = response_text

    # Remove all citations in the response
    cleaned = re.sub(regex_citation_in_text, "", cleaned)

    # Remove the specific list of outputs from controls based on their regex
    match = re.search(regex_control_responses_list, cleaned, re.DOTALL)
    if match:
        cleaned = cleaned[: match.start()].strip()
    return cleaned


def _validate_response(response_text: str, citation_info):
    start = re.escape(CITE_START)
    end = re.escape(CITE_END)
    pattern = f"{start}(?:(?!({start}|{end})).)*{start}(?:(?!({start}|{end})).)*{end}"
    if re.search(pattern, response_text):
        logger.warning(f"Response contains nested citations: '{response_text}'.")

    opening_tag_count = response_text.count(CITE_START)
    closing_tag_count = response_text.count(CITE_END)

    if opening_tag_count != closing_tag_count:
        logger.warning(
            f"Response contains different number of cite start and end symbols: "
            f"'{response_text}'."
        )

    if opening_tag_count != len(citation_info):  # type: ignore[arg-type]
        logger.warning(
            f"Response contains different number of citations than those listed': "
            f"{response_text}"
        )


def _split_model_output_into_parts(model_output: str) -> tuple[str, str, str]:
    """Divide model output into response, citations, and hallucinations."""
    citations_text = ""
    hallucinations_text = ""

    if HALLUCINATIONS_START in model_output and CITATIONS_START not in model_output:
        # rsplit because sometimes the model produces multiple copies of the
        # hallucinations output.
        response_text, hallucinations_text = model_output.rsplit(
            HALLUCINATIONS_START, 1
        )
    elif CITATIONS_START in model_output and HALLUCINATIONS_START not in model_output:
        response_text, citations_text = model_output.split(CITATIONS_START)
    elif CITATIONS_START in model_output and HALLUCINATIONS_START in model_output:
        pre_citation_split, post_citation_split = model_output.split(CITATIONS_START)
        if HALLUCINATIONS_START in pre_citation_split:
            response_text, hallucinations_text = pre_citation_split.split(
                HALLUCINATIONS_START
            )
            citations_text = post_citation_split
        else:
            citations_text, hallucinations_text = post_citation_split.split(
                HALLUCINATIONS_START
            )
            response_text = pre_citation_split
    else:
        response_text = model_output

    # TODO:  Need a better way to turn on and then remove special tokens
    sop = "<|start_of_plugin|>"
    eop = "<|end_of_plugin|>"
    response_text = response_text.replace(sop, "").replace(eop, "")
    citations_text = citations_text.replace(sop, "").replace(eop, "")
    hallucinations_text = hallucinations_text.replace(sop, "").replace(eop, "")
    return response_text.strip(), citations_text.strip(), hallucinations_text.strip()


def _validate_spans_in_parser_output(parsed_task: dict):
    """Validate that hallucination and citation spans correspond to the model response."""
    hallucinations = parsed_task.get("hallucinations")
    for hallucination in hallucinations if hallucinations else []:
        if "response_end" in hallucination and (
            hallucination["response_text"]
            != parsed_task["response"][
                hallucination["response_begin"] : hallucination["response_end"]
            ]
        ):
            logger.warning(
                "Hallucination span does not correspond to the model response."
            )
    citations = parsed_task.get("citations")
    for citation in citations if citations else []:
        docs_by_cit_doc_id = create_dict(
            parsed_task["docs"], citation_attrib="citation_id", document_attrib="doc_id"
        )
        try:
            dict_id = citation["citation_id"] + "-" + citation["doc_id"]
            doc = docs_by_cit_doc_id[dict_id]
        except KeyError:
            logger.warning(
                f"Document with id: {dict_id} not found "
                f"when validation citation context spans."
            )
            continue
        if (
            citation["context_text"]
            != doc["text"][citation["context_begin"] : citation["context_end"]]
        ):
            logger.warning("Citation span does not correspond to the model response.")


def _update_docs_text_with_input_docs(
    docs_from_input: list | None, docs_from_citation: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Update citation docs with input docs text.

    The documents passed in the chat completion call is the source of the documents
    used for the model output. The document text output by the model may not show
    the full context. Therefore, to have the full context, need to retrieve from the
    input passed to chat completion call, all documents and update the text in the
    citation documents to be aligned before finding the context spans.
    """
    augmented_docs_from_citation = copy.deepcopy(docs_from_citation)
    if docs_from_input is None:
        return augmented_docs_from_citation
    for citation_doc in augmented_docs_from_citation:
        for input_doc in docs_from_input:
            if citation_doc["text"].strip() in input_doc.text.strip():
                citation_doc["text"] = input_doc.text.strip()

    return augmented_docs_from_citation


def _parse_model_output(
    model_output: str, chat_completion: Granite33ChatCompletion
) -> dict:
    """Parse the constituents of the output (response) of a model into a format where they can be accessed individually.

    Args:
        model_output: The response from model request
        chat_completion: The full input given to the model, required to fix issue with
                old controls format appearing and to access documents for the
                new citations format in the model output
    Returns:
        Parsed part of the model output as follows:
            "docs": Document references
            "response": Model response without citations,
            "citations": Citations,
            "hallucinations": Hallucinations
    }
    """
    # Issue #173, no controls were specified but old citations and hallications
    # format sometimes appear in the output.
    # Clean the response text of these old controls format.
    if chat_completion.controls is None:
        response_text_without_controls = _remove_controls_output_from_response_text(
            model_output
        )
        result = {
            "docs": None,
            "response": response_text_without_controls,
            "citations": None,
            "hallucinations": None,
        }
        logger.debug(f"Response text without controls:\n{result}\n")
        return result

    docs_from_input = chat_completion._documents()

    # Split model output into its parts: response, citation, and hallucination section
    response_text, citations_text, hallucinations_text = _split_model_output_into_parts(
        model_output
    )

    # Get documents from citations
    docs_from_citation = _get_docs_from_citations(citations_text)

    # Update 'docs_from_citation' with text from docs used as input to model prompt
    # as they are the full source of text. The full text is required to retrieve
    # context spans.
    docs = _update_docs_text_with_input_docs(docs_from_input, docs_from_citation)

    # Parsed response text
    response_text_without_citations = _remove_citations_from_response_text(
        response_text
    ).strip()

    # Parse hallucinations text
    if len(hallucinations_text) > 0:
        hallucination_info = parse_hallucinations_text(hallucinations_text)
        augmented_hallucination_info = add_hallucination_response_spans(
            hallucination_info,
            response_text_without_citations,
            _remove_citations_from_response_text,
        )
    else:
        augmented_hallucination_info = []
    logger.debug(f"Parsed hallucination info:\n{augmented_hallucination_info}\n")

    # Parse citations text
    if len(citations_text) > 0:
        citation_info = _parse_citations_text(citations_text)
        citation_info_with_context_spans = add_citation_context_spans(
            citation_info, docs
        )
        citation_info_with_context_response_spans = _add_citation_response_spans(
            citation_info_with_context_spans,
            response_text,
            response_text_without_citations,
        )
        _validate_response(response_text, citation_info)
    else:
        citation_info_with_context_response_spans = []
    logger.debug(
        f"Parsed citation info:\n{citation_info_with_context_response_spans}\n"
    )

    # Join all objects into single output
    result = {
        "docs": docs if docs else None,
        "response": response_text_without_citations,
        "citations": (
            citation_info_with_context_response_spans
            if citation_info_with_context_response_spans
            else None
        ),
        "hallucinations": (
            augmented_hallucination_info if augmented_hallucination_info else None
        ),
    }
    logger.debug(f"Combined parser output:\n{result}\n")

    # Validate spans in parser output by checking if the citation/response text
    # matches the begin/end spans
    _validate_spans_in_parser_output(result)

    return result


class Granite33OutputProcessor(OutputProcessor):
    """Output processor for version 3.3 of the main Granite models, all sizes."""

    def transform(
        self, model_output: str, chat_completion: ChatCompletion | None = None
    ) -> AssistantMessage:
        """Parse Granite 3.3 model output into a structured assistant message.

        Args:
            model_output (str): Raw text output from the Granite 3.3 model.
            chat_completion (ChatCompletion | None): The original chat completion
                request that produced ``model_output``. Used to determine which
                output features (thinking, tools, citations, hallucinations) to
                parse. Defaults to ``None``.

        Returns:
            AssistantMessage: A :class:`Granite3AssistantMessage` containing the
                parsed response text, optional tool calls, chain-of-thought
                reasoning, citations, documents, and hallucination annotations.

        Raises:
            ValueError: If parsing citations, documents, or hallucinations from
                the model output fails.
        """
        # Downcast to a Granite-specific request type with possible additional fields.
        # This operation also performs additional validation.
        if chat_completion is None:
            chat_completion = Granite33ChatCompletion(messages=[])
        inputs = Granite33ChatCompletion.model_validate(chat_completion.model_dump())

        # Save a copy because code below mutates this variable
        original_output = model_output

        # Parse out CoT reasoning
        cot = None
        if inputs.thinking():
            cot_start_span = None
            cot_end_span = None
            for cot_start_str in COT_START_ALTERNATIVES:
                if (cot_start_pos := model_output.find(cot_start_str)) != -1:
                    cot_start_span = (cot_start_pos, cot_start_pos + len(cot_start_str))
                    break
            for cot_end_str in COT_END_ALTERNATIVES:
                if (cot_end_pos := model_output.find(cot_end_str)) != -1:
                    cot_end_span = (cot_end_pos, cot_end_pos + len(cot_end_str))
                    break

            if cot_start_span and cot_end_span and cot_end_span[0] > cot_start_span[1]:
                cot = model_output[cot_start_span[1] : cot_end_span[0]].strip()
                model_output = (
                    model_output[: cot_start_span[0]]
                    + model_output[cot_end_span[1] :].strip()
                )
                resp_start_span = None
                resp_end_span = None
                if (resp_start_pos := model_output.find(RESPONSE_START)) != -1:
                    resp_start_span = (
                        resp_start_pos,
                        resp_start_pos + len(RESPONSE_START),
                    )
                if (resp_end_pos := model_output.find(RESPONSE_END)) != -1:
                    resp_end_span = (resp_end_pos, resp_end_pos + len(RESPONSE_END))
                if (
                    resp_start_span
                    and resp_end_span
                    and resp_end_span[0] > resp_start_span[1]
                ):
                    model_output = model_output[
                        resp_start_span[1] : resp_end_span[0]
                    ].strip()

        # Parse out tool calls
        tool_calls = []
        if inputs.tools and model_output.startswith("<tool_call>"):
            # Basic tool call parsing: assume well-formed JSON that adheres to the
            # argument schema specified in the request.
            try:
                tool_calls = json.loads(model_output[len("<tool_call>") :])
                if not isinstance(tool_calls, list):
                    raise TypeError("Model didn't output a list of tool calls")
                tool_calls = [
                    ToolCall.model_validate(tool_call_json)
                    for tool_call_json in tool_calls
                ]
                for function_call in tool_calls:
                    # Model may decide not to produce IDs
                    if function_call.id is None:
                        function_call.id = random_uuid()
                # Output has been turned to tools
                model_output = ""
            except (ValueError, TypeError, pydantic.ValidationError):
                # Parsing failed; flow through
                pass

        # Parse out citations, documents and hallucinations
        try:
            parsed_output = _parse_model_output(model_output, inputs)
        except Exception as err:
            raise ValueError(
                "Failed to parse citations, documents and hallucinations "
                "from model output."
            ) from err

        content = parsed_output["response"]
        raw_content = original_output if original_output != content else None

        return Granite3AssistantMessage(
            citations=parsed_output["citations"],
            content=content,
            documents=parsed_output["docs"],
            hallucinations=parsed_output["hallucinations"],
            reasoning_content=cot,
            raw_content=raw_content,
            tool_calls=tool_calls,
        )
