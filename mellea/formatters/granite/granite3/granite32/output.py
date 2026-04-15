# SPDX-License-Identifier: Apache-2.0

"""Parser for Granite 3.2 model output.

Parser which receives Granite 3.2 model output and returns the constituents of the
output.

The input to the parser is assumed to be as follows:
    ```
    response_text

    # Citations:
    citations_text

    # Hallucinations:
    hallucinations_text
    ```

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
from collections.abc import Sized

# Third Party
import pydantic

# First Party
from ...base.io import OutputProcessor
from ...base.optional import nltk_check
from ...base.types import AssistantMessage, ChatCompletion, Document, ToolCall
from ...base.util import find_substring_in_text, random_uuid
from ...granite3.output import (
    add_citation_context_spans,
    add_hallucination_response_spans,
    create_dict,
    parse_hallucinations_text,
)
from ...granite3.types import Granite3AssistantMessage

# Local
from .constants import (
    CITATION_START,
    COT_END_ALTERNATIVES,
    COT_START_ALTERNATIVES,
    HALLUCINATION_START,
)
from .types import Granite32ChatCompletion

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

    Given the citations text output by model under the "# Citations:" section,
    extract the citation info as an array of the form:

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

    # Find begin spans of all citations
    matches_iter = re.finditer("<co>(\\d+)</co>", citations_text)
    matches = []
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
        matches_iter = re.finditer(
            '(?s)<co>(\\d+)</co>\\s*Document (\\d+): "(.+)$', citation_str
        )
        idx = 0
        for match in matches_iter:
            # If the last character is a double quote (as expected), do not include
            # it in the string
            if match.group(3)[-1] == '"':
                context_text = match.group(3)[:-1]
            # Otherwise, continue but show a warning that there is an error in the
            # output format
            else:
                context_text = match.group(3)
                logger.warning(
                    f"Last character of citation is not a double "
                    f"quote in: '{context_text}'."
                )
            cur_citation = {
                "citation_id": match.group(1),
                "doc_id": match.group(2),
                "context_text": context_text,
            }
            citations.append(cur_citation)

            # If the citation contains a nested Document x mention, then show a warning
            if re.search(r"\nDocument (\d+)", cur_citation["context_text"]):
                logger.warning(
                    f"Citation text contains another document mention: "
                    f"{cur_citation['context_text']}"
                )

            idx += 1

        if idx == 0:
            logger.warning(
                "Error in finding components of citation: "
                "Expected single RegEx match but found none."
            )
        if idx > 1:
            logger.warning(
                "Error in finding components of citation: "
                "Expected single RegEx match but found several."
            )

    return citations


def _add_citation_response_spans(
    citation_info: list[dict],
    response_text_with_citations: str,
    response_text_without_citations: str,
) -> list[dict]:
    """Add response spans to citations.

    Given the response text in two forms (the original with citation tags and
    the processed without) and an array of parsed citations of the form:

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
    with nltk_check("Granite 3.2 citation support"):
        # Third Party
        import nltk

    augmented_citation_info = copy.deepcopy(citation_info)

    # Split response into sentences
    response_sentences = nltk.sent_tokenize(response_text_with_citations)

    # Create dictionary of the response sentence (cleaned from citations) corresponding
    # to each citation ID
    response_sents_by_citation_id = {}
    for sent_idx, sent in enumerate(response_sentences):
        matches_iter = re.finditer("<co>(\\d+)</co>", sent)
        for match in matches_iter:
            citation_id = match.group(1)
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
                            "Error in extracting the response sentence "
                            "of a citation: Found empty sentence"
                        )
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
    for citation in augmented_citation_info:
        if citation["citation_id"] in response_sents_by_citation_id:
            response_text = response_sents_by_citation_id[citation["citation_id"]]
            matches = find_substring_in_text(
                response_text, response_text_without_citations
            )
            if len(matches) == 0:
                logger.warning(
                    "Error in extracting the response sentence of a "
                    "citation: match not found in response."
                )
                continue

            if len(matches) > 1:
                # Find the citation ID and the text preceding it
                citation_id_matches_iter = re.finditer(
                    "<co>" + citation["citation_id"] + "</co>",
                    response_text_with_citations,
                )
                citation_id_matches = tuple(citation_id_matches_iter)
                if len(citation_id_matches) == 0:
                    logger.warning(
                        "Error in extracting the response sentence of a citation. "
                        "Citation ID does not appear in the response text."
                    )
                    continue
                citation_id_match_begin = citation_id_matches[0].start()

                text_before_citation_id = response_text_with_citations[
                    :citation_id_match_begin
                ]
                text_before_citation_id_without_citations = (
                    _remove_citations_from_response_text(text_before_citation_id)
                )

                # Find citation response text as the last match in the response text
                # preceding the citation ID. To do that, first remove the punctuation
                # from the end of the response text, since that would appear after the
                # citation ID
                search_str = re.sub(r"[\.\?\:\;]+$", "", response_text)
                response_text_matches = find_substring_in_text(
                    search_str, text_before_citation_id_without_citations
                )
                if len(citation_id_matches) == 0:
                    logger.warning(
                        "Error in extracting the response sentence of a citation."
                    )
                    continue
                last_response_text_match = response_text_matches[-1]

                citation["response_text"] = response_text
                citation["response_begin"] = last_response_text_match["begin_idx"]
                citation["response_end"] = last_response_text_match["begin_idx"] + len(
                    response_text
                )

                if (
                    citation["response_text"]
                    != response_text_without_citations[
                        citation["response_begin"] : citation["response_end"]
                    ]
                ):
                    logger.warning(
                        "Error in extracting the response sentence of a citation."
                    )
                    continue
            else:
                citation["response_text"] = response_text
                citation["response_begin"] = matches[0]["begin_idx"]
                citation["response_end"] = matches[0]["end_idx"]
        else:
            logger.warning(
                "Error in extracting the response sentence of a "
                "citation. Citation ID does not appear in the response "
                "text."
            )
            continue

    return augmented_citation_info


def _get_docs_from_citations(docs: str) -> list[dict]:
    """Extract documents from citations.

    Given a multi-line string with document information per line, extract
    and add to dictionary list with "doc_id" and "text" fields

    Document line format:
    <co>"<citation_id>"</co> Document "<document_id>": "<text>"

    For example:
    <co>1</co> Document 2: "RAG, retrieval-augmented generation..."
    """
    doc_dicts: list = []
    start_citation = "<co>"
    end_citation = "</co>"
    start_document = "Document "
    colon = ":"
    if not docs or docs.isspace():
        return doc_dicts
    for line in docs.splitlines():
        if not line or line.isspace():
            continue
        if (
            start_citation not in line
            or end_citation not in line
            or start_document not in line
            or colon not in line
        ):
            continue
        citation_id = line.split(start_citation)[1].split(end_citation)[0].strip()
        if not citation_id.isdigit():
            logger.warning(f"Unable to retrieve citation id from: '{line}'")
            continue
        doc_id = line.split(start_document)[1].split(colon)[0].strip()
        if not doc_id.isdigit():
            logger.warning(f"Unable to retrieve doc id from: '{line}'.")
            continue
        line_separated = line.split(colon, 1)
        if len(line_separated) <= 1:
            logger.warning(f"Unable to retrieve doc text from: '{line}'")
            continue
        text = line_separated[1].strip().strip('"')
        doc_dicts.append({"citation_id": citation_id, "doc_id": doc_id, "text": text})
    return doc_dicts


def _remove_citations_from_response_text(response_text: str) -> str:
    """Remove citations from response text.

    Given a response text (potentially containing inline <co>...</co> tags),
    return the response text cleaned up from the <co>...</co> tags.
    """
    return re.sub(r"\s*<co>\d+</co>\s*", "", response_text)


def _validate_response(response_text: str, citation_info: Sized):
    if re.search(
        r"<co>(?:(?!(<co>|</co>)).)*<co>(?:(?!(<co>|</co>)).)*</co>", response_text
    ):
        logger.warning(f"Response contains nested <co> tags: {response_text}")

    opening_tag_count = response_text.count("<co>")
    closing_tag_count = response_text.count("</co>")

    if opening_tag_count != closing_tag_count:
        logger.warning(
            f"Response contains different number of <co> and </co> "
            f"tags: {response_text}"
        )

    if opening_tag_count != len(citation_info):
        logger.warning(
            f"Response contains different number of citations than those "
            f"mentioned under '# Citations' in: '{response_text}'."
        )


def _split_model_output_into_parts(model_output: str) -> tuple[str, str, str]:
    """Divide model output into parts.

    Divide the model output into its constituent parts: response_text, citations_text,
    and hallucinations_text.

    This assumes that the model output is of the following form:

    response_text
    # Citations
    citations_text
    # Hallucinations
    hallucinations_text

    where:
    - "# Citations" and "# Hallucinations> are literals, and
    - response_text, citations_text, hallucinations_text are variables corresponding
        the parts of the output

    Note:
    - The citations and hallucinations sections are both optional
    """
    response_text = ""
    citations_text = ""
    hallucinations_text = ""

    if HALLUCINATION_START in model_output and CITATION_START not in model_output:
        # rsplit because sometimes the model produces multiple copies of the
        # hallucinations output.
        response_text, hallucinations_text = model_output.rsplit(HALLUCINATION_START, 1)
    elif CITATION_START in model_output and HALLUCINATION_START not in model_output:
        response_text, citations_text = model_output.split(CITATION_START)
    elif CITATION_START in model_output and HALLUCINATION_START in model_output:
        pre_citation_split, post_citation_split = model_output.split(CITATION_START)
        if HALLUCINATION_START in pre_citation_split:
            response_text, hallucinations_text = pre_citation_split.split(
                HALLUCINATION_START
            )
            citations_text = post_citation_split
        else:
            citations_text, hallucinations_text = post_citation_split.split(
                HALLUCINATION_START
            )
            response_text = pre_citation_split
    else:
        response_text = model_output

    return response_text.strip(), citations_text.strip(), hallucinations_text.strip()


def _validate_spans_in_parser_output(parsed_task: dict):
    """Validate spans in parser output.

    Validate that the hallucination and citation spans correspond to
    the model response.
    """
    for hallucination in (
        parsed_task["hallucinations"] if parsed_task["hallucinations"] else []
    ):
        if (
            hallucination["response_text"]
            != parsed_task["response"][
                hallucination["response_begin"] : hallucination["response_end"]
            ]
        ):
            logger.warning(
                "Hallucination span does not correspond to the model response."
            )
    for citation in parsed_task["citations"] if parsed_task["citations"] else []:
        if (
            citation["response_text"]
            != parsed_task["response"][
                citation["response_begin"] : citation["response_end"]
            ]
        ):
            logger.warning("Citation span does not correspond to the model response.")
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
            logger.warning(
                "Citation text does no correspond to the generated "
                "citation context span."
            )


def _update_docs_text_with_input_docs(
    docs_from_input: list, docs_from_citation: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Update citation docs with input docs text.

    The documents passed in the chat completion call is the source of the documents
    used for the model output. The document text output by the model may not show
    the full context. Therefore, to have the full context, need to retrieve from the
    input passed to chat completion call, all documents and update the text in the
    citation documents to be aligned before finding the context spans.
    """
    augmented_docs_from_citation = copy.deepcopy(docs_from_citation)
    for citation_doc in augmented_docs_from_citation:
        for input_doc in docs_from_input:
            if citation_doc["text"].strip() in input_doc.text.strip():
                citation_doc["text"] = input_doc.text.strip()

    return augmented_docs_from_citation


def _parse_model_output(
    model_output: str, docs_from_input: list[Document] | None
) -> dict:
    """Parse model output constituents.

    Parse the constituents of the output (response) of a model into
    a format where they can be accessed individually

    Args:
        model_output: The response from model request
        docs_from_input: The documents passed in the chat completion call.

    Returns:
        Parsed part of the model output as follows:
            "docs": Document references
            "response": Model response without citations,
            "citations": Citations,
            "hallucinations": Hallucinations
    }
    """
    docs_from_input = [] if docs_from_input is None else docs_from_input

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

    # Model output
    logger.debug(f"Model output:\n{model_output}\n")

    # Parsed response text
    response_text_without_citations = _remove_citations_from_response_text(
        response_text
    ).strip()
    logger.debug(f"Parsed response text:\n{response_text_without_citations}\n")

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


class Granite32OutputProcessor(OutputProcessor):
    """Output processor for version 3.2 of the main Granite models, all sizes."""

    def transform(
        self, model_output: str, chat_completion: ChatCompletion | None = None
    ) -> AssistantMessage:
        """Parse Granite 3.2 model output into a structured assistant message.

        Args:
            model_output (str): Raw text output from the Granite 3.2 model.
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
        if chat_completion is None:
            chat_completion = Granite32ChatCompletion(messages=[])
        else:
            chat_completion = Granite32ChatCompletion.model_validate(
                chat_completion.model_dump()
            )
        have_thinking = chat_completion.thinking()

        # Save a copy because code below mutates this variable
        original_output = model_output

        # Parse out CoT reasoning
        cot = None
        if have_thinking:
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

        # Parse out tool calls
        tool_calls = []
        if chat_completion.tools and model_output.startswith("<tool_call>"):
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
            parsed_output = _parse_model_output(
                model_output, chat_completion._documents()
            )
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
