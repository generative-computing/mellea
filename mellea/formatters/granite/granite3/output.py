# SPDX-License-Identifier: Apache-2.0

"""Output processing code that is specific to the Granite 3 family of models, but not specific to a particular point release."""

# Standard
import copy
import logging
import re
import sys
from collections.abc import Callable

# First Party
from ..base.util import find_substring_in_text

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(
    logging.Formatter("%(levelname)s %(asctime)s %(message)s", datefmt="%H:%M:%S")
)
logger.addHandler(handler)


def create_dict(input_array, **key_attrib_names: str) -> dict:
    """Return a dict indexed by given attributes from an array of dicts.

    Given an array of dicts and the name of attribute(s) within the array, return a
    dict containing the contents of the array indexed by the given attribute(s)
    """
    new_dict: dict = {}

    for item in input_array:  # type: ignore[union-attr]
        new_dict_key_val: str = ""
        key_attribs_len = len(key_attrib_names)
        # Key for dictionary will be a combinations of attribute(s)
        # the dictionary that we are trying to index
        for key_attrib in key_attrib_names.values():
            new_dict_key_val += item[key_attrib]
            key_attribs_len -= 1
            if key_attribs_len > 0:
                new_dict_key_val += "-"

        if new_dict_key_val in new_dict:
            logger.warning(
                f"Found duplicate item while creating dictionary: "
                f"'{new_dict[new_dict_key_val]}'."
            )

        new_dict[new_dict_key_val] = item

    return new_dict


def parse_hallucinations_text(hallucinations_text: str) -> list[dict]:
    """Extract hallucinations info as an array from the "# Hallucinations:" section.

    Given the hallucinations text output by model under the "# Hallucinations:"
    section, extract the hallucinations info as an array of the form:

    [
        {
            "hallucination_id": "Hallucination ID output by model",
            "risk": "Hallucination risk flag",
            "response_text": "Substring of response text for which
                                hallucination risk is computed"
        },
        ...
    ]
    """
    hallucinations = []

    # Find begin spans of all hallucinations
    matches_iter = re.finditer(
        "(\\d+)\\.\\s*Risk (low|high|unanswerable):", hallucinations_text
    )
    matches = []
    for match in matches_iter:
        matches.append({"match_begin": match.start()})

    if len(matches) == 0:
        logger.warning(
            "Failed to extract hallucination info."
            "Expected hallucination info but none found."
        )

    # For each hallucination, extract its components (hallucination ID,
    # risk, response text)
    for i in range(len(matches)):  # pylint: disable=consider-using-enumerate
        cur_match = matches[i]

        # Select text corresponding to hallucination (which is the text from the
        # beginning of the hallucination until the beginning of the next
        # hallucination or the end of the text; whichever comes first)
        if i + 1 < len(matches):
            next_match_begin = matches[i + 1]["match_begin"] - 1
        else:
            next_match_begin = len(hallucinations_text)
        hallucination_str = hallucinations_text[
            cur_match["match_begin"] : next_match_begin
        ]

        # Within the hallucination text, extract the citation components
        # (hallucination ID, risk, response text)
        # Use ?s flag to include newlines in match
        matches_iter = re.finditer(
            "(?s)(\\d+)\\.\\s*Risk (low|high|unanswerable): (.+)$", hallucination_str
        )
        idx = 0
        for match in matches_iter:
            cur_hallucination = {
                "hallucination_id": match.group(1),
                "risk": match.group(2),
                "response_text": match.group(3),
            }
            hallucinations.append(cur_hallucination)

            idx += 1

        if idx == 0:
            logger.warning(
                "Error in finding components of hallucination: "
                "Expected single RegEx match but found none."
            )
        if idx > 1:
            logger.warning(
                "Error in finding components of hallucination: "
                "Expected single RegEx match but found several."
            )

    return hallucinations


def add_hallucination_response_spans(
    hallucination_info: list[dict],
    response_text_without_citations: str,
    remove_citations_from_response_text: Callable,
) -> list[dict]:
    """Add response spans to hallucination info.

    Given the response text (cleaned from citation tags) and a
    parsed hallucinations info of the form:

    [
        {
            "hallucination_id": "Hallucination ID output by model",
            "risk": "Hallucination risk flag",
            "response_text": "Substring of response text for which hallucination
                                risk is computed"
        },
        ...
    ]

    add to each hallucination element in the array the following attributes
    (the "response_text" replaces the attribute of the same name):

        "response_text": "The response text corresponding to the hallucination
                            element cleaned from citation tags"
        "response_begin": "The begin index of "response_text" within the response
                            text (without citation tags)"
        "response_end": "The end index of "response_text" within the response
                            text (without citation tags)"
    """
    augmented_hallucination_info = copy.deepcopy(hallucination_info)

    for hallucination in augmented_hallucination_info:
        hallucination_response_text_without_citations = (
            remove_citations_from_response_text(hallucination["response_text"])
        )
        matches = find_substring_in_text(
            hallucination_response_text_without_citations,
            response_text_without_citations,
        )
        if len(matches) == 0:
            logger.warning(
                "Error in adding the response spans to hallucination: "
                "Hallucination text not found in response"
            )
            # Install placeholder values to avoid breaking downstream code.
            hallucination["response_begin"] = 0
            hallucination["response_end"] = 0
            continue

        if len(matches) > 1:
            logger.warning(
                "Hallucination text found multiple times in "
                "response: Selecting first match"
            )
        hallucination["response_text"] = hallucination_response_text_without_citations
        hallucination["response_begin"] = matches[0]["begin_idx"]
        hallucination["response_end"] = matches[0]["end_idx"]

    return augmented_hallucination_info


def add_citation_context_spans(
    citation_info: list[dict], docs: list[dict]
) -> list[dict]:
    """Add context spans to citation info.

    Given a set of docs and an array of citations of the form:

    [
        {
            "citation_id": "Citation ID output by model",
            "doc_id": "ID of doc where the cited text is drawn from",
            "context_text": "The cited text from the context"
        },
        ...
    ]

    add to each citation in the array the following two attributes:

        "context_begin": "The begin index of "context_text" within document with
                            ID doc_id"
        "context_end": "The end index of "context_text" within document with ID doc_id"
    """
    augmented_citation_info = copy.deepcopy(citation_info)
    docs_by_cit_doc_id = create_dict(
        docs, citation_attrib="citation_id", document_attrib="doc_id"
    )
    for citation in augmented_citation_info:
        # Init values in event of error in processing
        citation["context_begin"] = 0
        citation["context_end"] = 0
        try:
            dict_id = citation["citation_id"] + "-" + citation["doc_id"]
            doc = docs_by_cit_doc_id[dict_id]
        except KeyError:
            logger.warning(
                f"Document with id: {dict_id} not found "
                f"when adding citation context spans."
            )
            continue

        matches = find_substring_in_text(citation["context_text"], doc["text"])
        if len(matches) == 0:
            logger.warning(
                "Error in adding the context spans to citation: "
                "Cited text not found in corresponding document."
            )
            continue

        if len(matches) > 1:
            logger.warning(
                "Cited text found multiple times in corresponding "
                "document: Selecting first match."
            )
        citation["context_begin"] = matches[0]["begin_idx"]
        citation["context_end"] = matches[0]["end_idx"]

    return augmented_citation_info
