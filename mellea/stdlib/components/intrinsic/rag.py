"""Intrinsic functions related to retrieval-augmented generation."""

import collections.abc

from ....backends.adapters import AdapterMixin
from ...components import Document
from ...context import ChatContext
from ..chat import Message
from ._util import call_intrinsic as _call_intrinsic

_ANSWER_RELEVANCE_CORRECTION_METHODS = {
    "Excessive unnecessary information": "removing the excessive information from the "
    "draft response",
    "Unduly restrictive": "providing answer without the unwarranted restriction, or "
    "indicating that the desired answer is not available",
    "Too vague or generic": "providing more crisp and to-the-point answer, or "
    "indicating that the desired answer is not available",
    "Contextual misalignment": "providing a response that answers the last user "
    "inquiry, taking into account the context of the conversation",
    "Misinterpreted inquiry": "providing answer only to the correct interpretation of "
    "the inquiry, or attempting clarification if the inquiry is ambiguous or otherwise "
    "confusing, or indicating that the desired answer is not available",
    "No attempt": "providing a relevant response if an inquiry should be answered, or "
    "providing a short response if the last user utterance contains no inquiry",
}
"""Prompting strings for the answer relevance rewriter. This model is a (a)LoRA adapter,
so it's important to stick to in-domain prompts."""


def check_answerability(
    question: str,
    documents: collections.abc.Iterable[Document],
    context: ChatContext,
    backend: AdapterMixin,
) -> float:
    """Test a user's question for answerability.

    Intrinsic function that checks whether the question in the last user turn of a
    chat can be answered by a provided set of RAG documents.

    :param context: Chat context containing the conversation thus far
    :param question: Question that the user has posed in response to the last turn in
        ``context``.
    :param documents: Document snippets retrieved that may or may not answer the
        indicated question.
    :param backend: Backend instance that supports adding the LoRA or aLoRA adapters
        for answerability checks

    :return: Answerability score as a floating-point value from 0 to 1.
    """
    result_json = _call_intrinsic(
        "answerability",
        context.add(Message("user", question, documents=list(documents))),
        backend,
    )
    return result_json["answerability_likelihood"]


def rewrite_question(
    question: str, context: ChatContext, backend: AdapterMixin
) -> float:
    """Rewrite a user's question for retrieval.

    Intrinsic function that rewrites the question in the next user turn into a
    self-contained query that can be passed to the retriever.

    :param context: Chat context containing the conversation thus far
    :param question: Question that the user has posed in response to the last turn in
        ``context``.
    :param backend: Backend instance that supports adding the LoRA or aLoRA adapters

    :return: Rewritten version of ``question``.
    """
    result_json = _call_intrinsic(
        "query_rewrite", context.add(Message("user", question)), backend
    )
    return result_json["rewritten_question"]


def clarify_query(
    question: str,
    documents: collections.abc.Iterable[Document],
    context: ChatContext,
    backend: AdapterMixin,
) -> str:
    """Generate clarification for an ambiguous query.

    Intrinsic function that determines if a user's question requires clarification
    based on the retrieved documents and conversation context, and generates an
    appropriate clarification question if needed.

    :param context: Chat context containing the conversation thus far
    :param question: Question that the user has posed
    :param documents: Document snippets retrieved for the question
    :param backend: Backend instance that supports the adapters that implement
        this intrinsic

    :return: Clarification question string (e.g., "Do you mean A or B?"), or
        the string "CLEAR" if no clarification is needed
    """
    result_json = _call_intrinsic(
        "query_clarification",
        context.add(Message("user", question, documents=list(documents))),
        backend,
    )
    return result_json["clarification"]


def find_citations(
    response: str,
    documents: collections.abc.Iterable[Document],
    context: ChatContext,
    backend: AdapterMixin,
) -> list[dict]:
    """Find information in documents that supports an assistant response.

    Intrinsic function that finds sentences in RAG documents that support sentences
    in a potential assistant response to a user question.

    :param context: Context of the dialog between user and assistant at the point where
        the user has just asked a question that will be answered with RAG documents
    :param response: Potential assistant response
    :param documents: Documents at were used to generate ``response``. These documents
        should set the ``doc_id`` field; otherwise the intrinsic will be unable to
        specify which document was the source of a given citation.
    :param backend: Backend that supports one of the adapters that implements this
        intrinsic.
    :return: List of records with the following fields:
        * ``response_begin``
        * ``response_end``
        * ``response_text``
        * ``citation_doc_id``
        * ``citation_begin``
        * ``citation_end``
        * ``citation_text``
    Begin and end offsets are character offsets into their respective UTF-8 strings.
    """
    result_json = _call_intrinsic(
        "citations",
        context.add(Message("assistant", response, documents=list(documents))),
        backend,
    )
    return result_json


def check_context_relevance(
    question: str, document: Document, context: ChatContext, backend: AdapterMixin
) -> float:
    """Test whether a document is relevant to a user's question.

    Intrinsic function that checks whether a single document contains part or all of
    the answer to a user's question. Does not consider the context in which the
    question was asked.

    :param context: The chat up to the point where the user asked a question.
    :param question: Question that the user has posed.
    :param document: A retrieved document snippet
    :param backend: Backend instance that supports the adapters that implement this
        intrinsic

    :return: Context relevance score as a floating-point value from 0 to 1.
    """
    result_json = _call_intrinsic(
        "context_relevance",
        context.add(Message("user", question)),
        backend,
        # Target document is passed as an argument
        kwargs={"document_content": document.text},
    )
    return result_json["context_relevance"]


def flag_hallucinated_content(
    response: str,
    documents: collections.abc.Iterable[Document],
    context: ChatContext,
    backend: AdapterMixin,
) -> float:
    """Flag potentially-hallucinated sentences in an agent's response.

    Intrinsic function that checks whether the sentences in an agent's response to a
    user question are faithful to the retrieved document snippets. Sentences that do not
    align with the retrieved snippets are flagged as potential hallucinations.

    :param context: A chat log that ends with a user asking a question
    :param response: The assistant's response to the user's question in the last turn
        of ``context``
    :param documents: Document snippets that were used to generate ``response``
    :param backend: Backend instance that supports the adapters that implement this
        intrinsic

    :return: List of records with the following fields:
        * response_begin
        * response_end
        * response_text
        * faithfulness_likelihood
        * explanation
    """
    result_json = _call_intrinsic(
        "hallucination_detection",
        context.add(Message("assistant", response, documents=list(documents))),
        backend,
    )
    return result_json


def rewrite_answer_for_relevance(
    response: str,
    documents: collections.abc.Iterable[Document],
    context: ChatContext,
    backend: AdapterMixin,
    /,
    rewrite_threshold: float = 0.5,
) -> str:
    """Rewrite an assistant answer to improve relevance to the user's question.

    :param context: A chat log that ends with a user asking a question
    :param response: The assistant's response to the user's question in the last turn
        of ``context``
    :param documents: Document snippets that were used to generate ``response``
    :param backend: Backend instance that supports the adapters that implement this
        intrinsic
    :param rewrite_threshold: Number between 0.0 and 1.0 that determines how eagerly
        to skip rewriting the assistant's answer for relevance. 0.0 means never rewrite
        and 1.0 means always rewrite.

    :returns: Either the original response, or a rewritten version of the original
        response.
    """
    # First run the classifier to determine the likelihood of a relevant answer
    # Output will have three fields:
    # * answer_relevance_analysis
    # * answer_relevance_category
    # * answer_relevance_likelihood
    result_json = _call_intrinsic(
        "answer_relevance_classifier",
        context.add(Message("assistant", response, documents=list(documents))),
        backend,
    )
    if result_json["answer_relevance_likelihood"] >= rewrite_threshold:
        return response

    # If we get here, the classifier indicated a likely irrelevant response. Trigger
    # rewrite.
    # Rewrite needs a prompt string that is an expanded version of the classifier's
    # short output.
    correction_method = _ANSWER_RELEVANCE_CORRECTION_METHODS[
        result_json["answer_relevance_category"]
    ]

    result_json = _call_intrinsic(
        "answer_relevance_rewriter",
        context.add(Message("assistant", response, documents=list(documents))),
        backend,
        kwargs={
            "answer_relevance_category": result_json["answer_relevance_category"],
            "answer_relevance_analysis": result_json["answer_relevance_analysis"],
            "correction_method": correction_method,
        },
    )
    # Unpack boxed string
    return result_json["answer_relevance_rewrite"]
