"""Intrinsic functions related to retrieval-augmented generation."""

import collections.abc

from ....backends import ModelOption
from ....backends.adapters import AdapterMixin, AdapterType, IntrinsicAdapter
from ....core import FancyLogger
from ....stdlib import functional as mfuncs
from ...components import Document
from ...context import ChatContext
from ..chat import Message
from ._util import call_intrinsic


def check_answerability(
    question: str,
    documents: collections.abc.Iterable[Document],
    context: ChatContext,
    backend: AdapterMixin,
) -> float:
    """Test a user's question for answerability.

    Intrinsic function that checks whether the question in the last user turn of a
    chat can be answered by a provided set of RAG documents.

    Args:
        question: Question that the user has posed in response to the last turn in
            ``context``.
        documents: Document snippets retrieved that may or may not answer the
            indicated question.
        context: Chat context containing the conversation thus far.
        backend: Backend instance that supports adding the LoRA or aLoRA adapters
            for answerability checks.

    Returns:
        Answerability score as a floating-point value from 0 to 1.
    """
    result_json = call_intrinsic(
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

    Args:
        question: Question that the user has posed in response to the last turn in
            ``context``.
        context: Chat context containing the conversation thus far.
        backend: Backend instance that supports adding the LoRA or aLoRA adapters.

    Returns:
        Rewritten version of ``question``.
    """
    result_json = call_intrinsic(
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

    Args:
        question: Question that the user has posed.
        documents: Document snippets retrieved for the question.
        context: Chat context containing the conversation thus far.
        backend: Backend instance that supports the adapters that implement
            this intrinsic.

    Returns:
        Clarification question string (e.g., "Do you mean A or B?"), or
        the string "CLEAR" if no clarification is needed.
    """
    result_json = call_intrinsic(
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

    Args:
        response: Potential assistant response.
        documents: Documents that were used to generate ``response``. These documents
            should set the ``doc_id`` field; otherwise the intrinsic will be unable to
            specify which document was the source of a given citation.
        context: Context of the dialog between user and assistant at the point where
            the user has just asked a question that will be answered with RAG documents.
        backend: Backend that supports one of the adapters that implements this
            intrinsic.

    Returns:
        List of records with the following fields: ``response_begin``,
        ``response_end``, ``response_text``, ``citation_doc_id``, ``citation_begin``,
        ``citation_end``, ``citation_text``. Begin and end offsets are character
        offsets into their respective UTF-8 strings.
    """
    result_json = call_intrinsic(
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

    Args:
        question: Question that the user has posed.
        document: A retrieved document snippet.
        context: The chat up to the point where the user asked a question.
        backend: Backend instance that supports the adapters that implement this
            intrinsic.

    Returns:
        Context relevance score as a floating-point value from 0 to 1.
    """
    result_json = call_intrinsic(
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
) -> list[dict]:
    """Flag potentially-hallucinated sentences in an agent's response.

    Intrinsic function that checks whether the sentences in an agent's response to a
    user question are faithful to the retrieved document snippets. Sentences that do not
    align with the retrieved snippets are flagged as potential hallucinations.

    Args:
        response: The assistant's response to the user's question in the last turn
            of ``context``.
        documents: Document snippets that were used to generate ``response``.
        context: A chat log that ends with a user asking a question.
        backend: Backend instance that supports the adapters that implement this
            intrinsic.

    Returns:
        List of records with the following fields: ``response_begin``,
        ``response_end``, ``response_text``, ``faithfulness_likelihood``,
        ``explanation``.
    """
    logger = FancyLogger.get_logger()

    # Log input
    doc_list = list(documents)
    logger.info(
        f"flag_hallucinated_content input: response_len={len(response)}, "
        f"num_documents={len(doc_list)}, context_len={len(context.as_list())}"
    )

    result_json = call_intrinsic(
        "hallucination_detection",
        context.add(Message("assistant", response, documents=doc_list)),
        backend,
    )

    # Log output
    num_segments = len(result_json) if isinstance(result_json, list) else 0
    if num_segments > 0:
        hallucinated_count = sum(
            1 for r in result_json if r.get("faithfulness_likelihood", 1.0) < 0.5
        )
        logger.info(
            f"flag_hallucinated_content output: total_segments={num_segments}, "
            f"potentially_hallucinated={hallucinated_count}"
        )
    else:
        logger.info("flag_hallucinated_content output: no segments detected")

    return result_json
