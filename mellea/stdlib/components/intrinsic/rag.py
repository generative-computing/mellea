"""Intrinsic functions related to retrieval-augmented generation."""

import collections.abc

from ....backends.adapters import AdapterMixin
from ...components import Document
from ...context import ChatContext
from ..chat import Message
from ..docs.document import _coerce_to_document, _coerce_to_documents
from ._util import _resolve_question, _resolve_response, call_intrinsic


def check_answerability(
    question: str | None,
    documents: collections.abc.Iterable[str | Document],
    context: ChatContext,
    backend: AdapterMixin,
) -> str:
    """Test a user's question for answerability.

    Intrinsic function that checks whether the question in the last user turn of a
    chat can be answered by a provided set of RAG documents.

    Args:
        question: Question that the user has posed in response to the last turn in
            ``context``. When ``None``, the question is extracted from the last
            user message in ``context``.
        documents: Document snippets retrieved that may or may not answer the
            indicated question. Each element may be a ``Document`` or a plain
            string (automatically wrapped in ``Document``).
        context: Chat context containing the conversation thus far.
        backend: Backend instance that supports adding the LoRA or aLoRA adapters
            for answerability checks.

    Returns:
        A string value of either "answerable" or "unanswerable"
    """
    question, context = _resolve_question(question, context, backend)
    result_json = call_intrinsic(
        "answerability",
        context.add(
            Message("user", question, documents=_coerce_to_documents(documents))
        ),
        backend,
    )
    return result_json["answerability"]


def rewrite_question(
    question: str | None, context: ChatContext, backend: AdapterMixin
) -> str:
    """Rewrite a user's question for retrieval.

    Intrinsic function that rewrites the question in the next user turn into a
    self-contained query that can be passed to the retriever.

    Args:
        question: Question that the user has posed in response to the last turn in
            ``context``. When ``None``, the question is extracted from the last
            user message in ``context``.
        context: Chat context containing the conversation thus far.
        backend: Backend instance that supports adding the LoRA or aLoRA adapters.

    Returns:
        Rewritten version of ``question``.
    """
    question, context = _resolve_question(question, context, backend)
    result_json = call_intrinsic(
        "query_rewrite", context.add(Message("user", question)), backend
    )
    return result_json["rewritten_question"]


def clarify_query(
    question: str | None,
    documents: collections.abc.Iterable[str | Document],
    context: ChatContext,
    backend: AdapterMixin,
) -> str:
    """Generate clarification for an ambiguous query.

    Intrinsic function that determines if a user's question requires clarification
    based on the retrieved documents and conversation context, and generates an
    appropriate clarification question if needed.

    Args:
        question: Question that the user has posed. When ``None``, the question
            is extracted from the last user message in ``context``.
        documents: Document snippets retrieved for the question. Each element
            may be a ``Document`` or a plain string (automatically wrapped in
            ``Document``).
        context: Chat context containing the conversation thus far.
        backend: Backend instance that supports the adapters that implement
            this intrinsic.

    Returns:
        Clarification question string (e.g., "Do you mean A or B?"), or
        the string "CLEAR" if no clarification is needed.
    """
    question, context = _resolve_question(question, context, backend)
    result_json = call_intrinsic(
        "query_clarification",
        context.add(
            Message("user", question, documents=_coerce_to_documents(documents))
        ),
        backend,
    )
    return result_json["clarification"]


def find_citations(
    response: str | None,
    documents: collections.abc.Iterable[str | Document],
    context: ChatContext,
    backend: AdapterMixin,
) -> list[dict]:
    """Find information in documents that supports an assistant response.

    Intrinsic function that finds sentences in RAG documents that support sentences
    in a potential assistant response to a user question.

    Args:
        response: Potential assistant response. When ``None``, the response is
            extracted from the last assistant output in ``context``.
        documents: Documents that were used to generate ``response``. Each element
            may be a ``Document`` or a plain string. Strings are wrapped in
            ``Document`` with an auto-generated ``doc_id`` (``"0"``, ``"1"``, ...);
            for explicit control, pass ``Document`` objects with ``doc_id`` set.
            ``Document`` objects without ``doc_id`` trigger a warning because the
            intrinsic uses ``doc_id`` to identify citation sources.
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
    response, context = _resolve_response(response, context)
    result_json = call_intrinsic(
        "citations",
        context.add(
            Message(
                "assistant",
                response,
                documents=_coerce_to_documents(documents, auto_doc_id=False),
            )
        ),
        backend,
    )
    return result_json


def check_context_relevance(
    question: str | None,
    document: str | Document,
    context: ChatContext,
    backend: AdapterMixin,
) -> str:
    """Test whether a document is relevant to a user's question.

    Intrinsic function that checks whether a single document contains part or all of
    the answer to a user's question. Does not consider the context in which the
    question was asked.

    Args:
        question: Question that the user has posed. When ``None``, the question
            is extracted from the last user message in ``context``.
        document: A retrieved document snippet. May be a ``Document`` or a plain
            string (automatically wrapped in ``Document``).
        context: The chat up to the point where the user asked a question.
        backend: Backend instance that supports the adapters that implement this
            intrinsic.

    Returns:
        Context relevance judgement as one of the following strings:
        - "relevant"
        - "irrelevant"
        - "partially relevant"
    """
    question, context = _resolve_question(question, context, backend)
    document = _coerce_to_document(document)
    result_json = call_intrinsic(
        "context_relevance",
        context.add(Message("user", question)),
        backend,
        # Target document is passed as an argument
        kwargs={"document_content": document.text},
    )
    return result_json["context_relevance"]


def flag_hallucinated_content(
    response: str | None,
    documents: collections.abc.Iterable[str | Document],
    context: ChatContext,
    backend: AdapterMixin,
) -> list[dict]:
    """Flag potentially-hallucinated sentences in an agent's response.

    Intrinsic function that checks whether the sentences in an agent's response to a
    user question are faithful to the retrieved document snippets. Sentences that do not
    align with the retrieved snippets are flagged as potential hallucinations.

    Args:
        response: The assistant's response to the user's question in the last turn
            of ``context``. When ``None``, the response is extracted from the last
            assistant output in ``context``.
        documents: Document snippets that were used to generate ``response``. Each
            element may be a ``Document`` or a plain string (automatically wrapped
            in ``Document``).
        context: A chat log that ends with a user asking a question.
        backend: Backend instance that supports the adapters that implement this
            intrinsic.

    Returns:
        List of records with the following fields: ``response_begin``,
        ``response_end``, ``response_text``, ``faithfulness``,
        ``explanation``.
    """
    response, context = _resolve_response(response, context)
    result_json = call_intrinsic(
        "hallucination_detection",
        context.add(
            Message("assistant", response, documents=_coerce_to_documents(documents))
        ),
        backend,
    )
    return result_json
