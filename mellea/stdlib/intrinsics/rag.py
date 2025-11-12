"""Intrinsic functions related to retrieval-augmented generation."""

import collections.abc
import json

import granite_common

import mellea.stdlib.functional as mfuncs
from mellea.backends import Backend
from mellea.backends.adapters.adapter import (
    AdapterMixin,
    AdapterType,
    GraniteCommonAdapter,
)
from mellea.stdlib.base import ChatContext, Document
from mellea.stdlib.chat import Message
from mellea.stdlib.intrinsics.intrinsic import Intrinsic

RAG_REPO = "ibm-granite/rag-intrinsics-lib"
# RAG_REPO = "generative-computing/rag-intrinsics-lib"

# Mapping from name to <repository, adapter_type>.
_CATALOG = {
    "answer_relevance_classifier": (RAG_REPO, AdapterType.LORA),
    "answer_relevance_rewriter": (RAG_REPO, AdapterType.LORA),
    "answerability": (RAG_REPO, AdapterType.LORA),
    "citations": (RAG_REPO, AdapterType.LORA),
    "context_relevance": (RAG_REPO, AdapterType.LORA),
    "hallucination_detection": (RAG_REPO, AdapterType.LORA),
    "query_rewrite": (RAG_REPO, AdapterType.LORA),
}
ANSWERABILITY_MODEL_NAME = "answerability"

_ANSWER_RELEVANCE_CORRECTION_METHODS = {
    "Excessive unnecessary information": "removing the excessive information from the draft response",
    "Unduly restrictive": "providing answer without the unwarranted restriction, or indicating that the desired answer is not available",
    "Too vague or generic": "providing more crisp and to-the-point answer, or indicating that the desired answer is not available",
    "Contextual misalignment": "providing a response that answers the last user inquiry, taking into account the context of the conversation",
    "Misinterpreted inquiry": "providing answer only to the correct interpretation of the inquiry, or attempting clarification if the inquiry is ambiguous or otherwise confusing, or indicating that the desired answer is not available",
    "No attempt": "providing a relevant response if an inquiry should be answered, or providing a short response if the last user utterance contains no inquiry",
}
"""Prompting strings for the answer relevance rewriter. This model is a (a)LoRA adapter,
so it's important to stick to in-domain prompts."""


def _call_intrinsic(
    intrinsic_name: str,
    context: ChatContext,
    backend: AdapterMixin,  # type: ignore
    /,
    kwargs: dict | None = None,
):
    """Shared code for invoking intrinsics.

    :returns: Result of the call in JSON format.
    """
    # Adapter needs to be present in the backend before it can be invoked.
    # We must create the Adapter object in order to determine whether we need to create
    # the Adapter object.
    if intrinsic_name not in _CATALOG:
        raise ValueError(
            f"Unexpected intrinsic name '{intrinsic_name}'. "
            f"Should be one of {list(_CATALOG.keys())}"
        )
    repo_id, adapter_type = _CATALOG[intrinsic_name]
    base_model_name = backend.model_id  # type: ignore
    if base_model_name is None:
        raise ValueError("Backend has no model ID")
    adapter = GraniteCommonAdapter(
        repo_id, intrinsic_name, adapter_type, base_model_name=base_model_name
    )
    if adapter.qualified_name not in backend.list_adapters():
        backend.add_adapter(adapter)

    # Create the AST node for the action we wish to perform.
    intrinsic = Intrinsic(
        repo_id, intrinsic_name, adapter_types=[adapter_type], intrinsic_kwargs=kwargs
    )

    # Execute the AST node.
    model_output_thunk, _ = mfuncs.act(
        intrinsic,
        context,
        backend,  # type: ignore
        # No rejection sampling, please
        strategy=None,
    )

    # act() can return a future. Don't know how to handle one from non-async code.
    assert model_output_thunk.is_computed()

    # Output of an Intrinsic action is the string representation of the output of the
    # intrinsic. Parse the string.
    result_str = model_output_thunk.value
    if result_str is None:
        raise ValueError("Model output is None.")
    result_json = json.loads(result_str)
    return result_json


def check_answerability(
    context: ChatContext,
    question: str,
    documents: collections.abc.Iterable[Document],
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
    context: ChatContext, question: str, backend: AdapterMixin
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


def find_citations(
    context: ChatContext,
    response: str,
    documents: collections.abc.Iterable[Document],
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
    context: ChatContext, question: str, document: Document, backend: AdapterMixin
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
    context: ChatContext,
    response: str,
    documents: collections.abc.Iterable[Document],
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
    context,
    response: str,
    documents: collections.abc.Iterable[Document],
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
            "answer_relevance_analysis": result_json["answer_relevance_category"],
            "correction_method": correction_method,
        },
    )
    # Unpack boxed string
    return result_json["answer_relevance_rewrite"]
