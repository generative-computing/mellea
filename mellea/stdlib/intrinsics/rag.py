"""Intrinsic functions related to retrieval-augmented generation."""

import collections.abc
import json

import granite_common

import mellea.stdlib.funcs
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
    "answerability": (RAG_REPO, AdapterType.LORA),
    "query_rewrite": (RAG_REPO, AdapterType.LORA),
    "citations": (RAG_REPO, AdapterType.LORA),
    "context_relevance": (RAG_REPO, AdapterType.LORA),
}
ANSWERABILITY_MODEL_NAME = "answerability"


def _call_intrinsic(
    intrinsic_name: str, context: ChatContext, backend, /, kwargs: dict | None = None
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
    base_model_name = backend.model_id
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
    model_output_thunk, _ = mellea.stdlib.funcs.act(
        intrinsic,
        context,
        backend,
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
    backend,  # Can't put type hints here because linter complains
) -> float:
    """Test a user's question for answerability.

    Intrinsic function that checks whether the question in the last user turn of a
    chat can be answered by a provided set of RAG documents.

    :param context: Chat context containing the conversation thus far
    :param question: Question that the user has posed in response to the last turn in
        ``context``.
    :param documents: A set of documents retrieved that may or may not answer the
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
    context: ChatContext,
    question: str,
    backend,  # Can't put type hints here because linter complains
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
    backend,  # Can't put type hints here because linter complains
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
    context: ChatContext,
    question: str,
    document: Document,
    backend,  # Can't put type hints here because linter complains
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
