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

# Temporary value
RAG_REPO = "ibm-granite/rag-intrinsics-lib"
# RAG_REPO = "generative-computing/rag-intrinsics-lib"

# Mapping from name to repository, adapter_type
_CATALOG = {"answerability": (RAG_REPO, AdapterType.LORA)}
ANSWERABILITY_MODEL_NAME = "answerability"


def check_answerability(
    context: ChatContext,
    question: str,
    docs: collections.abc.Iterable[Document],
    backend,  # Can't put type annotations here because linter complains
) -> float:
    """Test a user's question for answerability.

    Intrinsic function that checks whether the question in the last user turn of a
    chat can be answered by a provided set of RAG documents.

    :param context: Chat context containing the conversation thus far
    :param question: Question that the user has posed in response to the last turn in
        ``context``.
    :param docs: A set of documents retrieved that may or may not answer the indicated
        question.
    :param backend: Backend instance that supports adding the LoRA or aLoRA adapters
        for answerability checks

    :return: Answerability score as a floating-point value from 0 to 1.
    """
    # Adapter needs to be present in the backend before it can be invoked.
    # The current APIs require us to create the Adapter object in order to determine
    # whether we need to create the Adapter object.
    intrinsic_name = "answerability"
    repo_id, adapter_type = _CATALOG[intrinsic_name]
    base_model_name = backend.model_id
    if base_model_name is None:
        raise ValueError("Backend has no model ID")
    adapter = GraniteCommonAdapter(
        repo_id, intrinsic_name, adapter_type, base_model_name=base_model_name
    )
    if adapter.qualified_name not in backend.list_adapters():
        backend.add_adapter(adapter)

    # Append the user's question and the documents to the existing conversation.
    # This operation creates a copy of the immutable context.
    context = context.add(Message("user", question, documents=list(docs)))

    # Create the AST node for the action we wish to perform.
    intrinsic = Intrinsic(repo_id, intrinsic_name, adapter_types=[adapter_type])

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
    if (
        not isinstance(result_json, dict)
        or "answerability_likelihood" not in result_json
    ):
        raise ValueError(
            f"Unexpected low-level result format from {intrinsic_name} "
            f"adapter: '{result_str}'"
        )
    return result_json["answerability_likelihood"]
