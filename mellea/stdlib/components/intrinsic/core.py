"""Intrinsic functions for core model capabilities."""

import collections.abc

from ....backends.adapters import AdapterMixin
from ...components import Document, Message
from ...context import ChatContext
from ._util import call_intrinsic


def check_certainty(context: ChatContext, backend: AdapterMixin) -> float:
    """Estimate the model's certainty about its last response.

    Intrinsic function that evaluates how certain the model is about the
    assistant's response to a user's question. The context should end with
    a user question followed by an assistant answer.

    :param context: Chat context containing user question and assistant answer.
    :param backend: Backend instance that supports LoRA/aLoRA adapters.

    :return: Certainty score as a float (higher = more certain).
    """
    result_json = call_intrinsic("uncertainty", context, backend)
    return result_json["certainty"]


_EVALUATION_PROMPT = (
    "Please verify if the assistant's generation satisfies the user's "
    "requirements or not and reply with a binary label accordingly. "
    'Respond with a json {"score": "yes"} if the constraints are '
    'satisfied or respond with {"score": "no"} if the constraints are not '
    "satisfied."
)


def requirement_check(
    context: ChatContext, backend: AdapterMixin, requirement: str
) -> float:
    """Detect if text adheres to provided requirements.

    Intrinsic function that determines if the text satisfies the given
    requirements. Appends an evaluation prompt to the context following
    the format specified by the Granite Guardian requirement checker model card.

    :param context: Chat context containing user question and assistant answer.
    :param backend: Backend instance that supports LoRA/aLoRA adapters.
    :param requirement: set of requirements to satisfy

    :return: Score as a float between 0.0 and 1.0 (higher = more likely satisfied).
    """
    eval_message = f"<requirements>: {requirement}\n{_EVALUATION_PROMPT}"
    context = context.add(Message("user", eval_message))
    result_json = call_intrinsic("requirement_check", context, backend)
    return result_json["requirement_check"]["score"]


def find_context_attributions(
    response: str,
    documents: collections.abc.Iterable[Document],
    context: ChatContext,
    backend: AdapterMixin,
) -> list[dict]:
    """Find sentences in conversation history and documents that most influence an LLM's response.

    Intrinsic function that finds sentences in prior conversation messages and RAG
    documents that were most important to the LLM in generating each sentence in the
    assistant response.

    :param response: Assistant response
    :param documents: Documents that were used to generate ``response``
    :param context: Context of the dialog between user and assistant, ending with a
        user query
    :param backend: Backend that supports intrinsic adapters

    :return: List of records with the following fields:
        * ``response_begin``
        * ``response_end``
        * ``response_text``
        * ``attribution_doc_id``
        * ``attribution_msg_index``
        * ``attribution_begin``
        * ``attribution_end``
        * ``attribution_text``
    Begin and end offsets are character offsets into their respective UTF-8 strings.
    """
    result_json = call_intrinsic(
        "context-attribution",
        context.add(Message("assistant", response, documents=list(documents))),
        backend,
    )
    return result_json
