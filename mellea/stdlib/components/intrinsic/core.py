"""Adapter functions for core model capabilities."""

import collections.abc

from ....backends.adapters import AdapterMixin
from ...components import Document, Message
from ...context import ChatContext
from ..docs.document import _coerce_to_documents
from ._util import _resolve_response, call_intrinsic


def check_certainty(context: ChatContext, backend: AdapterMixin) -> float:
    """Estimate the model's certainty about its last response.

    Adapter function that evaluates how certain the model is about the
    assistant's response to a user's question. The context should end with
    a user question followed by an assistant answer.

    Args:
        context: Chat context containing user question and assistant answer.
        backend: Backend instance that supports LoRA/aLoRA adapters.

    Returns:
        Certainty score as a float (higher = more certain).
    """
    result_json = call_intrinsic("uncertainty", context, backend)
    return result_json["certainty"]


def requirement_check(
    context: ChatContext, backend: AdapterMixin, requirement: str
) -> float:
    """Detect if text adheres to provided requirements.

    Adapter function that determines if the text satisfies the given
    requirements. The requirement text is passed through to the adapter's
    `io.yaml` `instruction` template via `IntrinsicsRewriter`, which
    appends the formatted evaluation prompt as a new user message.

    Args:
        context: Chat context containing user question and assistant answer.
        backend: Backend instance that supports LoRA/aLoRA adapters.
        requirement: Set of requirements to satisfy.

    Returns:
        Score as a float between 0.0 and 1.0 (higher = more likely satisfied).
    """
    result_json = call_intrinsic(
        "requirement-check", context, backend, kwargs={"requirement": requirement}
    )
    return result_json["requirement_check"]["score"]


def find_context_attributions(
    response: str | None,
    documents: collections.abc.Iterable[str | Document],
    context: ChatContext,
    backend: AdapterMixin,
) -> list[dict]:
    """Find sentences in conversation history and documents that most influence an LLM's response.

    Adapter function that finds sentences in prior conversation messages and RAG
    documents that were most important to the LLM in generating each sentence in the
    assistant response.

    :param response: Assistant response. When `None`, the response is extracted
        from the last assistant output in `context`.
    :param documents: Documents that were used to generate `response`. Each element
        may be a `Document` or a plain string. Strings are wrapped in `Document`
        with an auto-generated `doc_id` (`"0"`, `"1"`, ...); for explicit
        control, pass `Document` objects with `doc_id` set. `Document` objects
        without `doc_id` trigger a warning because the intrinsic uses `doc_id` to
        identify attribution sources.
    :param context: Context of the dialog between user and assistant, ending with a
        user query
    :param backend: Backend that supports intrinsic adapters

    :return: List of records with the following fields:
        * `response_begin`
        * `response_end`
        * `response_text`
        * `attribution_doc_id`
        * `attribution_msg_index`
        * `attribution_begin`
        * `attribution_end`
        * `attribution_text`
    Begin and end offsets are character offsets into their respective UTF-8 strings.
    """
    response, context = _resolve_response(response, context)
    result_json = call_intrinsic(
        "context-attribution",
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
