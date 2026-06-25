"""Adapter functions for core model capabilities."""

import collections.abc
from typing import cast
import math

from ....backends.adapters import AdapterMixin, AdapterSchemaMismatchError
from ...components import Document, Message
from ...context import ChatContext
from ..docs.document import _coerce_to_documents
from ._util import _resolve_response, call_intrinsic


def check_certainty(
    context: ChatContext, backend: AdapterMixin, model_options: dict | None = None
) -> float:
    """Estimate the model's certainty about its last response.

    Adapter function that evaluates how certain the model is about the
    assistant's response to a user's question. The context should end with
    a user question followed by an assistant answer.

    Args:
        context: Chat context containing user question and assistant answer.
        backend: Backend instance that supports LoRA/aLoRA adapters.
        model_options: Optional model-level overrides forwarded to the backend
            (e.g. ``{ModelOption.MAX_NEW_TOKENS: 64}``). When ``None``, defaults
            apply.

    Returns:
        Certainty score as a float (higher = more certain).
    """
    result_json = call_intrinsic(
        "uncertainty", context, backend, model_options=model_options
    )
    return result_json["certainty"]


def requirement_check(
    context: ChatContext,
    backend: AdapterMixin,
    requirement: str,
    model_options: dict | None = None,
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
        model_options: Optional model-level overrides forwarded to the backend
            (e.g. ``{ModelOption.MAX_NEW_TOKENS: 64}``). When ``None``, defaults
            apply.

    Returns:
        Score as a float between 0.0 and 1.0 (higher = more likely satisfied).

    Raises:
        AdapterSchemaMismatchError: If the adapter output does not match the
            expected ``{"requirement_check": {"score": <float>}}`` contract, or
            if the score is not a finite number in the range 0.0-1.0.
    """
    result_json = call_intrinsic(
        "requirement-check",
        context,
        backend,
        kwargs={"requirement": requirement},
        model_options=model_options,
    )
    # Mirrors the validation in requirement_check_to_bool() in requirement.py; Phase 2 will consolidate via IOContract.
    req_check = result_json.get("requirement_check")
    if not isinstance(req_check, dict):
        raise AdapterSchemaMismatchError(
            name="requirement-check",
            observed_keys=frozenset(result_json.keys()),
            expected_keys=frozenset({"requirement_check"}),
        )
    score = req_check.get("score")
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)  # bool subclasses int; exclude it explicitly
        or not math.isfinite(score)
        or not 0.0 <= score <= 1.0
    ):
        raise AdapterSchemaMismatchError(
            name="requirement-check",
            observed_keys=frozenset(req_check.keys()),
            expected_keys=frozenset({"score"}),
        )
    return score


def find_context_attributions(
    response: str | None,
    documents: collections.abc.Iterable[str | Document],
    context: ChatContext,
    backend: AdapterMixin,
    model_options: dict | None = None,
) -> list[dict]:
    """Find sentences in conversation history and documents that most influence an LLM's response.

    Adapter function that finds sentences in prior conversation messages and RAG
    documents that were most important to the LLM in generating each sentence in the
    assistant response.

    Args:
        response (str | None): Assistant response. When `None`, extracted from the
            last assistant output in `context`.
        documents (collections.abc.Iterable[str | Document]): Documents used to
            generate `response`. Each element may be a
            `Document` or a plain string. Strings are wrapped in `Document` with an
            auto-generated `doc_id` (`"0"`, `"1"`, ...); for explicit control, pass
            `Document` objects with `doc_id` set. `Document` objects without `doc_id`
            trigger a warning because the intrinsic uses `doc_id` to identify sources.
        context (ChatContext): Dialog context between user and assistant, ending with
            a user query.
        backend (AdapterMixin): Backend that supports intrinsic adapters.
        model_options: Optional model-level overrides forwarded to the backend
            (e.g. ``{ModelOption.MAX_NEW_TOKENS: 64}``). When ``None``, defaults
            apply.

    Returns:
        list[dict]: Records with fields `response_begin`, `response_end`,
            `response_text`, `attribution_doc_id`, `attribution_msg_index`,
            `attribution_begin`, `attribution_end`, and `attribution_text`.
            Begin and end offsets are character offsets into their respective
            UTF-8 strings.
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
        model_options=model_options,
    )
    return cast(list[dict], result_json)
