# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter functions for core model capabilities."""

import collections.abc
from typing import cast

from ....backends.adapters import (
    Adapter,
    AdapterMixin,
    Identity,
    LocalFileBinding,
    get_io_contract,
)
from ...components import Document, Message
from ...context import ChatContext
from ..docs.document import _coerce_to_documents
from ._util import _resolve_response, call_intrinsic

# ---------------------------------------------------------------------------
# _REQUIREMENT_CHECK_ADAPTER: a test-visible handle onto the `requirement-check`
# capability's canonical Adapter, used by test_core_schema.py to stub
# `resolve_adapter`'s return value. `check_certainty` and `find_context_attributions`
# have no equivalent constant — nothing outside a test needs one, since their
# io_contract is looked up by resolve_adapter() at call time
# (mellea.backends.adapters.io_contracts), never declared here. Declaring one
# unused by any caller is exactly the parallel-declaration problem issue #1516
# closes.
# ---------------------------------------------------------------------------

_REQUIREMENT_CHECK_ADAPTER = Adapter(
    identity=Identity("requirement-check", "alora", capability="requirement_check"),
    io_contract=get_io_contract("requirement-check"),
    weights=LocalFileBinding(),
)


def check_certainty(
    context: ChatContext, backend: AdapterMixin, model_options: dict | None = None
) -> float:
    """Estimate the model's certainty about its last response.

    Adapter function that evaluates how certain the model is about the
    assistant's response to a user's question. The context should end with
    a user question followed by an assistant answer.

    Output contract — required key: `certainty`.  Missing the key raises
    `AdapterSchemaMismatchError`; extra optional keys do not raise (forward-compatible).

    Args:
        context: Chat context containing user question and assistant answer.
        backend: Backend instance that supports LoRA/aLoRA adapters.
        model_options: Optional model-level overrides forwarded to the backend
            (e.g. `{ModelOption.MAX_NEW_TOKENS: 64}`). When `None`, defaults
            apply.

    Returns:
        Certainty score as a float (higher = more certain).

    Raises:
        ValueError: When the model output is not valid JSON.
        AdapterSchemaMismatchError: When the model output is missing the required
            `certainty` field.
    """
    result_json = call_intrinsic(
        "uncertainty", context, backend, model_options=model_options
    )
    return cast(float, result_json["certainty"])


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

    Output contract — required shape: `{"requirement_check": {"score": <float>}}`,
    with `score` a finite number (not a `bool`) in the closed range `[0.0, 1.0]`.
    Any deviation raises `AdapterSchemaMismatchError`.

    Args:
        context: Chat context containing user question and assistant answer.
        backend: Backend instance that supports LoRA/aLoRA adapters.
        requirement: Set of requirements to satisfy.
        model_options: Optional model-level overrides forwarded to the backend
            (e.g. `{ModelOption.MAX_NEW_TOKENS: 64}`). When `None`, defaults
            apply.

    Returns:
        Score as a float between 0.0 and 1.0 (higher = more likely satisfied).

    Raises:
        ValueError: When the model output is not valid JSON.
        AdapterSchemaMismatchError: If the adapter output does not match the
            expected `{"requirement_check": {"score": <float>}}` contract, or
            if the score is not a finite number in the range 0.0-1.0.
    """
    result_json = call_intrinsic(
        "requirement-check",
        context,
        backend,
        kwargs={"requirement": requirement},
        model_options=model_options,
    )
    return cast(
        float, cast(dict[str, object], result_json["requirement_check"])["score"]
    )


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

    Output contract — each record must contain: `response_begin`, `response_end`,
    `response_text`, `attribution_doc_id`, `attribution_msg_index`,
    `attribution_begin`, `attribution_end`, `attribution_text`.  A record missing
    any of these keys raises `AdapterSchemaMismatchError`; extra optional keys do
    not raise (forward-compatible).

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
            (e.g. `{ModelOption.MAX_NEW_TOKENS: 64}`). When `None`, defaults
            apply.

    Returns:
        list[dict]: Records with fields `response_begin`, `response_end`,
            `response_text`, `attribution_doc_id`, `attribution_msg_index`,
            `attribution_begin`, `attribution_end`, and `attribution_text`.
            Begin and end offsets are character offsets into their respective
            UTF-8 strings.

    Raises:
        ValueError: When the model output is not valid JSON.
        AdapterSchemaMismatchError: When any record in the output is missing a
            required field.
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
    return cast(list[dict], result_json["items"])
