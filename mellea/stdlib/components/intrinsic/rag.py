# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter functions related to retrieval-augmented generation."""

import collections.abc
import json
import warnings
from typing import cast

from ....backends.adapters import (
    Adapter,
    AdapterMixin,
    AdapterSchemaMismatchError,
    Identity,
    IOContract,
    LocalFileBinding,
)
from ....backends.adapters._core import _DictContract
from ....core import Component
from ...components import Document
from ...context import ChatContext
from ..chat import Message
from ..docs.document import _coerce_to_document, _coerce_to_documents
from ._util import _resolve_question, _resolve_response, call_intrinsic

# ---------------------------------------------------------------------------
# IOContract implementations
# ---------------------------------------------------------------------------


class _ListContract(IOContract):
    """Validate list-of-dicts adapter output and wrap it under key ``"items"``.

    Each item in the list is checked for the declared required keys.  The
    validated list is returned wrapped in ``{"items": [...]}`` so that
    :func:`call_intrinsic` can always return a plain ``dict``.

    Args:
        name: Adapter capability name; included in
            :class:`~mellea.backends.adapters.AdapterSchemaMismatchError` messages.
        required_item_keys: Keys that must be present in every item dict.
    """

    def __init__(self, name: str, required_item_keys: frozenset[str]) -> None:
        self._name = name
        self._required_item_keys = required_item_keys

    def build_prompt(self, **_kwargs: object) -> Component:
        raise NotImplementedError(
            "build_prompt is not used in Phase 1; implemented in Phase 2."
        )

    def parse(self, raw: str) -> dict[str, object]:
        """Parse and validate a list-of-dicts adapter output.

        Args:
            raw (str): Raw JSON string from the model.

        Returns:
            dict[str, object]: ``{"items": [list of validated dicts]}``.
                An empty list parses to ``{"items": []}``.

        Raises:
            ValueError: When *raw* is not valid JSON, is not a JSON array, or
                contains a non-object element.
            AdapterSchemaMismatchError: When any item is missing a required key.
        """
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(
                f"Adapter '{self._name}' output must be a JSON array, "
                f"got {type(data).__name__}."
            )
        for item in data:
            if not isinstance(item, dict):
                raise ValueError(
                    f"Adapter '{self._name}' output array must contain only JSON "
                    f"objects, got a {type(item).__name__} element."
                )
            observed = frozenset(item.keys())
            missing = self._required_item_keys - observed
            if missing:
                raise AdapterSchemaMismatchError(
                    self._name, observed, self._required_item_keys
                )
        return {"items": data}


# ---------------------------------------------------------------------------
# Module-level Adapter constants (one per helper)
# ---------------------------------------------------------------------------

_ANSWERABILITY_ADAPTER = Adapter(
    identity=Identity("answerability", "alora", capability="answerability"),
    io_contract=_DictContract("answerability", frozenset({"answerability"})),
    weights=LocalFileBinding(),
)

_QUERY_REWRITE_ADAPTER = Adapter(
    identity=Identity("query_rewrite", "alora", capability="query_rewrite"),
    io_contract=_DictContract("query_rewrite", frozenset({"rewritten_question"})),
    weights=LocalFileBinding(),
)

_QUERY_CLARIFY_ADAPTER = Adapter(
    identity=Identity("query_clarification", "alora", capability="query_clarification"),
    io_contract=_DictContract("query_clarification", frozenset({"clarification"})),
    weights=LocalFileBinding(),
)

_CITATIONS_ADAPTER = Adapter(
    identity=Identity("citations", "alora", capability="citations"),
    io_contract=_ListContract(
        "citations",
        frozenset(
            {
                "response_begin",
                "response_end",
                "response_text",
                "citation_doc_id",
                "citation_begin",
                "citation_end",
                "citation_text",
            }
        ),
    ),
    weights=LocalFileBinding(),
)

_CONTEXT_RELEVANCE_ADAPTER = Adapter(
    identity=Identity("context_relevance", "alora", capability="context_relevance"),
    io_contract=_DictContract("context_relevance", frozenset({"context_relevance"})),
    weights=LocalFileBinding(),
)

_HALLUCINATION_ADAPTER = Adapter(
    identity=Identity(
        "hallucination_detection", "alora", capability="hallucination_detection"
    ),
    io_contract=_ListContract(
        "hallucination_detection",
        frozenset(
            {
                "response_begin",
                "response_end",
                "response_text",
                "faithfulness",
                "explanation",
            }
        ),
    ),
    weights=LocalFileBinding(),
)


# ---------------------------------------------------------------------------
# High-level helper functions
# ---------------------------------------------------------------------------


def check_answerability(
    question: str | None,
    documents: collections.abc.Iterable[str | Document],
    context: ChatContext,
    backend: AdapterMixin,
    *,
    model_options: dict | None = None,
) -> str:
    """Test a user's question for answerability.

    Adapter function that checks whether the question in the last user turn of a
    chat can be answered by a provided set of RAG documents.

    Output contract — required key: ``answerability``.  Missing the key raises
    :class:`~mellea.backends.adapters.AdapterSchemaMismatchError`; extra optional
    keys in the model output do not raise (forward-compatible).

    Args:
        question: Question that the user has posed in response to the last turn in
            `context`. When `None`, the question is extracted from the last
            user message in `context`.
        documents: Document snippets retrieved that may or may not answer the
            indicated question. Each element may be a `Document` or a plain
            string (automatically wrapped in `Document`).
        context: Chat context containing the conversation thus far.
        backend: Backend instance that supports adding the LoRA or aLoRA adapters
            for answerability checks.
        model_options: Optional model-option overrides (e.g.
            ``{"temperature": 0.1}``).  Merged on top of the adapter default
            (``temperature=0.0``).

    Returns:
        A string value of either ``"answerable"`` or ``"unanswerable"``.

    Raises:
        ValueError: When the model output is not valid JSON.
        AdapterSchemaMismatchError: When the model output is missing the required
            ``answerability`` field.
    """
    question, context = _resolve_question(question, context, backend)
    result = call_intrinsic(
        "answerability",
        context.add(
            Message("user", question, documents=_coerce_to_documents(documents))
        ),
        backend,
        io_contract=_ANSWERABILITY_ADAPTER.io_contract,
        model_options=model_options,
    )
    return cast(str, result["answerability"])


def rewrite_question(
    question: str | None,
    context: ChatContext,
    backend: AdapterMixin,
    *,
    model_options: dict | None = None,
) -> str:
    """Rewrite a user's question for retrieval.

    Adapter function that rewrites the question in the next user turn into a
    self-contained query that can be passed to the retriever.

    Output contract — required key: ``rewritten_question``.  Missing the key
    raises :class:`~mellea.backends.adapters.AdapterSchemaMismatchError`;
    extra optional keys do not raise (forward-compatible).

    Args:
        question: Question that the user has posed in response to the last turn in
            `context`. When `None`, the question is extracted from the last
            user message in `context`.
        context: Chat context containing the conversation thus far.
        backend: Backend instance that supports adding the LoRA or aLoRA adapters.
        model_options: Optional model-option overrides (e.g.
            ``{"temperature": 0.1}``).  Merged on top of the adapter default
            (``temperature=0.0``).

    Returns:
        Rewritten version of `question`.

    Raises:
        ValueError: When the model output is not valid JSON.
        AdapterSchemaMismatchError: When the model output is missing the required
            ``rewritten_question`` field.
    """
    question, context = _resolve_question(question, context, backend)
    result = call_intrinsic(
        "query_rewrite",
        context.add(Message("user", question)),
        backend,
        io_contract=_QUERY_REWRITE_ADAPTER.io_contract,
        model_options=model_options,
    )
    return cast(str, result["rewritten_question"])


def clarify_query(
    question: str | None,
    documents: collections.abc.Iterable[str | Document],
    context: ChatContext,
    backend: AdapterMixin,
    *,
    model_options: dict | None = None,
) -> str:
    """Generate clarification for an ambiguous query.

    Adapter function that determines if a user's question requires clarification
    based on the retrieved documents and conversation context, and generates an
    appropriate clarification question if needed.

    Output contract — required key: ``clarification``.  Missing the key raises
    :class:`~mellea.backends.adapters.AdapterSchemaMismatchError`; extra optional
    keys do not raise (forward-compatible).

    Args:
        question: Question that the user has posed. When `None`, the question
            is extracted from the last user message in `context`.
        documents: Document snippets retrieved for the question. Each element
            may be a `Document` or a plain string (automatically wrapped in
            `Document`).
        context: Chat context containing the conversation thus far.
        backend: Backend instance that supports the adapters that implement
            this adapter function.
        model_options: Optional model-option overrides (e.g.
            ``{"temperature": 0.1}``).  Merged on top of the adapter default
            (``temperature=0.0``).

    Returns:
        Clarification question string (e.g., ``"Do you mean A or B?"``), or
        the string ``"CLEAR"`` if no clarification is needed.

    Raises:
        ValueError: When the model output is not valid JSON.
        AdapterSchemaMismatchError: When the model output is missing the required
            ``clarification`` field.
    """
    question, context = _resolve_question(question, context, backend)
    result = call_intrinsic(
        "query_clarification",
        context.add(
            Message("user", question, documents=_coerce_to_documents(documents))
        ),
        backend,
        io_contract=_QUERY_CLARIFY_ADAPTER.io_contract,
        model_options=model_options,
    )
    return cast(str, result["clarification"])


def find_citations(
    response: str | None,
    documents: collections.abc.Iterable[str | Document],
    context: ChatContext,
    backend: AdapterMixin,
    *,
    model_options: dict | None = None,
) -> list[dict]:
    """Find information in documents that supports an assistant response.

    Adapter function that finds sentences in RAG documents that support sentences
    in a potential assistant response to a user question.

    Output contract — each record must contain: ``response_begin``,
    ``response_end``, ``response_text``, ``citation_doc_id``, ``citation_begin``,
    ``citation_end``, ``citation_text``.  A record missing any of these keys
    raises :class:`~mellea.backends.adapters.AdapterSchemaMismatchError`; extra
    optional keys do not raise (forward-compatible).

    Args:
        response: Potential assistant response. When `None`, the response is
            extracted from the last assistant output in `context`.
        documents: Documents that were used to generate `response`. Each element
            may be a `Document` or a plain string. Strings are wrapped in
            `Document` with an auto-generated `doc_id` (`"0"`, `"1"`, ...);
            for explicit control, pass `Document` objects with `doc_id` set.
            `Document` objects without `doc_id` trigger a warning because the
            adapter function uses `doc_id` to identify citation sources.
        context: Context of the dialog between user and assistant at the point where
            the user has just asked a question that will be answered with RAG documents.
        backend: Backend that supports one of the adapters that implements this
            adapter function.
        model_options: Optional model-option overrides (e.g.
            ``{"temperature": 0.1}``).  Merged on top of the adapter default
            (``temperature=0.0``).

    Returns:
        List of records with fields ``response_begin``, ``response_end``,
        ``response_text``, ``citation_doc_id``, ``citation_begin``,
        ``citation_end``, ``citation_text``.  Begin and end offsets are
        character offsets into their respective UTF-8 strings.

    Raises:
        ValueError: When the model output is not valid JSON.
        AdapterSchemaMismatchError: When any record in the output is missing a
            required field.
    """
    response, context = _resolve_response(response, context)
    result = call_intrinsic(
        "citations",
        context.add(
            Message(
                "assistant",
                response,
                documents=_coerce_to_documents(documents, auto_doc_id=False),
            )
        ),
        backend,
        io_contract=_CITATIONS_ADAPTER.io_contract,
        model_options=model_options,
    )
    return cast(list[dict], result["items"])


def check_context_relevance(
    question: str | None,
    document: str | Document,
    context: ChatContext,
    backend: AdapterMixin,
    *,
    model_options: dict | None = None,
) -> str:
    """Test whether a document is relevant to a user's question.

    Deprecated: this function uses a Granite 4.0-only adapter that will not
    receive a Granite 4.1 version and is not maintained. There is no direct
    adapter replacement — use a @generative function for per-document relevance
    filtering. See docs/advanced/intrinsics for a migration example. This
    function will be removed in a future release.

    Adapter function that checks whether a single document contains part or all of
    the answer to a user's question. Does not consider the context in which the
    question was asked.

    This helper uses a Granite 4.0 adapter (``ibm-granite/granite-4.0-micro``) and
    is not available for Granite 4.1 models.

    Output contract — required key: ``context_relevance``.  Missing the key raises
    :class:`~mellea.backends.adapters.AdapterSchemaMismatchError`; extra optional
    keys do not raise (forward-compatible).

    Args:
        question: Question that the user has posed. When `None`, the question
            is extracted from the last user message in `context`.
        document: A retrieved document snippet. May be a `Document` or a plain
            string (automatically wrapped in `Document`).
        context: The chat up to the point where the user asked a question.
        backend: Backend instance that supports the adapters that implement this
            adapter function.
        model_options: Optional model-option overrides (e.g.
            ``{"temperature": 0.1}``).  Merged on top of the adapter default
            (``temperature=0.0``).

    Returns:
        Context relevance judgement as one of the following strings:
        ``"relevant"``, ``"irrelevant"``, or ``"partially relevant"``.

    Raises:
        ValueError: When the model output is not valid JSON.
        AdapterSchemaMismatchError: When the model output is missing the required
            ``context_relevance`` field.
    """
    warnings.warn(
        "check_context_relevance() is deprecated and will be removed in a future "
        "release. The context_relevance adapter is Granite 4.0 only and is not "
        "maintained for newer models. Use a @generative function for per-document "
        "relevance filtering — see docs/advanced/intrinsics for guidance.",
        DeprecationWarning,
        stacklevel=2,
    )
    question, context = _resolve_question(question, context, backend)
    document = _coerce_to_document(document)
    result = call_intrinsic(
        "context_relevance",
        context.add(Message("user", question)),
        backend,
        kwargs={"document_content": document.text},
        io_contract=_CONTEXT_RELEVANCE_ADAPTER.io_contract,
        model_options=model_options,
    )
    return cast(str, result["context_relevance"])


def flag_hallucinated_content(
    response: str | None,
    documents: collections.abc.Iterable[str | Document],
    context: ChatContext,
    backend: AdapterMixin,
    *,
    model_options: dict | None = None,
) -> list[dict]:
    """Flag potentially-hallucinated sentences in an agent's response.

    Adapter function that checks whether the sentences in an agent's response to a
    user question are faithful to the retrieved document snippets. Sentences that do
    not align with the retrieved snippets are flagged as potential hallucinations.

    The ``faithfulness`` field in each record is a string label (e.g.
    ``"faithful"``, ``"hallucinated"``); coercion to a boolean is the caller's
    responsibility.

    Output contract — each record must contain: ``response_begin``,
    ``response_end``, ``response_text``, ``faithfulness``, ``explanation``.  A
    record missing any of these keys raises
    :class:`~mellea.backends.adapters.AdapterSchemaMismatchError`; extra optional
    keys do not raise (forward-compatible).

    Args:
        response: The assistant's response to the user's question in the last turn
            of `context`. When `None`, the response is extracted from the last
            assistant output in `context`.
        documents: Document snippets that were used to generate `response`. Each
            element may be a `Document` or a plain string (automatically wrapped
            in `Document`).
        context: A chat log that ends with a user asking a question.
        backend: Backend instance that supports the adapters that implement this
            adapter function.
        model_options: Optional model-option overrides (e.g.
            ``{"temperature": 0.1}``).  Merged on top of the adapter default
            (``temperature=0.0``).

    Returns:
        List of records with fields ``response_begin``, ``response_end``,
        ``response_text``, ``faithfulness``, ``explanation``.

    Raises:
        ValueError: When the model output is not valid JSON.
        AdapterSchemaMismatchError: When any record in the output is missing a
            required field.
    """
    response, context = _resolve_response(response, context)
    result = call_intrinsic(
        "hallucination_detection",
        context.add(
            Message("assistant", response, documents=_coerce_to_documents(documents))
        ),
        backend,
        io_contract=_HALLUCINATION_ADAPTER.io_contract,
        model_options=model_options,
    )
    return cast(list[dict], result["items"])
