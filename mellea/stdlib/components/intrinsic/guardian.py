"""Adapter functions for Guardian safety and hallucination detection.

The Guardian adapters (`guardian-core`, `policy-guardrails`,
`factuality-detection`, `factuality-correction`) require a
`<guardian>`-prefixed envelope as the last user message of the request.
That envelope is built from the `instruction:` field of each adapter's
`io.yaml` via :class:`IntrinsicsRewriter`; the helpers below just resolve
any convenience inputs (e.g. :data:`CRITERIA_BANK` lookups) and pass the
resolved kwargs through.
"""

import collections.abc
import warnings
from typing import cast

from ....backends.adapters import AdapterMixin
from ....core.utils import MelleaLogger
from ...components import Document
from ...context import ChatContext
from ..chat import Message
from ._util import call_intrinsic

_UNSET: object = object()
"""Sentinel distinguishing 'caller omitted scoring_schema' from 'caller passed
the default value explicitly'. Used only to detect conflicts with the
deprecated `target_role` kwarg."""

_TARGET_ROLE_TO_SCHEMA = {"user": "user_prompt", "assistant": "assistant_response"}
"""Mapping used by the deprecated `target_role` path of :func:`guardian_check`."""


def policy_guardrails(
    context: ChatContext,
    backend: AdapterMixin,
    policy_text: str,
    *,
    model_options: dict | None = None,
) -> str:
    """Check whether the last context turn complies with a policy.

    Uses the policy-guardrails LoRA adapter to judge whether the scenario
    described in the last message in `context` is compliant with the given
    `policy_text`.

    Args:
        context: Chat context containing the conversation to evaluate.
        backend: Backend instance that supports LoRA adapters.
        policy_text: Policy text against which compliance is checked.
        model_options: Optional model-generation overrides forwarded to the
            backend (e.g. ``{"temperature": 0}``).

    Returns:
        Compliance label as `"Yes"`, `"No"`, or `"Ambiguous"` (Yes = compliant).

    Raises:
        ValueError: If the adapter returns a result with neither or both of
            `"label"` and `"score"` fields.
    """
    result_json = call_intrinsic(
        "policy-guardrails",
        context,
        backend,
        kwargs={"policy_text": policy_text},
        model_options=model_options,
    )

    has_label = "label" in result_json
    has_score = "score" in result_json
    if has_label == has_score:
        if has_label:
            raise ValueError(
                "Expected Guardian result to have label xor score, but found both."
            )
        raise ValueError(
            "Expected Guardian result to have label xor score, but found neither."
        )
    return result_json["label"] if has_label else result_json["score"]


SCORING_SCHEMA_BANK = {
    "assistant_response": (
        "If the last assistant's text meets the criteria, return 'yes'; "
        "otherwise, return 'no'."
    ),
    "user_prompt": (
        "If the last user's text meets the criteria, return 'yes'; "
        "otherwise, return 'no'."
    ),
    "last_turn": (
        "If the previous turn meets the criteria, return 'yes'; otherwise, return 'no'."
    ),
    "tool_call": (
        "If the assistant's tool call meets the criteria, return 'yes'; "
        "otherwise, return 'no'."
    ),
}
"""Pre-baked scoring-schema phrasings for :func:`guardian_check`.

Keys can be passed directly as the `scoring_schema` parameter; any
other string is used verbatim. Entries must resolve to a yes/no
verdict — the adapter's `io.yaml` constrains output to `"yes"` or
`"no"`, so a schema like `'return "safe" or "unsafe"'` will be
coerced to yes/no by constrained decoding.
"""


CRITERIA_BANK = {
    "harm": (
        "Texts deemed universally harmful are those that pose harm across all "
        "contexts and audiences."
    ),
    "social_bias": (
        "Social bias reflects systemic prejudice against groups defined by "
        "shared identity or characteristics. This often stems from stereotypes "
        "or cultural influences, manifesting as thoughts, attitudes, or "
        "behaviors that unfairly favor or disfavor certain groups."
    ),
    "jailbreak": (
        "A jailbreak text refers to the deliberate evasion of an AI system's "
        "built-in safeguards or ethical boundaries. This practice often "
        "involves crafting specific prompts or scenarios intended to "
        "manipulate the AI into producing restricted or inappropriate content."
    ),
    "profanity": (
        "Profanity refers to the use of language deemed offensive or socially "
        "unacceptable in a given context. This primarily includes curse words, "
        "swear words, and crude expressions, though it typically excludes "
        "slurs or derogatory terms specifically targeting certain groups."
    ),
    "unethical_behavior": (
        "Unethical behavior refers to actions that violate moral or "
        "professional standards, particularly those involving exploitation "
        "for personal gain or a disregard for others' well-being. This "
        "category excludes physical violence or explicit bias and instead "
        "encompasses activities like fraud, theft, or the abuse of power."
    ),
    "violence": (
        "Violence refers to content that promotes or depicts physical harm "
        "to individuals or groups. This includes depictions of assault, "
        "self-harm, or the creation of a hostile environment through threats "
        "or intimidation."
    ),
    "groundedness": (
        "A text is considered ungrounded or unfaithful if it includes "
        "information lacking support from, or directly contradicting, the "
        "provided document(s). This risk arises when the text fabricates "
        "details, misinterprets the content, or makes unsupported "
        "extrapolations beyond what is explicitly stated in the document(s)."
    ),
    "answer_relevance": (
        "A text is considered inadequate if it fails to address or adequately "
        "respond to the posed query. This includes providing off-topic "
        "information, misinterpreting the query, or omitting key details "
        "requested in the query. Information, even if factually sound, is "
        "irrelevant if it fails to directly answer or meet the specific "
        "intent of the query."
    ),
    "context_relevance": (
        "A document is deemed irrelevant when it doesn't contain information "
        "pertinent to the query's specific needs. This means the retrieved or "
        "provided content fails to adequately address the question at hand. "
        "Irrelevant information could be on a different topic, originate from "
        "an unrelated field, or simply not offer any valuable insights for "
        "crafting a suitable response."
    ),
    "function_call": (
        "Function call hallucination occurs when a text includes function "
        "calls that either don't adhere to the correct format defined by the "
        "available tools or are inconsistent with the query's requirements. "
        "This risk arises from function calls containing incorrect argument "
        "names, values, or types that clash with the tool definitions or the "
        "query itself. Common examples include calling functions not present "
        "in the tool definitions, providing invalid argument values, or "
        "attempting to use parameters that don't exist."
    ),
}
"""Pre-baked criteria definitions from the Granite Guardian model card.

Keys can be passed directly to :func:`guardian_check` as the `criteria`
parameter.
"""


def guardian_check(
    context: ChatContext,
    backend: AdapterMixin,
    criteria: str,
    scoring_schema: str | object = _UNSET,
    target_role: str | None = None,
    *,
    model_options: dict | None = None,
) -> float:
    """Check whether text meets specified safety/quality criteria.

    Uses the guardian-core LoRA adapter to judge whether the span
    identified by `scoring_schema` in `context` meets the given
    criteria.

    Args:
        context: Chat context containing the conversation to evaluate.
        backend: Backend instance that supports LoRA adapters.
        criteria: Description of the criteria to check against. Can be a
            key from :data:`CRITERIA_BANK` (e.g. `"harm"`) or a custom
            criteria string.
        scoring_schema: Sentence that tells the judge which span to
            evaluate and how to decide. Can be a key from
            :data:`SCORING_SCHEMA_BANK` (e.g. `"user_prompt"`) or a
            custom string. Defaults to `"assistant_response"`. Must
            still resolve to a yes/no verdict — the adapter's
            `response_format` constrains output to `"yes"`/`"no"`.
        target_role: Deprecated. Role whose last message is being
            evaluated (`"user"` or `"assistant"`). Prefer
            `scoring_schema` with a key from
            :data:`SCORING_SCHEMA_BANK`. Passing both
            `scoring_schema` and `target_role` raises
            :class:`TypeError`.
        model_options: Optional model-generation overrides forwarded to the
            backend (e.g. ``{"temperature": 0}``).

    Returns:
        Risk score as a float between 0.0 (no risk) and 1.0 (risk detected).
    """
    if target_role is not None:
        warnings.warn(
            "`target_role` is deprecated; use `scoring_schema` instead "
            "(e.g. scoring_schema='user_prompt'). Will be removed in a "
            "future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        if scoring_schema is not _UNSET:
            raise TypeError("Pass either `scoring_schema` or `target_role`, not both.")
        if target_role not in _TARGET_ROLE_TO_SCHEMA:
            raise ValueError(
                f"target_role must be 'user' or 'assistant', got {target_role!r}"
            )
        resolved_schema = _TARGET_ROLE_TO_SCHEMA[target_role]
    elif scoring_schema is _UNSET:
        resolved_schema = "assistant_response"
    else:
        assert isinstance(scoring_schema, str)
        if scoring_schema in _TARGET_ROLE_TO_SCHEMA:
            # Looks like an old-style target_role value passed positionally.
            suggested = _TARGET_ROLE_TO_SCHEMA[scoring_schema]
            MelleaLogger.get_logger().warning(
                "guardian_check(scoring_schema=%r) looks like an old-style "
                "target_role value. It will be used as a literal "
                "scoring-schema sentence, which is probably not what you "
                "want. Did you mean scoring_schema=%r? (target_role is "
                "deprecated; prefer SCORING_SCHEMA_BANK keys like "
                "'user_prompt' or 'assistant_response'.)",
                scoring_schema,
                suggested,
            )
        resolved_schema = scoring_schema

    criteria_text = CRITERIA_BANK.get(criteria, criteria)
    scoring_schema_text = SCORING_SCHEMA_BANK.get(resolved_schema, resolved_schema)
    result_json = call_intrinsic(
        "guardian-core",
        context,
        backend,
        kwargs={"criteria": criteria_text, "scoring_schema": scoring_schema_text},
        model_options=model_options,
    )
    return result_json["guardian"]["score"]


def factuality_detection(
    context: ChatContext,
    backend: AdapterMixin,
    *,
    documents: collections.abc.Iterable[str | Document] | None = None,
    model_options: dict | None = None,
) -> str:
    """Determine whether the last assistant response is factually incorrect.

    Adapter function that evaluates the factuality of the assistant's response
    to a user's question. The context should end with a user question followed
    by an assistant answer.

    Reference documents can be supplied in any of these ways:

    1. Pass them explicitly via ``documents=``.
    2. Include them on messages already in ``context``
       (e.g. ``Message("assistant", response, documents=[doc])``).
    3. Perform retrieval and add the resulting documents to ``context``
       before calling this function.

    When ``documents=`` is given it replaces any documents already on the
    last assistant turn; documents on other messages in ``context`` are
    not affected.

    Args:
        context: Chat context ending with a user question and an assistant
            answer.
        backend: Backend instance that supports LoRA/aLoRA adapters.
        documents: Optional reference documents used to ground the factuality
            check.  Each element may be a :class:`~mellea.stdlib.components.Document`
            or a plain string (automatically wrapped in ``Document``).
            When ``None``, any documents already present in ``context`` are
            used.
        model_options: Optional model-generation overrides forwarded to the
            backend (e.g. ``{"temperature": 0}``).

    Returns:
        Factuality label: ``"yes"`` if the response is factually incorrect,
        ``"no"`` if it is correct.

    Raises:
        ValueError: If ``documents`` is provided but ``context`` does not end
            with an assistant turn whose content can be extracted.
    """
    if documents is not None:
        context = _inject_documents(context, documents)
    result_json = call_intrinsic(
        "factuality-detection", context, backend, model_options=model_options
    )
    return result_json["score"]


def factuality_correction(
    context: ChatContext,
    backend: AdapterMixin,
    *,
    documents: collections.abc.Iterable[str | Document] | None = None,
    model_options: dict | None = None,
) -> str:
    """Correct the last assistant response to make it factually accurate.

    Adapter function that rewrites the assistant's response to a user's
    question so that it is consistent with the supplied reference documents.
    The context should end with a user question followed by an assistant answer.

    Reference documents can be supplied in any of these ways:

    1. Pass them explicitly via ``documents=``.
    2. Include them on messages already in ``context``
       (e.g. ``Message("assistant", response, documents=[doc])``).
    3. Perform retrieval and add the resulting documents to ``context``
       before calling this function.

    When ``documents=`` is given it replaces any documents already on the
    last assistant turn; documents on other messages in ``context`` are
    not affected.

    Args:
        context: Chat context ending with a user question and an assistant
            answer.
        backend: Backend instance that supports LoRA/aLoRA adapters.
        documents: Optional reference documents used to ground the correction.
            Each element may be a :class:`~mellea.stdlib.components.Document`
            or a plain string (automatically wrapped in ``Document``).
            When ``None``, any documents already present in ``context`` are
            used.
        model_options: Optional model-generation overrides forwarded to the
            backend (e.g. ``{"temperature": 0}``).

    Returns:
        Corrected assistant response as a plain string.

    Raises:
        ValueError: If ``documents`` is provided but ``context`` does not end
            with an assistant turn whose content can be extracted.
    """
    if documents is not None:
        context = _inject_documents(context, documents)
    result_json = call_intrinsic(
        "factuality-correction", context, backend, model_options=model_options
    )
    return result_json["correction"]


def _inject_documents(
    context: ChatContext, documents: collections.abc.Iterable[str | Document]
) -> ChatContext:
    """Return a copy of *context* with *documents* attached to the last assistant turn.

    Supports both session-generated contexts (where the last turn is a
    :class:`ModelOutputThunk`) and manually-constructed contexts (where the
    last turn is a :class:`Message` added directly via
    :meth:`ChatContext.add`).

    Args:
        context: Chat context whose last element is an assistant response.
        documents: Reference documents to attach.

    Returns:
        New context identical to the input except the last assistant turn now
        carries the supplied documents.

    Raises:
        ValueError: If the last element of *context* is not an assistant
            response that can be extracted, or if the assistant response
            has not been computed yet.
    """
    turn = context.last_turn()
    if turn is None:
        raise ValueError(
            "Context is empty; cannot attach documents to an assistant turn."
        )

    if turn.output is not None and turn.output.value is not None:
        # Session-generated response stored as a ModelOutputThunk.
        # Only the text value is preserved; thunk metadata is intentionally dropped.
        response_text: str = turn.output.value
        prev_ctx = context.previous_node
    elif turn.output is not None and turn.output.value is None:
        raise ValueError(
            "Cannot attach documents: the assistant response has not been computed yet. "
            "Await the response before calling factuality_detection or "
            "factuality_correction with documents=."
        )
    elif (
        turn.model_input is not None
        and isinstance(turn.model_input, Message)
        and turn.model_input.role == "assistant"
    ):
        # Manually-added assistant Message (e.g. built from test fixtures)
        response_text = turn.model_input.content
        prev_ctx = context.previous_node
    else:
        raise ValueError(
            "Cannot attach documents: the last context element is not an "
            "assistant response. Ensure the context ends with an assistant "
            "message before passing documents= to factuality_detection or "
            "factuality_correction."
        )

    if prev_ctx is None:
        raise ValueError(
            "Context has no previous node; cannot reconstruct context with documents."
        )

    return cast(
        ChatContext,
        prev_ctx.add(Message("assistant", response_text, documents=documents)),
    )
