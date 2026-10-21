# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-token decision helpers: `decide`, `choose`, and `score`.

Ask a model a fixed-answer question about some state (a string, a dict or
list rendered as JSON, or a `ChatContext` for conversational state) and get
back a probability, a choice, or a score; never a sentence to parse. The
answer is the model's first generated token, read as a restricted softmax
over the option-label token logprobs, so a decision costs one forward pass
(prefill-only) and is fast enough for control flow.

The capability is served by a Granite Switch checkpoint that embeds the
decision adapter function (named `decision`). Serve it via
vLLM with `--max-logprobs 20` and register the model's embedded adapter
functions, exactly as for the other Switch adapter functions.

Failure is loud: when the answer distribution does not concentrate on the
declared options the call raises rather than returning a plausible-looking
wrong answer.
"""

import copy
import dataclasses
import hashlib
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from typing import cast

import pydantic

from ....backends import ModelOption
from ....backends.adapters import AdapterMixin
from ....backends.adapters.capabilities import DECISION_CAPABILITY
from ....backends.tools import add_tools_from_context_actions
from ....core import Context
from ....core.utils import MelleaLogger
from ...context import ChatContext
from ._util import _call_intrinsic_raw

_LABEL_POOL = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
"""Helper label pool, in call order. Must stay a prefix-compatible match with
the `labels` configured on the capability's `first_token` rule (asserted by a
unit test and by a runtime sanity check on returned labels); the label-mass
floor is the runtime backstop when a served client cannot tokenize."""

_MIN_OPTIONS = 2
_MAX_OPTIONS = 10
"""v1 limits: the readout's coverage bound (`top_logprobs` mechanics) pending
the model team's trained maximum. 11-20 options would need a raised server
`--max-logprobs` and validation."""

_LABEL_MASS_FLOOR = 0.5
"""Below this mass on the option labels the readout raises (R5 fail-closed)."""

_LABEL_MASS_WARN = 0.9
"""Below this mass (but at or above the floor) the readout warns."""

_MISSING_LABEL_BOUND = 0.005
"""Worst-case probability shift tolerated before a missing label raises
as a truncation/server-cap problem rather than being treated as zero."""

_logger = MelleaLogger.get_logger()

_PINNED_TEMPERATURE = 0.0
"""Decisions always decode greedily. On `LocalHFBackend` the label
probabilities are read from the raw LM-head logits, so no generation option
can move them; the pin keeps the sampled token deterministic and protects
served backends whose logprobs may be taken after sampling settings."""

_LOGIT_SHAPING_KEYS: dict[str, str] = {
    "logit_bias": "it shifts token logits before the logprobs may be taken",
    "frequency_penalty": "it reshapes token logits before the logprobs may be taken",
    "presence_penalty": "it reshapes token logits before the logprobs may be taken",
    "repetition_penalty": "it reshapes token logits before the logprobs may be taken",
    "logits_processor": "a custom processor can alter the label probabilities",
}
"""Options that reshape the distribution on a served backend. Whether the
returned logprobs reflect them depends on server configuration Mellea cannot
see (for example vLLM's `--logprobs-mode`), so they are rejected rather than
trusted. They are checked at the top level and inside `extra_body`."""


class PredicateAnswer(pydantic.BaseModel):
    """Answer of `decide`: a probability you can threshold.

    **A bare `if answer:` is always true**: test `answer.probability`
    (or compare the answer directly against a number, which delegates to
    `.probability`). `==` between answers is model equality; compare values
    via `.probability`.

    Attributes:
        probability (float): Probability of the question being true
            (P(true), unrounded).
        probabilities (dict[str, float]): Full distribution, keyed
            `"false"` and `"true"` (unrounded; sums to 1.0).
    """

    probability: float
    probabilities: dict[str, float]

    def __lt__(self, other: object) -> bool:
        """Compare `.probability` against a number."""
        if isinstance(other, (int, float)):
            return self.probability < other
        return NotImplemented

    def __le__(self, other: object) -> bool:
        """Compare `.probability` against a number."""
        if isinstance(other, (int, float)):
            return self.probability <= other
        return NotImplemented

    def __gt__(self, other: object) -> bool:
        """Compare `.probability` against a number."""
        if isinstance(other, (int, float)):
            return self.probability > other
        return NotImplemented

    def __ge__(self, other: object) -> bool:
        """Compare `.probability` against a number."""
        if isinstance(other, (int, float)):
            return self.probability >= other
        return NotImplemented


class ChoiceAnswer(pydantic.BaseModel):
    """Answer of `choose`: the best option key plus the distribution.

    **A bare `if answer:` is always true**: test `answer.choice == "..."`
    instead. `==` between answers is model equality.

    `confidence` is a distribution-shape statistic, not a calibrated
    probability of correctness: the top probability rescaled so a uniform
    distribution over the K options reads 0 and all mass on one option
    reads 1, `(max(p) - 1/K) / (1 - 1/K)`. This is the formula the OpenAI
    decisions specification uses.

    Attributes:
        choice (str): Key of the highest-probability option (first key on
            an exact tie).
        probabilities (dict[str, float]): Distribution over option keys,
            keyed as passed (unrounded; sums to 1.0).
        confidence (float): `(max(p) - 1/K) / (1 - 1/K)`, in `[0.0, 1.0]`.
    """

    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(pydantic.BaseModel):
    """Answer of `score`: the expected level plus per-level probabilities.

    **A bare `if answer:` is always true**: test `answer.score` (or compare
    the answer directly against a number, which delegates to `.score`). `==`
    between answers is model equality; compare values via `.score`.

    `score` is the expected level and may land between levels.
    `confidence` is a distribution-shape statistic, not a calibrated
    probability of correctness: 1 minus the expected distance of the
    level from the most likely level, divided by the mean distance every
    level has from the scale midpoint (a uniform distribution's own
    expected distance), floored at 0. It respects level order, so a
    50/50 split between the lowest and highest levels reads low, and it
    is the formula the OpenAI decisions specification uses.

    Attributes:
        score (float): Expected level, in `[0, K-1]` (unrounded; may land
            between levels).
        probabilities (dict[str, float]): Distribution over levels, keyed
            by level index as a string, `"0"` (lowest) to `"K-1"`
            (unrounded; sums to 1.0).
        confidence (float): Distance-from-mode confidence, in `[0.0, 1.0]`.
    """

    score: float
    probabilities: dict[str, float]
    confidence: float

    def __lt__(self, other: object) -> bool:
        """Compare `.score` against a number."""
        if isinstance(other, (int, float)):
            return self.score < other
        return NotImplemented

    def __le__(self, other: object) -> bool:
        """Compare `.score` against a number."""
        if isinstance(other, (int, float)):
            return self.score <= other
        return NotImplemented

    def __gt__(self, other: object) -> bool:
        """Compare `.score` against a number."""
        if isinstance(other, (int, float)):
            return self.score > other
        return NotImplemented

    def __ge__(self, other: object) -> bool:
        """Compare `.score` against a number."""
        if isinstance(other, (int, float)):
            return self.score >= other
        return NotImplemented


def _text_of(value: object) -> str:
    """Render an option description or state value as the trained layout expects.

    Mirrors the reference implementation's `text_of`: dicts and lists render
    as compact JSON with `ensure_ascii=False`; anything else as `str(x)`
    (`None` as the empty string, which callers map to their fallback).
    Pydantic models and dataclass instances render as the JSON of their
    fields, like a dict.

    Args:
        value: The description or state value.

    Returns:
        Rendered text.
    """
    if isinstance(value, pydantic.BaseModel):
        value = value.model_dump(mode="json")
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return "" if value is None else str(value)


_STRUCTURAL_TOKEN = re.compile(r"<(\|[\w-]+\|>)")
"""A `<|name|>` special-token string (roles, end of text, adapter control
tokens, audio, fim). Tag tokens such as `<tool_call>` and `<think>` cannot
close a turn or switch adapters, so they are not matched."""

_TOKEN_BREAK = "\u2060"
"""U+2060 WORD JOINER: invisible, and splits a special-token string so the
tokenizer reads it as text rather than as the special token."""


def _escape_structural_tokens(text: str) -> str:
    """Break every `<|name|>` special-token string in caller text.

    The chat template turns these strings into real special tokens, and the
    adapter index is derived from the control tokens in the request, so
    unescaped caller text could forge a turn or move the adapter. Ordinary
    text, including `<` and `|` on their own, is unchanged.

    Args:
        text: Caller-supplied text.

    Returns:
        The text with each structural token split by a word joiner.
    """
    return _STRUCTURAL_TOKEN.sub(lambda m: "<" + _TOKEN_BREAK + m.group(1), text)


def _render_options_block(labels: list[str], descriptions: Sequence[object]) -> str:
    """Render option lines exactly as the trained layout expects.

    Each line is `f"{label}: {description}"`, with an empty or `None`
    description falling back to the label and dict/list descriptions
    JSON-rendered; lines are joined with newlines and no trailing newline.
    Structural tokens in a description are escaped.

    Args:
        labels: Label pool slice, one per option.
        descriptions: Per-option descriptions, in call order.

    Returns:
        The `options_block` string for the instruction template.
    """
    lines = []
    for label, description in zip(labels, descriptions, strict=True):
        rendered = _text_of(description) if description not in (None, "") else label
        lines.append(f"{label}: {_escape_structural_tokens(rendered)}")
    return "\n".join(lines)


def _plain_state(state: object) -> object:
    """Prepare a plain-data state for the request: rendered, then escaped.

    Empty values (`""`, `None`, `{}`, `[]`) are returned unchanged so the
    formatter still renders its `(empty)` sentinel. Anything else is
    rendered to text and has its structural tokens escaped.

    Args:
        state: The caller's plain-data state.

    Returns:
        The state to pass to the formatter.
    """
    if state in ("", None, {}, []):
        return state
    return _escape_structural_tokens(_text_of(state))


def _reject_conflicting_options(
    model_options: dict | None, *, source: str = "model_options"
) -> None:
    """Reject model options that would break the one-token readout.

    Args:
        model_options: Model options to check: the caller's per-call options,
            or the backend's own construction-time options.
        source: Where the options came from, named in the error message.

    Raises:
        ValueError: When an option conflicts with a decision-critical
            pinned setting.
    """
    if not model_options:
        return
    conflicts: dict[object, str] = {
        ModelOption.MAX_NEW_TOKENS: (
            "the readout pins max_completion_tokens=1 (one answer token)"
        ),
        ModelOption.TOOLS: "decision calls send no tools",
        ModelOption.TEMPERATURE: (
            f"the readout pins temperature={_PINNED_TEMPERATURE} so the "
            "sampled token is deterministic"
        ),
        "top_p": "the readout decodes greedily; sampling settings would change the answer",
        "top_k": "the readout decodes greedily; sampling settings would change the answer",
        "logprobs": "the readout requests logprobs itself",
        "top_logprobs": ("the readout sizes top_logprobs to the label pool (up to 20)"),
        "n": "the one-token readout requires a single choice",
        "max_completion_tokens": (
            "the readout pins max_completion_tokens=1 (one answer token)"
        ),
        "max_tokens": "the readout pins the completion length to one token",
    }
    # Per-call, `_call_intrinsic_raw` re-asserts these two after this check,
    # so a backend-level value is overridden rather than a conflict.
    overridden_per_call = (
        {ModelOption.MAX_NEW_TOKENS, ModelOption.TEMPERATURE}
        if source != "model_options"
        else set()
    )
    for key, why in conflicts.items():
        if key in model_options and key not in overridden_per_call:
            raise ValueError(
                f"{source} key {key!r} conflicts with the decision "
                f"readout: {why}. Remove it from {source}."
            )
    extra_body = model_options.get("extra_body")
    for scope, options in (
        ("", model_options),
        ("extra_body.", extra_body if isinstance(extra_body, dict) else {}),
    ):
        for key, why in _LOGIT_SHAPING_KEYS.items():
            if key in options:
                raise ValueError(
                    f"{source} key {scope + key!r} conflicts with the decision "
                    f"readout: {why}. Remove it from {source}."
                )
    if model_options.get(ModelOption.STREAM):
        raise ValueError(
            f"{source} requested streaming, but the one-token decision "
            f"readout requires the non-streaming response. Remove "
            f"ModelOption.STREAM from {source}."
        )


def _reject_context_tools(context: ChatContext) -> None:
    """Reject a conversational state whose actions advertise tools.

    Args:
        context: The `ChatContext` being used as decision state.

    Raises:
        ValueError: When any action in the context carries tools.
    """
    tools: dict = {}
    add_tools_from_context_actions(tools, context.actions_for_available_tools() or [])
    if tools:
        raise ValueError(
            "Decision calls do not support tools: the conversational state "
            f"advertises tools {sorted(tools)}, which would be appended to "
            "the trained system prompt by the chat template and break the "
            "one-token readout."
        )


def _require_capability(backend: AdapterMixin) -> None:
    """Check that this backend serves the decision capability.

    Args:
        backend: A backend that supports adapter functions.

    Raises:
        ValueError: When the capability does not resolve, with the underlying
            error wrapped in a clear message.
    """
    try:
        backend.resolve_adapter(DECISION_CAPABILITY)
    except (ValueError, KeyError) as e:
        raise ValueError(
            f"No decision capability ({DECISION_CAPABILITY!r}) is registered "
            "on this backend. Serve a Granite Switch checkpoint that embeds "
            "the decision adapter function via vLLM (with --max-logprobs 20) "
            "and register the model's embedded adapter functions. "
            f"Underlying error: {e}"
        ) from e


def _state_text_for_log(state: object) -> str:
    """Return the text a state contributes to the request, for length/hash logging.

    Args:
        state: The caller's state (plain data or a `ChatContext`).

    Returns:
        The rendered state text, or the concatenated message contents for a
        conversational state.
    """
    if isinstance(state, ChatContext):
        return "\n".join(
            getattr(span, "content", "") for span in state.view_for_generation() or []
        )
    return _text_of(state)


class _RawDecision:
    """Private result of one decision call, before per-type semantics."""

    __slots__ = (
        "capability",
        "label_mass",
        "latency_ms",
        "missing_labels",
        "model",
        "probs",
        "thunk",
        "token",
    )

    def __init__(
        self,
        probs: dict[str, float],
        label_mass: float,
        missing_labels: list[str],
        token: str,
        capability: str,
        model: str,
        latency_ms: float,
        thunk: object,
    ):
        self.probs = probs
        self.label_mass = label_mass
        self.missing_labels = missing_labels
        self.token = token
        self.capability = capability
        self.model = model
        self.latency_ms = latency_ms
        self.thunk = thunk


def _decide_raw(
    state: object,
    backend: AdapterMixin,
    question: str,
    descriptions: Sequence[object],
    *,
    model_options: dict | None = None,
) -> _RawDecision:
    """Run one decision call and apply the label-mass validation.

    Shared machinery of the three public helpers: builds the option block,
    invokes the capability through the adapter-function machinery, and
    validates the readout against the label-mass floor; the single runtime
    criterion; computed over the raw returned logprobs before
    renormalization.

    Args:
        state: The thing being judged; a string, a dict/list (rendered
            as JSON), or a `ChatContext` for conversational state —
            whose own messages then serve as the state (no state turn
            is rendered).
        backend: A backend serving a model with the decision capability.
        question: The fixed-answer question.
        descriptions: Per-option descriptions, in call order (for `decide`,
            ordered `[false, true]`).
        model_options: Optional model-option overrides. Every
            decision-critical setting, including `temperature` (pinned
            to 0.0), is fixed by the capability contract and a conflict
            raises. The backend's own `model_options` are checked the same
            way.

    Returns:
        The raw decision: per-label probabilities over the used labels
        (missing labels treated as zero when within the missing-label
        bound), the label mass, per-call metadata, and the computed
        `ModelOutputThunk` (the raw response survives on
        `mot.raw.response`).

    Raises:
        ValueError: When the option count is outside 2-10; when a caller
            model option conflicts with a pinned decision-critical setting;
            when the capability is not registered on the backend; when
            labels were not returned beyond the missing-label bound
            (truncation/server-cap; an operations problem); or when the
            answer distribution did not concentrate on the options
            (a model/layout problem).
        TypeError: When `state` is a `Context` other than a `ChatContext`.
        AdapterSchemaMismatchError: When the readout output does not match
            the capability's contract.
        AloraActivationError: When the invocation marker does not occur
            exactly once in the rewritten request (state-content
            contamination is the usual cause).
    """
    if not _MIN_OPTIONS <= len(descriptions) <= _MAX_OPTIONS:
        raise ValueError(
            f"A decision needs between {_MIN_OPTIONS} and {_MAX_OPTIONS} "
            f"options, got {len(descriptions)}. The cap is the readout's "
            "coverage bound (top_logprobs mechanics) pending the trained "
            "maximum from the model team."
        )
    _reject_conflicting_options(model_options)
    # The backend's own construction-time options reach the request too, so
    # check them against the same pins. Mocks and third-party backends may
    # not carry a plain dict.
    backend_options = getattr(backend, "model_options", None)
    if isinstance(backend_options, dict):
        _reject_conflicting_options(
            backend_options, source="the backend's model_options"
        )
    # `default_extra_body` is merged into every request but lives outside
    # `model_options`, so check it against the same extra_body pins.
    default_extra_body = getattr(backend, "_default_extra_body", None)
    if isinstance(default_extra_body, dict):
        _reject_conflicting_options(
            {"extra_body": default_extra_body},
            source="the backend's default_extra_body",
        )
    # Re-assert the one-token and greedy pins as per-call options: per-call
    # options outrank backend-level defaults in every later merge, so a
    # backend constructed with its own MAX_NEW_TOKENS or TEMPERATURE cannot
    # silently replace them.
    forced_options = dict(model_options) if model_options else {}
    forced_options[ModelOption.MAX_NEW_TOKENS] = 1
    forced_options[ModelOption.TEMPERATURE] = _PINNED_TEMPERATURE
    # Thinking is overridden rather than rejected: the carried config pins
    # `enable_thinking=false`, and the decision template has no thinking
    # branch, so a chat-configured backend should not fail every decision.
    # An explicit `None` outranks a backend-level `ModelOption.THINKING`.
    forced_options[ModelOption.THINKING] = None
    forced_extra_body = forced_options.get("extra_body")
    if isinstance(forced_extra_body, dict):
        forced_extra_body = copy.deepcopy(forced_extra_body)
        ctk = forced_extra_body.get("chat_template_kwargs")
        if isinstance(ctk, dict):
            ctk.pop("enable_thinking", None)
            if not ctk:
                forced_extra_body.pop("chat_template_kwargs")
        forced_options["extra_body"] = forced_extra_body

    _require_capability(backend)
    labels = _LABEL_POOL[: len(descriptions)]
    kwargs: dict[str, object] = {
        "question": _escape_structural_tokens(question),
        "options_block": _render_options_block(labels, list(descriptions)),
    }
    if isinstance(state, ChatContext):
        _reject_context_tools(state)
        call_context = state
    elif isinstance(state, Context):
        raise TypeError(
            f"A decision state must be plain data or a ChatContext, got "
            f"{type(state).__name__}. Pass a ChatContext for conversational "
            "state."
        )
    else:
        kwargs["state"] = _plain_state(state)
        call_context = ChatContext()

    state_log_text = _state_text_for_log(state)
    started = time.monotonic()
    result, thunk = _call_intrinsic_raw(
        DECISION_CAPABILITY,
        call_context,
        backend,
        kwargs=kwargs,
        model_options=forced_options,
        # The trained decision prompt has no tool block; never expose
        # tools (from model options or context actions) on decision calls.
        tool_calls=False,
    )
    latency_ms = (time.monotonic() - started) * 1000.0

    label_logprobs = {
        str(k): float(v)
        for k, v in cast(Mapping[str, float], result["label_logprobs"]).items()
    }
    min_returned_logprob = float(cast(float, result["min_returned_logprob"]))
    configured_labels = [str(label) for label in cast(list, result["labels"])]

    # The capability's configured label pool must cover the requested
    # options. Without this check, a published config whose pool is
    # shorter than the option count would silently score the extra
    # options as zero whenever the label-mass check still passes.
    uncovered = [label for label in labels if label not in configured_labels]
    if uncovered:
        raise ValueError(
            f"The decision capability's configured label pool "
            f"({len(configured_labels)} labels: {configured_labels}) does "
            f"not cover {uncovered} of the {len(labels)} requested "
            f"options. Options beyond the pool would be silently "
            f"unreachable; reduce the option count or extend the pool."
        )

    # Prefix-compatibility sanity: the capability's label alphabet must
    # agree with this helper's pool, or the readout is being mapped onto
    # the wrong labels.
    unexpected = sorted(label for label in label_logprobs if label not in _LABEL_POOL)
    if unexpected:
        raise ValueError(
            f"The decision capability returned labels {unexpected} outside "
            f"the expected pool {''.join(_LABEL_POOL)}. The capability's "
            "configured label alphabet does not match the helper's; the "
            "readout cannot be mapped onto the options."
        )

    present = {
        label: math.exp(logprob)
        for label, logprob in label_logprobs.items()
        if label in labels
    }
    missing_labels = [label for label in labels if label not in label_logprobs]
    label_mass = sum(present.values())

    # Label-mass floor first: a distribution that did not concentrate
    # on the option labels means the model answered with something
    # other than a label. Reporting this before the missing-label check
    # avoids sending users down the --max-logprobs path when the real
    # issue is that the model did not answer the question.
    if label_mass < _LABEL_MASS_FLOOR:
        raise ValueError(
            f"Decision readout failed: the answer distribution did not "
            f"concentrate on the option labels (label mass {label_mass:.4f} "
            f"< {_LABEL_MASS_FLOOR}). The sampled token was "
            f"{result['token']!r}. The model answered with something other "
            "than an option label, which usually indicates a serving or "
            "template problem rather than a question-quality issue."
        )
    # Missing labels are treated as zero only when the worst-case
    # probability shift stays within the bound; otherwise the readout could
    # not see the labels (truncation/server-cap; an operations problem).
    if missing_labels:
        worst_case_shift = len(missing_labels) * math.exp(min_returned_logprob)
        if worst_case_shift > _MISSING_LABEL_BOUND:
            raise ValueError(
                f"Decision readout could not see {len(missing_labels)} of "
                f"{len(labels)} option labels (missing: {missing_labels}); "
                f"the worst-case probability shift {worst_case_shift:.4f} "
                f"exceeds the tolerance {_MISSING_LABEL_BOUND}. This is a "
                "truncation/server-cap problem: check that the server "
                "allows enough logprobs (vLLM --max-logprobs, the readout "
                "requests up to 20)."
            )
    if label_mass < _LABEL_MASS_WARN:
        _logger.warning(
            "Decision label mass %.4f is below the warning threshold %.2f "
            "(question=%r, option labels=%s).",
            label_mass,
            _LABEL_MASS_WARN,
            question,
            labels,
        )

    total = sum(present.values())
    probs = {label: p / total for label, p in present.items()}
    generation = getattr(thunk, "generation", None)
    model = str(getattr(generation, "model", None) or "")
    raw = _RawDecision(
        probs=probs,
        label_mass=label_mass,
        missing_labels=missing_labels,
        token=str(result["token"]),
        capability=DECISION_CAPABILITY,
        model=model,
        latency_ms=latency_ms,
        thunk=thunk,
    )
    returned_used = sum(1 for label in labels if label in label_logprobs)
    _log_decision(raw, question, labels, descriptions, state_log_text, returned_used)
    return raw


def _log_decision(
    raw: _RawDecision,
    question: str,
    labels: list[str],
    descriptions: Sequence[object],
    state_log_text: str,
    returned_label_count: int,
) -> None:
    """Emit the per-decision log entries.

    Info level carries no state content: state length and a short hash only.
    Option descriptions and the full state go to debug.

    Args:
        raw: The raw decision result.
        question: The question asked.
        labels: The used labels.
        descriptions: The per-option descriptions.
        state_log_text: Rendered state text (for length/hash only at info).
        returned_label_count: Number of labels present in the returned
            logprobs.
    """
    state_hash = hashlib.sha256(state_log_text.encode("utf-8")).hexdigest()[:8]
    distribution = {label: round(p, 4) for label, p in sorted(raw.probs.items())}
    _logger.info(
        "decision: capability=%s model=%s question=%r option_labels=%s "
        "answer_token=%s distribution=%s label_mass=%.4f missing_labels=%s "
        "returned_labels=%d/%d latency_ms=%.1f state_chars=%d "
        "state_sha256_8=%s",
        raw.capability,
        raw.model,
        question,
        labels,
        raw.token,
        distribution,
        raw.label_mass,
        raw.missing_labels,
        returned_label_count,
        len(labels),
        raw.latency_ms,
        len(state_log_text),
        state_hash,
    )
    _logger.debug(
        "decision details: question=%r descriptions=%s state=%r",
        question,
        list(descriptions),
        state_log_text,
    )


def decide(
    state: object,
    backend: AdapterMixin,
    question: str,
    *,
    true: str | None = None,
    false: str | None = None,
    model_options: dict | None = None,
) -> PredicateAnswer:
    """Ask a yes/no question about a state and get a probability.

    The predicate form of a decision: the answer is a probability you
    threshold. Ordering operators delegate to `.probability`, so
    `if decisions.decide(ticket, backend, "...") >= 0.7:` reads naturally.
    Decoding is pinned to greedy (`temperature=0.0`) along with every
    other decision-critical setting. On `LocalHFBackend` the probabilities
    come from the raw logits at the label token ids, so generation options
    cannot move them.

    Args:
        state: The thing being judged; a string, a dict/list (rendered
            as JSON), or a `ChatContext` for conversational state. A
            plain state is rendered as the trained `State:` turn; a
            `ChatContext` forwards its own messages as the state (no
            state turn is rendered for it). A multi-turn context
            diverges from the trained single-state-turn layout.
        backend: A backend serving a Granite Switch model with the decision
            capability.
        question: The yes/no question, in plain English.
        true: Optional description of what "true" means for your domain
            (reference default: `"Yes / true"`).
        false: Optional description of what "false" means for your domain
            (reference default: `"No / false"`).
        model_options: Optional model-option overrides. All
            decision-critical settings, including `temperature`, are
            pinned and a conflict raises.

    Returns:
        PredicateAnswer with `.probability` (P(true), unrounded) and
        `.probabilities` keyed `{"false", "true"}`.

    Raises:
        ValueError: When a caller model option conflicts with a pinned
            decision-critical setting; when the capability is not
            registered on the backend; when the answer distribution did
            not concentrate on the options; or when labels were not
            returned beyond the missing-label bound.
        TypeError: When `state` is a `Context` other than a `ChatContext`.
        AdapterSchemaMismatchError: When the readout output does not match
            the capability's contract.
        AloraActivationError: When the invocation marker does not occur
            exactly once in the rewritten request (state-content
            contamination is the usual cause).

    Example:
        ```python
        if decisions.decide(ticket, backend, "Does this require urgent attention?") >= 0.7:
            escalate(ticket)
        ```
    """
    descriptions = [false or "No / false", true or "Yes / true"]
    raw = _decide_raw(
        state, backend, question, descriptions, model_options=model_options
    )
    p_false = raw.probs.get(_LABEL_POOL[0], 0.0)
    p_true = raw.probs.get(_LABEL_POOL[1], 0.0)
    return PredicateAnswer(
        probability=p_true, probabilities={"false": p_false, "true": p_true}
    )


def choose(
    state: object,
    backend: AdapterMixin,
    question: str,
    options: dict[str, str],
    *,
    model_options: dict | None = None,
) -> ChoiceAnswer:
    """Ask a fixed-choice question and get the best option key plus the distribution.

    `options` maps option key to description; only the description is
    rendered into the prompt (in insertion order), and the answer maps back
    onto the keys. Ties break to the first key in insertion order.

    Args:
        state: The thing being judged; a string, a dict/list (rendered
            as JSON), or a `ChatContext` for conversational state. A
            plain state is rendered as the trained `State:` turn; a
            `ChatContext` forwards its own messages as the state (no
            state turn is rendered for it). A multi-turn context
            diverges from the trained single-state-turn layout.
        backend: A backend serving a Granite Switch model with the decision
            capability.
        question: The question, in plain English.
        options: Mapping of option key to description, with 2-10 entries.
        model_options: Optional model-option overrides. All
            decision-critical settings, including `temperature`, are
            pinned and a conflict raises.

    Returns:
        ChoiceAnswer with `.choice` (best option key), `.probabilities`
        (keyed as passed, unrounded, sums to 1.0), and `.confidence`
        (`(max(p) - 1/K) / (1 - 1/K)`, an uncalibrated distribution-shape
        statistic).

    Raises:
        ValueError: When the option count is outside 2-10; when a caller
            model option conflicts with a pinned decision-critical
            setting; when the capability is not registered on the backend;
            when the answer distribution did not concentrate on the
            options; or when labels were not returned beyond the
            missing-label bound.
        TypeError: When `state` is a `Context` other than a `ChatContext`.
        AdapterSchemaMismatchError: When the readout output does not match
            the capability's contract.
        AloraActivationError: When the invocation marker does not occur
            exactly once in the rewritten request (state-content
            contamination is the usual cause).

    Example:
        ```python
        route = decisions.choose(ticket, backend, "Which team handles this?",
                                 {"payments": "Payout failures and refunds",
                                  "account": "Login and account access"})
        route.choice        # "payments"
        route.probabilities # {"payments": 0.91, "account": 0.09}
        ```
    """
    keys = list(options)
    # An empty description falls back to the option key, which is the only
    # meaningful text available; the label letter would just repeat itself.
    descriptions = [
        options[key] if options[key] not in (None, "") else key for key in keys
    ]
    raw = _decide_raw(
        state, backend, question, descriptions, model_options=model_options
    )
    probs = [raw.probs.get(label, 0.0) for label in _LABEL_POOL[: len(keys)]]
    best = max(range(len(keys)), key=lambda i: probs[i])
    k = len(keys)
    # jev's choice confidence: the top probability rescaled so a uniform
    # distribution reads 0 and all mass on one option reads 1.
    confidence = (probs[best] - 1.0 / k) / (1.0 - 1.0 / k)
    return ChoiceAnswer(
        choice=keys[best],
        probabilities=dict(zip(keys, probs, strict=True)),
        confidence=confidence,
    )


def score(
    state: object,
    backend: AdapterMixin,
    question: str,
    levels: list[str],
    *,
    model_options: dict | None = None,
) -> ScoreAnswer:
    """Ask a rating question and get the expected level plus per-level probabilities.

    `levels` describes the levels from lowest to highest; the score is the
    expected level and may land between levels.

    Args:
        state: The thing being judged; a string, a dict/list (rendered
            as JSON), or a `ChatContext` for conversational state. A
            plain state is rendered as the trained `State:` turn; a
            `ChatContext` forwards its own messages as the state (no
            state turn is rendered for it). A multi-turn context
            diverges from the trained single-state-turn layout.
        backend: A backend serving a Granite Switch model with the decision
            capability.
        question: The question, in plain English.
        levels: Level descriptions, low to high, with 2-10 entries.
        model_options: Optional model-option overrides. All
            decision-critical settings, including `temperature`, are
            pinned and a conflict raises.

    Returns:
        ScoreAnswer with `.score` (expected level in `[0, K-1]`, unrounded,
        may land between levels), `.probabilities` (keyed by level index as
        a string, unrounded, sums to 1.0), and `.confidence`
        (distance-from-mode, an uncalibrated distribution-shape statistic
        in `[0.0, 1.0]`).

    Raises:
        ValueError: When the level count is outside 2-10; when a caller
            model option conflicts with a pinned decision-critical
            setting; when the capability is not registered on the backend;
            when the answer distribution did not concentrate on the
            options; or when labels were not returned beyond the
            missing-label bound.
        TypeError: When `state` is a `Context` other than a `ChatContext`.
        AdapterSchemaMismatchError: When the readout output does not match
            the capability's contract.
        AloraActivationError: When the invocation marker does not occur
            exactly once in the rewritten request (state-content
            contamination is the usual cause).

    Example:
        ```python
        sev = decisions.score(ticket, backend, "How urgent is this?",
                              ["can wait a week", "this week", "today"])
        sev.score           # 1.7
        sev.probabilities   # {"0": 0.05, "1": 0.20, "2": 0.75}
        ```
    """
    raw = _decide_raw(
        state, backend, question, list(levels), model_options=model_options
    )
    probs = [raw.probs.get(label, 0.0) for label in _LABEL_POOL[: len(levels)]]
    k = len(levels)
    expected = sum(i * p for i, p in enumerate(probs))
    # jev's score confidence: 1 minus the expected distance from the modal
    # level, over the mean distance every level has from the scale midpoint
    # (a uniform distribution's own expected distance).
    mode = max(range(k), key=lambda i: probs[i])
    distance_from_mode = sum(p * abs(i - mode) for i, p in enumerate(probs))
    midpoint = (k - 1) / 2.0
    uniform_distance = sum(abs(i - midpoint) for i in range(k)) / k
    confidence = max(0.0, 1.0 - distance_from_mode / uniform_distance)
    return ScoreAnswer(
        score=expected,
        probabilities={str(i): p for i, p in enumerate(probs)},
        confidence=confidence,
    )
