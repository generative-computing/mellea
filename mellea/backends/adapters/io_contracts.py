# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical output contracts for adapter functions (Epic #929 Phase 2, issue #1516).

Before this module existed, an adapter function's output contract travelled to
:func:`~mellea.stdlib.components.intrinsic._util.call_intrinsic` as a caller-supplied
argument, built from a module-level constant in `rag.py` or `guardian.py`, while the
adapter actually resolved by :meth:`~mellea.backends.adapters.AdapterMixin.resolve_adapter`
carried an unrelated placeholder contract. Nothing tied the two together, so passing the
wrong constant was a silent mismatch.

:data:`_INTRINSIC_IO_CONTRACTS` is the single source of truth instead: it is keyed by the
adapter function's catalog name (:attr:`~mellea.backends.adapters.catalog.IntrinsicsCatalogEntry.name`,
e.g. `"guardian-core"`; the same string passed to `call_intrinsic` and
`resolve_adapter`). The shim adapters in `adapter.py` read from it at construction
(via :func:`get_io_contract`); the high-level helpers reach it indirectly, through
the adapter `resolve_adapter` returns. Declaring a capability's contract anywhere
else reintroduces the parallel-argument problem this module exists to close.

An adapter a user registers with a backend takes precedence over the registry for its
own contract: `call_intrinsic` parses with whatever contract the resolved adapter
carries, and only adapters Mellea constructs itself (the lazy shims) get theirs from here.
"""

import json
import math

from ...core import Component
from ._core import (
    _BUILD_PROMPT_NOT_IMPLEMENTED,
    AdapterSchemaMismatchError,
    IOContract,
    _DictContract,
    _ListContract,
)
from .capabilities import DECIDER_CAPABILITY_NAMES
from .catalog import known_intrinsic_names


class _PolicyGuardrailsContract(IOContract):
    """Validate policy-guardrails adapter output: exactly one of `label` or `score`.

    The adapter returns either `{"label": "Yes"|"No"|"Ambiguous"}` or
    `{"score": "Yes"|"No"|"Ambiguous"}`; never both, never neither.
    """

    def build_prompt(self, **_kwargs: object) -> Component:
        raise NotImplementedError(_BUILD_PROMPT_NOT_IMPLEMENTED)

    def parse(self, raw: str) -> dict[str, object]:
        """Parse and validate policy-guardrails output.

        Args:
            raw (str): Raw JSON string from the model.

        Returns:
            dict[str, object]: Parsed output dict with exactly one of `label` or `score`.

        Raises:
            ValueError: When *raw* is not valid JSON or is not a JSON object.
            AdapterSchemaMismatchError: When neither or both of `label` / `score` are present.
        """
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(
                f"Adapter 'policy-guardrails' output must be a JSON object, "
                f"got {type(data).__name__}."
            )
        has_label = "label" in data
        has_score = "score" in data
        if not has_label and not has_score:
            raise AdapterSchemaMismatchError(
                "policy-guardrails",
                frozenset(data.keys()),
                frozenset({"label", "score"}),
                "neither `label` nor `score` was present",
            )
        if has_label and has_score:
            raise AdapterSchemaMismatchError(
                "policy-guardrails",
                frozenset(data.keys()),
                frozenset({"label", "score"}),
                "both `label` and `score` were present",
            )
        return data


class _GuardianCheckContract(IOContract):
    """Validate guardian-core adapter output: `{"guardian": {"score": <float>}}`.

    Checks that the outer `guardian` key is present and that it contains
    a nested `score` key.
    """

    def build_prompt(self, **_kwargs: object) -> Component:
        raise NotImplementedError(_BUILD_PROMPT_NOT_IMPLEMENTED)

    def parse(self, raw: str) -> dict[str, object]:
        """Parse and validate guardian-core output.

        Args:
            raw (str): Raw JSON string from the model.

        Returns:
            dict[str, object]: Parsed output dict containing `{"guardian": {"score": ...}}`.

        Raises:
            ValueError: When *raw* is not valid JSON or is not a JSON object.
            AdapterSchemaMismatchError: When `guardian` key is absent or `guardian.score`
                is absent.
        """
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(
                f"Adapter 'guardian-core' output must be a JSON object, "
                f"got {type(data).__name__}."
            )
        if "guardian" not in data:
            raise AdapterSchemaMismatchError(
                "guardian-core", frozenset(data.keys()), frozenset({"guardian"})
            )
        guardian_val = data["guardian"]
        if not isinstance(guardian_val, dict) or "score" not in guardian_val:
            raise AdapterSchemaMismatchError(
                "guardian-core",
                frozenset(guardian_val.keys())
                if isinstance(guardian_val, dict)
                else frozenset(data.keys()),
                frozenset({"score"}),
            )
        return data


class _DecisionContract(IOContract):
    """Validate one-token decision readout output from the `first_token` rule.

    Expected shape (the `first_token` transformation rule's output, as it
    reaches `call_intrinsic` through the result processor's JSON round-trip):
    a dict holding the sampled top-1 token (`"token"`), a mapping of each
    pool label found among the returned logprobs to its logprob
    (`"label_logprobs"`), the smallest logprob across all returned tokens
    (`"min_returned_logprob"`), and the configured label pool
    (`"labels"`).
    `label_logprobs` carries raw (full-vocabulary) logprobs and may be
    empty or partial; the caller's label-mass check decides whether the
    distribution is usable. `labels` is the capability's configured label
    pool, which the caller checks covers the requested options.
    """

    _REQUIRED_KEYS = frozenset(
        {"token", "label_logprobs", "min_returned_logprob", "labels"}
    )

    def build_prompt(self, **_kwargs: object) -> Component:
        raise NotImplementedError(_BUILD_PROMPT_NOT_IMPLEMENTED)

    def parse(self, raw: str) -> dict[str, object]:
        """Parse and validate one-token decision readout output.

        Args:
            raw (str): Raw JSON string from the `first_token` rule.

        Returns:
            dict[str, object]: Parsed output dict with `token` (the sampled
            top-1 token string), `label_logprobs` (mapping label to raw
            logprob), `min_returned_logprob` (the smallest logprob across
            all returned tokens), and `labels` (the configured label pool).

        Raises:
            ValueError: When *raw* is not valid JSON or is not a JSON object.
            AdapterSchemaMismatchError: When any required key is missing or
                has the wrong type, or when `label_logprobs` maps to
                non-numeric values.
        """
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(
                f"Decision readout output must be a JSON object, "
                f"got {type(data).__name__}."
            )
        missing = self._REQUIRED_KEYS - frozenset(data.keys())
        if missing:
            raise AdapterSchemaMismatchError(
                "decision", frozenset(data.keys()), self._REQUIRED_KEYS
            )
        if not isinstance(data["token"], str):
            raise AdapterSchemaMismatchError(
                "decision",
                frozenset(data.keys()),
                self._REQUIRED_KEYS,
                "'token' must be a string",
            )
        label_logprobs = data["label_logprobs"]
        if not isinstance(label_logprobs, dict) or not all(
            isinstance(k, str)
            and isinstance(v, (int, float))
            and not isinstance(v, bool)
            for k, v in label_logprobs.items()
        ):
            raise AdapterSchemaMismatchError(
                "decision",
                frozenset(data.keys()),
                self._REQUIRED_KEYS,
                "'label_logprobs' must map label strings to numbers",
            )
        if not isinstance(data["min_returned_logprob"], (int, float)) or isinstance(
            data["min_returned_logprob"], bool
        ):
            raise AdapterSchemaMismatchError(
                "decision",
                frozenset(data.keys()),
                self._REQUIRED_KEYS,
                "'min_returned_logprob' must be a number",
            )
        if not isinstance(data["labels"], list) or not all(
            isinstance(label, str) for label in data["labels"]
        ):
            raise AdapterSchemaMismatchError(
                "decision",
                frozenset(data.keys()),
                self._REQUIRED_KEYS,
                "'labels' must be a list of strings",
            )
        return data


class _RequirementCheckContract(IOContract):
    """Validate requirement-check output: `{"requirement_check": {"score": <0.0-1.0 float>}}`.

    Consolidates the score-range validation that both production consumers of this
    capability's output; `core.requirement_check()`
    (`mellea/stdlib/components/intrinsic/core.py`) and `requirement_check_to_bool()`
    (`mellea/stdlib/requirements/requirement.py`); used to hand-roll independently,
    into one declared contract, per issue #1516.
    """

    def build_prompt(self, **_kwargs: object) -> Component:
        raise NotImplementedError(_BUILD_PROMPT_NOT_IMPLEMENTED)

    def parse(self, raw: str) -> dict[str, object]:
        """Parse and validate requirement-check output.

        Args:
            raw (str): Raw JSON string from the model.

        Returns:
            dict[str, object]: Parsed output dict containing
                `{"requirement_check": {"score": <float>}}`.

        Raises:
            ValueError: When *raw* is not valid JSON or is not a JSON object.
            AdapterSchemaMismatchError: When `requirement_check` is absent or not a
                dict, or when its `score` is absent, not a finite number, is a `bool`,
                or falls outside the closed range `[0.0, 1.0]`.
        """
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(
                f"Adapter 'requirement-check' output must be a JSON object, "
                f"got {type(data).__name__}."
            )
        req_check = data.get("requirement_check")
        if not isinstance(req_check, dict):
            raise AdapterSchemaMismatchError(
                "requirement-check",
                frozenset(data.keys()),
                frozenset({"requirement_check"}),
            )
        score = req_check.get("score")
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)  # bool subclasses int; exclude it explicitly
            or not math.isfinite(score)
            or not 0.0 <= score <= 1.0
        ):
            raise AdapterSchemaMismatchError(
                "requirement-check", frozenset(req_check.keys()), frozenset({"score"})
            )
        return data


_INTRINSIC_IO_CONTRACTS: dict[str, IOContract] = {
    "answerability": _DictContract("answerability", frozenset({"answerability"})),
    "query_rewrite": _DictContract("query_rewrite", frozenset({"rewritten_question"})),
    "query_clarification": _DictContract(
        "query_clarification", frozenset({"clarification"})
    ),
    "citations": _ListContract(
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
    "hallucination_detection": _ListContract(
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
    "policy-guardrails": _PolicyGuardrailsContract(),
    "guardian-core": _GuardianCheckContract(),
    "factuality-detection": _DictContract("factuality-detection", frozenset({"score"})),
    "factuality-correction": _DictContract(
        "factuality-correction", frozenset({"correction"})
    ),
    "uncertainty": _DictContract("uncertainty", frozenset({"certainty"})),
    "requirement-check": _RequirementCheckContract(),
    "context-attribution": _ListContract(
        "context-attribution",
        frozenset(
            {
                "response_begin",
                "response_end",
                "response_text",
                "attribution_doc_id",
                "attribution_msg_index",
                "attribution_begin",
                "attribution_end",
                "attribution_text",
            }
        ),
    ),
}
# The decision capability carries its contract derived from the canonical
# set (capabilities.py).
for _decision_name in DECIDER_CAPABILITY_NAMES:
    _INTRINSIC_IO_CONTRACTS[_decision_name] = _DecisionContract()
# Canonical output contract for every built-in adapter function, keyed by its
# catalog `name` (see module docstring). Capture the built-in names before
# `CustomIntrinsicAdapter` can extend the global catalogue at runtime, then keep
# the registry exhaustive over that fixed set at import time; mirroring the
# duplicate-`effective_capability` check in catalog.py.
_BUILTIN_INTRINSIC_NAMES = frozenset(known_intrinsic_names())
_missing_contracts = _BUILTIN_INTRINSIC_NAMES - set(_INTRINSIC_IO_CONTRACTS)
if _missing_contracts:
    raise ValueError(
        f"Built-in catalogued adapter functions with no declared IOContract: "
        f"{_missing_contracts}"
    )
del _missing_contracts


def get_io_contract(name: str) -> IOContract:
    """Return the canonical output contract for an adapter function.

    Args:
        name (str): Catalog name of the adapter function (e.g. `"answerability"`,
            `"guardian-core"`); :attr:`~mellea.backends.adapters.catalog.IntrinsicsCatalogEntry.name`,
            never :attr:`~mellea.backends.adapters.catalog.IntrinsicsCatalogEntry.effective_capability`.
            This is the same string passed to
            :func:`~mellea.stdlib.components.intrinsic._util.call_intrinsic` and
            :meth:`~mellea.backends.adapters.AdapterMixin.resolve_adapter`.

    Returns:
        IOContract: The contract declared in :data:`_INTRINSIC_IO_CONTRACTS` for
            `name`. Adapter functions outside the catalog (e.g. one registered
            through the deprecated `CustomIntrinsicAdapter`) fall back to a dict
            contract with no required keys; permissive about *which* keys are
            present, but still requires the parsed JSON to be an object; a
            top-level array or scalar still raises `ValueError`, since Mellea has
            no declared schema for these adapter functions to relax that far.
    """
    contract = _INTRINSIC_IO_CONTRACTS.get(name)
    if contract is not None:
        return contract
    # Custom/user-defined adapters outside the catalog have no declared schema;
    # accept any JSON object rather than rejecting on an arbitrary key set.
    return _DictContract(name, frozenset())
