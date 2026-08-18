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
e.g. `"guardian-core"` — the same string passed to `call_intrinsic` and
`resolve_adapter`), and both the shim adapters in `adapter.py` (via :func:`get_io_contract`)
and the high-level helpers in `mellea.stdlib.components.intrinsic` read from it. Declaring
a capability's contract anywhere else reintroduces the parallel-argument problem this
module exists to close.
"""

import json
import math

from ...core import Component
from ._core import AdapterSchemaMismatchError, IOContract, _DictContract, _ListContract


class _PolicyGuardrailsContract(IOContract):
    """Validate policy-guardrails adapter output: exactly one of `label` or `score`.

    The adapter returns either `{"label": "Yes"|"No"|"Ambiguous"}` or
    `{"score": "Yes"|"No"|"Ambiguous"}` — never both, never neither.
    """

    def build_prompt(self, **_kwargs: object) -> Component:
        raise NotImplementedError(
            "build_prompt is not used in Phase 1; implemented in Phase 2."
        )

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
            )
        if has_label and has_score:
            raise AdapterSchemaMismatchError(
                "policy-guardrails",
                frozenset(data.keys()),
                frozenset({"label", "score"}),
            )
        return data


class _GuardianCheckContract(IOContract):
    """Validate guardian-core adapter output: `{"guardian": {"score": <float>}}`.

    Checks that the outer `guardian` key is present and that it contains
    a nested `score` key.
    """

    def build_prompt(self, **_kwargs: object) -> Component:
        raise NotImplementedError(
            "build_prompt is not used in Phase 1; implemented in Phase 2."
        )

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


class _RequirementCheckContract(IOContract):
    """Validate requirement-check output: `{"requirement_check": {"score": <0.0-1.0 float>}}`.

    Consolidates the score-range validation `requirement_check()` (in
    `mellea.stdlib.components.intrinsic.core`) previously hand-rolled after each call —
    mirroring `requirement_check_to_bool()` in `requirement.py` — into a declared contract,
    per issue #1516.
    """

    def build_prompt(self, **_kwargs: object) -> Component:
        raise NotImplementedError(
            "build_prompt is not used in Phase 1; implemented in Phase 2."
        )

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
    "context_relevance": _DictContract(
        "context_relevance", frozenset({"context_relevance"})
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
"""Canonical output contract for every catalogued adapter function, keyed by its
catalog `name` (see module docstring). Kept exhaustive over
:func:`~mellea.backends.adapters.catalog.known_intrinsic_names` by
`test/backends/test_adapters/test_io_contracts.py`."""


def get_io_contract(name: str) -> IOContract:
    """Return the canonical output contract for an adapter function.

    Args:
        name (str): Catalog name of the adapter function (e.g. `"answerability"`),
            i.e. the same string passed to
            :func:`~mellea.stdlib.components.intrinsic._util.call_intrinsic` and
            :meth:`~mellea.backends.adapters.AdapterMixin.resolve_adapter`.

    Returns:
        IOContract: The contract declared in :data:`_INTRINSIC_IO_CONTRACTS` for
            `name`. Adapter functions outside the catalog (e.g. one registered
            through the deprecated `CustomIntrinsicAdapter`) fall back to a
            permissive dict contract with no required keys, since Mellea has no
            declared schema for them.
    """
    contract = _INTRINSIC_IO_CONTRACTS.get(name)
    if contract is not None:
        return contract
    return _DictContract(name, frozenset())
