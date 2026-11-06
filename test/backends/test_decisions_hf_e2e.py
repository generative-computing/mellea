# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for the one-token decision helpers on `LocalHFBackend`.

The served-endpoint battery lives in `test_decisions_e2e.py`; this file checks
that the local raw-logits readout gives the same typed answers. Runs on any
GPU with enough memory for the checkpoint (CUDA or Apple MPS). Set
`SWITCH_DECISIONS_HF_MODEL_ID` to use a different Granite Switch checkpoint
that embeds the decision adapter function.
"""

import os
from collections.abc import Callable, Iterator

import pytest

from test.conftest import cleanup_gpu_backend, hf_skip
from test.predicates import require_gpu

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.huggingface,
    require_gpu(min_vram_gb=20),
    pytest.mark.skipif(
        int(os.environ.get("CICD", 0)) == 1,
        reason="Skipping local Granite Switch e2e test in CI",
    ),
]

from mellea.backends.huggingface import LocalHFBackend
from mellea.stdlib.components.intrinsic import decisions

_MODEL_ID = os.environ.get(
    "SWITCH_DECISIONS_HF_MODEL_ID", "ibm-granite/granite-switch-4.1-8b"
)
_OUTAGE = (
    "Ticket: customers report card payments failing at checkout since 09:00; "
    "refunds are also stuck."
)
_UNRELATED = "Ticket: how do I change my profile picture?"
_QUESTION = "Does this ticket describe a payment system outage?"


@pytest.fixture(scope="module")
def backend() -> Iterator[LocalHFBackend]:
    """Load the checkpoint with its embedded adapters registered."""
    with hf_skip():
        backend = LocalHFBackend(model_id=_MODEL_ID, load_embedded_adapters=True)
    if not any(name.startswith("decision_") for name in backend.list_adapters()):
        cleanup_gpu_backend(backend, "huggingface")
        pytest.skip(f"{_MODEL_ID} does not embed the decision adapter function")
    yield backend
    cleanup_gpu_backend(backend, "huggingface")


def test_decide_separates_obvious_yes_and_no(backend: LocalHFBackend) -> None:
    yes = decisions.decide(_OUTAGE, backend, _QUESTION)
    no = decisions.decide(_UNRELATED, backend, _QUESTION)
    assert yes.probability > 0.5 > no.probability


def test_choose_routes_to_the_matching_team(backend: LocalHFBackend) -> None:
    answer = decisions.choose(
        _OUTAGE,
        backend,
        "Which team handles this?",
        {
            "payments": "Payment processing and refunds",
            "account": "Login and profile settings",
        },
    )
    assert answer.choice == "payments"
    assert sum(answer.probabilities.values()) == pytest.approx(1.0)


def test_score_rates_an_outage_as_urgent(backend: LocalHFBackend) -> None:
    answer = decisions.score(
        _OUTAGE,
        backend,
        "How urgent is this?",
        ["not urgent", "somewhat urgent", "very urgent"],
    )
    assert answer.score > 1.0


def test_activation_changes_the_readout(backend: LocalHFBackend) -> None:
    """The decision adapter must be what produces the answer.

    The cases above also pass on the plain base model, which is near one-hot
    on them, so they cannot detect lost activation. This replays the same
    calls with `adapter_name` stripped from the chat template call and needs
    a clear gap on at least one case.
    """
    cases: list[Callable[[], decisions.ChoiceAnswer | decisions.ScoreAnswer]] = [
        lambda: decisions.choose(
            "I was charged twice and cannot log in.",
            backend,
            "Which team handles this?",
            {
                "payments": "Payout failures and payment processing",
                "account": "Login and account access",
                "other": "Something else",
            },
        ),
        lambda: decisions.score(
            "My payouts have failed three times. The bank says everything is fine.",
            backend,
            "How urgent is this?",
            ["can wait a week", "this week", "today"],
        ),
    ]
    tokenizer = backend._tokenizer
    original = tokenizer.apply_chat_template
    stripped_names: list[str] = []

    def without_adapter(*args, **kwargs):
        name = kwargs.pop("adapter_name", None)
        if name is not None:
            stripped_names.append(name)
        return original(*args, **kwargs)

    gaps = []
    for call in cases:
        activated = call().probabilities
        tokenizer.apply_chat_template = without_adapter  # type: ignore[method-assign]
        try:
            base = call().probabilities
        finally:
            tokenizer.apply_chat_template = original  # type: ignore[method-assign]
        gaps.append(max(abs(activated[k] - base[k]) for k in activated))
    assert stripped_names, "the decision call never passed adapter_name"
    assert max(gaps) >= 0.15, (
        "The readout with the decision adapter is indistinguishable from the "
        f"base model's (gaps {gaps}); the adapter is probably not activated."
    )
