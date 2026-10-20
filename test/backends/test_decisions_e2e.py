# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for the one-token decision capability against a served endpoint.

Gated on a Granite Switch checkpoint with the decision adapter function,
served via vLLM (or any OpenAI-compatible server) with `--max-logprobs 20`:

    SWITCH_DECISIONS_BASE_URL   e.g. http://127.0.0.1:8000/v1
    SWITCH_DECISIONS_MODEL_ID   the served model id
    SWITCH_DECISIONS_API_KEY    optional; defaults to "EMPTY" (vLLM)

Falls back to VLLM_SWITCH_TEST_BASE_URL / GRANITE_SWITCH_MODEL_ID (the
env vars used by the other switch intrinsics e2e tests and the nightly
script), so a single server configuration drives both batteries.

Skips with an actionable reason when the endpoint is not configured.

Battery (spec §4.3): the deployment activation probe (a case whose
unactivated answer is known wrong — the guard for the serving template's
adapter-map miss), the reference implementation's built-in example as the
smoke target, the sanity battery (obvious yes/no, order-swap on `choose`
only, label mass, a conversational case), p50/p95 latency, prefix-cache
hits via `/metrics`, `finish_reason == "length"`, and the
`max_completion_tokens` server compatibility.

The activation probe combines an objectively-verifiable semantic case with
order-swap consistency (the position-bias signature of an unactivated base
model): an adapter-map miss leaves the base model answering the trained
format, which is position-biased and much flatter over the labels. A
model-team-validated "known wrong" case should replace this probe when
one is published (see the design doc's model-team ask).
"""

import logging
import os
import re
import statistics
import time

import pytest

from mellea.backends import ModelOption
from mellea.backends.openai import OpenAIBackend
from mellea.formatters import TemplateFormatter
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import decisions
from mellea.stdlib.context import ChatContext

_BASE_URL = (
    os.environ.get("SWITCH_DECISIONS_BASE_URL")
    or os.environ.get("VLLM_SWITCH_TEST_BASE_URL", "")
).rstrip("/")
_MODEL_ID = os.environ.get("SWITCH_DECISIONS_MODEL_ID") or os.environ.get(
    "GRANITE_SWITCH_MODEL_ID", ""
)
_API_KEY = os.environ.get("SWITCH_DECISIONS_API_KEY", "EMPTY")

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.openai,
    pytest.mark.vllm,
    pytest.mark.skipif(
        not _BASE_URL or not _MODEL_ID,
        reason=(
            "No decision endpoint configured: serve a Granite Switch "
            "checkpoint with the decision adapter function via vLLM with "
            "--max-logprobs 20, then set SWITCH_DECISIONS_BASE_URL (e.g. "
            "http://127.0.0.1:8000/v1) and SWITCH_DECISIONS_MODEL_ID to "
            "run this battery."
        ),
    ),
]

# The reference implementation's built-in example.
_REFERENCE_STATE = (
    "My payouts have failed three times. The bank says everything is fine. "
    "Can someone please fix this?"
)
_REFERENCE_OPTIONS = {
    "payments": "Payout failures and payment processing",
    "account": "Login and account access",
    "other": "Something else",
}


@pytest.fixture(scope="module")
def backend():
    """An OpenAI backend with the served model's embedded adapters."""
    return OpenAIBackend(
        model_id=_MODEL_ID,
        formatter=TemplateFormatter(model_id=_MODEL_ID),
        base_url=_BASE_URL,
        api_key=_API_KEY,
        load_embedded_adapters=True,
    )


def _label_mass(caplog) -> float:
    """Extract the highest label mass from the captured decision log entries."""
    masses = [
        float(m.group(1))
        for r in caplog.records
        if r.getMessage().startswith("decision:")
        for m in [re.search(r"label_mass=([0-9.]+)", r.getMessage())]
        if m
    ]
    assert masses, "no decision log entry captured"
    return min(masses)


class TestActivationProbe:
    """The deployment-time guard for the serving template's adapter-map miss."""

    def test_objectively_verifiable_case(self, backend, caplog):
        """An obvious semantic case: an unactivated base model is position-biased."""
        with caplog.at_level(logging.INFO):
            answer = decisions.decide(
                "The ticket says: the server is on fire and customers cannot pay.",
                backend,
                "Does this ticket describe a payment system outage?",
            )
        assert answer.probability > 0.5
        assert _label_mass(caplog) >= 0.95

    def test_order_swap_consistency_on_choose(self, backend):
        """The same option must win regardless of presentation order.

        An unactivated base model exhibits position bias (prefers the first
        label); the activated adapter judges the option descriptions.
        Deliberately not run on `decide`: swapping its options inverts the
        trained [false, true] order, where a gap is expected.
        """
        forward = decisions.choose(
            _REFERENCE_STATE,
            backend,
            "Which team should handle this ticket?",
            _REFERENCE_OPTIONS,
        )
        swapped = decisions.choose(
            _REFERENCE_STATE,
            backend,
            "Which team should handle this ticket?",
            {k: _REFERENCE_OPTIONS[k] for k in reversed(list(_REFERENCE_OPTIONS))},
        )
        assert forward.choice == swapped.choice


class TestReferenceExampleSmoke:
    """The reference implementation's built-in example, as the smoke target."""

    def test_choose_queue_routing(self, backend):
        route = decisions.choose(
            _REFERENCE_STATE,
            backend,
            "Which team should handle this ticket?",
            _REFERENCE_OPTIONS,
        )
        assert route.choice in _REFERENCE_OPTIONS
        assert set(route.probabilities) == set(_REFERENCE_OPTIONS)
        assert sum(route.probabilities.values()) == pytest.approx(1.0)
        assert 0.0 <= route.confidence <= 1.0

    def test_decide_escalation(self, backend):
        answer = decisions.decide(
            _REFERENCE_STATE,
            backend,
            "Does this message require urgent human attention?",
        )
        assert 0.0 <= answer.probability <= 1.0
        assert set(answer.probabilities) == {"false", "true"}

    def test_score_severity(self, backend):
        sev = decisions.score(
            _REFERENCE_STATE,
            backend,
            "How urgent is this?",
            ["can wait a week", "this week", "today"],
        )
        assert 0.0 <= sev.score <= 2.0
        assert set(sev.probabilities) == {"0", "1", "2"}


class TestSanityBattery:
    def test_obvious_yes(self, backend):
        answer = decisions.decide(
            "The sky is clear and blue today.", backend, "Is the sky described as blue?"
        )
        assert answer.probability >= 0.5

    def test_obvious_no(self, backend):
        answer = decisions.decide(
            "The sky is clear and blue today.",
            backend,
            "Is the sky described as green?",
        )
        assert answer.probability <= 0.5

    def test_label_mass_high(self, backend, caplog):
        with caplog.at_level(logging.INFO):
            decisions.decide(_REFERENCE_STATE, backend, "Is this urgent?")
        assert _label_mass(caplog) >= 0.95

    def test_conversational_state(self, backend):
        """A ChatContext state forwards its messages as the state."""
        context = (
            ChatContext()
            .add(Message("user", "My payouts have failed three times."))
            .add(Message("assistant", "I can help you with the payment issue."))
        )
        answer = decisions.decide(
            context, backend, "Is this conversation about payments?"
        )
        assert answer.probability > 0.5

    def test_temperature_override_rejected(self, backend):
        with pytest.raises(ValueError, match="temperature"):
            decisions.decide(
                _REFERENCE_STATE,
                backend,
                "Does this message require urgent human attention?",
                model_options={ModelOption.TEMPERATURE: 0.0},
            )


class TestDiverseQuestions:
    """Verify sensible answers across question types, state formats, and domains.

    These are not accuracy benchmarks; they check that the model produces
    reasonable answers on a diverse set of inputs.
    """

    def test_factual_reasoning(self, backend):
        """The model can read numeric facts from the state."""
        answer = decisions.decide(
            "The server has 4 CPU cores and 16 GB RAM, 91% of memory is in use.",
            backend,
            "Is memory usage above 80 percent?",
        )
        assert answer.probability > 0.8

    def test_subjective_judgment(self, backend):
        """The model can make a judgment call on a subjective question."""
        answer = decisions.decide(
            "A customer waited 45 minutes for a response to their urgent billing question.",
            backend,
            "Is a 45-minute wait acceptable for urgent billing questions?",
        )
        assert answer.probability < 0.3

    def test_routing_login_issue(self, backend):
        """A login/password issue routes to account, not payments."""
        route = decisions.choose(
            "Customer cannot log in, password reset email not arriving, "
            "tried 3 times over 2 hours.",
            backend,
            "Which team handles this?",
            {
                "account": "Login, password, and account access",
                "payments": "Billing and payment issues",
                "other": "Something else",
            },
        )
        assert route.choice == "account"

    def test_severity_degraded_performance(self, backend):
        """A degraded service scores in the upper half of the severity scale."""
        severity = decisions.score(
            "The database is responding but queries take 30 seconds "
            "instead of the usual 200ms.",
            backend,
            "How severe is this performance issue?",
            ["cosmetic", "annoying", "impacting users", "critical"],
        )
        # Score lands between "annoying" and "impacting users"
        assert severity.score >= 1.5

    def test_categorization_intent(self, backend):
        """The model can categorize customer intent from free text."""
        route = decisions.choose(
            "The user is asking how to export their data before closing their account.",
            backend,
            "What is the customer intent?",
            {
                "churn": "Wants to leave or cancel",
                "support": "Needs help with a feature",
                "sales": "Interested in buying more",
            },
        )
        assert route.choice == "churn"

    def test_json_state(self, backend):
        """A dict state is rendered as JSON and judged correctly."""
        answer = decisions.decide(
            {
                "service": "payment-api",
                "error_rate": 0.15,
                "latency_p99_ms": 4200,
                "status": "degraded",
            },
            backend,
            "Is this service healthy?",
        )
        assert answer.probability < 0.3


class TestReadoutMechanics:
    def test_finish_reason_is_length(self, backend):
        """`max_completion_tokens: 1` ends the completion at the answer token."""
        raw = decisions._decide_raw(
            _REFERENCE_STATE, backend, "Is this urgent?", ["No / false", "Yes / true"]
        )
        response = raw.thunk.raw.response
        finish_reasons = [c["finish_reason"] for c in response["choices"]]
        assert finish_reasons == ["length"]

    def test_max_completion_tokens_accepted_by_server(self, backend):
        """A successful one-token readout proves the parameter name is accepted."""
        answer = decisions.decide(_REFERENCE_STATE, backend, "Is this urgent?")
        assert 0.0 <= answer.probability <= 1.0


class TestLatency:
    def test_p50_max_latency(self, backend):
        """Report p50/max decision latency (prefill-only cost)."""
        latencies = []
        for _ in range(10):
            started = time.monotonic()
            decisions.decide(_REFERENCE_STATE, backend, "Is this urgent?")
            latencies.append((time.monotonic() - started) * 1000.0)
        latencies.sort()
        p50 = statistics.median(latencies)
        p95 = latencies[-1]
        print(f"\ndecision latency ms: p50={p50:.1f} max={p95:.1f} n={len(latencies)}")
        # Generous bound: a prefill-only readout must stay well under a
        # second once warm; this catches a silently-degenerate deployment.
        assert p50 < 5000.0

    def test_latency_vs_requirement_check(self, backend):
        """Compare against the stock requirement-check intrinsic when served."""
        from mellea.backends.adapters import AdapterMixin

        try:
            backend.resolve_adapter("requirement-check")
        except (ValueError, KeyError):
            pytest.skip(
                "requirement-check is not served on this endpoint; the "
                "cross-intrinsic latency comparison needs a checkpoint that "
                "embeds it"
            )
        from mellea.stdlib.components.intrinsic import core

        context = (
            ChatContext()
            .add(Message("user", "Write a haiku about the sea."))
            .add(Message("assistant", "Blue waves touch the sky."))
        )
        decision_latencies = []
        requirement_latencies = []
        for _ in range(5):
            started = time.monotonic()
            decisions.decide(_REFERENCE_STATE, backend, "Is this urgent?")
            decision_latencies.append((time.monotonic() - started) * 1000.0)
            started = time.monotonic()
            core.requirement_check(context, backend, "The response is a haiku.")
            requirement_latencies.append((time.monotonic() - started) * 1000.0)
        print(
            f"\ndecision p50={statistics.median(decision_latencies):.1f}ms vs "
            f"requirement-check p50={statistics.median(requirement_latencies):.1f}ms"
        )


class TestPrefixCache:
    def test_shared_state_hits_prefix_cache(self, backend):
        """Repeated decisions over one state must hit the engine's prefix cache."""
        import urllib.request

        metrics_url = (
            _BASE_URL[: -len("/v1")] if _BASE_URL.endswith("/v1") else _BASE_URL
        )
        metrics_url = metrics_url + "/metrics"
        try:
            with urllib.request.urlopen(metrics_url, timeout=10) as response:
                before = _prefix_cache_hits(response.read().decode())
        except Exception as e:
            pytest.skip(f"metrics endpoint not reachable at {metrics_url}: {e}")

        # Same state, different questions: the state prefix is shared.
        for question in (
            "Is this urgent?",
            "Is this about payments?",
            "Should a manager look at it?",
        ):
            decisions.decide(_REFERENCE_STATE, backend, question)

        try:
            with urllib.request.urlopen(metrics_url, timeout=10) as response:
                after = _prefix_cache_hits(response.read().decode())
        except Exception as e:
            pytest.skip(f"metrics endpoint not reachable at {metrics_url}: {e}")
        assert after > before, (
            f"prefix cache hits did not increase ({before} -> {after}); the "
            "shared-state prefix-cache claim is not holding on this deployment"
        )


def _prefix_cache_hits(metrics_text: str) -> int:
    """Sum vLLM's prefix cache hit counters from a /metrics payload.

    Matches both the bare Prometheus name (older vLLM) and the
    `vllm:`-prefixed, engine-labeled form (vLLM 0.26+). The counter counts
    cached tokens, not requests — any increase proves prefix reuse.
    """
    total = 0
    for match in re.finditer(
        r"^(?:vllm:)?prefix_cache_hits_total(?:\{[^}]*\})?\s+([0-9.eE+-]+)",
        metrics_text,
        re.MULTILINE,
    ):
        total += int(float(match.group(1)))
    return total
