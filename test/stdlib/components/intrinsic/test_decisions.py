# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the decision helpers (decide/choose/score).

Stubs `mfuncs.act` and `backend.resolve_adapter` (the `test_core_contracts.py`
pattern) and exercises the helper boundary: answer types and comparison
semantics, the label-mass floor and its two error classes, pinned-parameter
conflicts, alias resolution, and the state paths.
"""

import json
import logging
import math
from unittest.mock import MagicMock

import pytest

from mellea.backends import ModelOption
from mellea.backends.adapters import (
    AdapterSchemaMismatchError,
    AloraActivationError,
    get_io_contract,
)
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import _util, decisions
from mellea.stdlib.components.intrinsic.decisions import (
    ChoiceAnswer,
    PredicateAnswer,
    ScoreAnswer,
)
from mellea.stdlib.context import ChatContext

_STATE = "My payouts have failed three times."


def _rule_output(
    label_logprobs: dict[str, float],
    token: str = "A",
    min_returned: float = -20.0,
    labels: list[str] | None = None,
) -> str:
    """Serialize a `first_token` rule output as the helper receives it."""
    return json.dumps(
        {
            "token": token,
            "label_logprobs": label_logprobs,
            "min_returned_logprob": min_returned,
            "labels": labels if labels is not None else decisions._LABEL_POOL,
        }
    )


def _stub_backend(
    raw_output: str | Exception, monkeypatch, intrinsic_name: str = "decision"
) -> MagicMock:
    """Stub the model call; returns the backend mock.

    `raw_output` is either the JSON the rule produced or an exception to
    raise from the model call.
    """
    backend = MagicMock()
    backend.resolve_adapter.return_value = MagicMock(
        io_contract=get_io_contract(intrinsic_name)
    )

    def fake_act(_intrinsic, context, _backend, *, model_options=None, **_kwargs):
        if isinstance(raw_output, Exception):
            raise raw_output
        thunk = MagicMock()
        thunk.is_computed.return_value = True
        thunk.value = raw_output
        thunk.generation.model = "test-switch-model"
        return thunk, context

    monkeypatch.setattr(_util.mfuncs, "act", fake_act)
    return backend


# ---------------------------------------------------------------------------
# decide
# ---------------------------------------------------------------------------


class TestDecide:
    def test_decide_returns_probability(self, monkeypatch):
        # [false, true] order: A = false, B = true.
        backend = _stub_backend(
            _rule_output({"A": -2.0, "B": -0.5}, token="B"), monkeypatch
        )
        answer = decisions.decide(_STATE, backend, "Is this urgent?")
        assert isinstance(answer, PredicateAnswer)
        expected_true = math.exp(-0.5) / (math.exp(-2.0) + math.exp(-0.5))
        assert answer.probability == pytest.approx(expected_true)
        assert answer.probabilities["false"] == pytest.approx(1 - expected_true)
        assert answer.probabilities["true"] == pytest.approx(expected_true)
        assert sum(answer.probabilities.values()) == pytest.approx(1.0)

    def test_decide_default_descriptions_rendered(self, monkeypatch):
        backend = _stub_backend(_rule_output({"A": -1.0, "B": -1.0}), monkeypatch)
        captured = {}

        def fake_act(intrinsic, context, _backend, *, model_options=None, **_kw):
            captured["kwargs"] = dict(intrinsic.intrinsic_kwargs)
            captured["context"] = context
            thunk = MagicMock()
            thunk.is_computed.return_value = True
            thunk.value = _rule_output({"A": -1.0, "B": -1.0})
            thunk.generation.model = "m"
            return thunk, context

        monkeypatch.setattr(_util.mfuncs, "act", fake_act)
        decisions.decide(_STATE, backend, "Is this urgent?")
        assert captured["kwargs"]["options_block"] == "A: No / false\nB: Yes / true"
        assert captured["kwargs"]["question"] == "Is this urgent?"
        assert captured["kwargs"]["state"] == _STATE
        # The plain-state path passes an empty context (no history).
        assert captured["context"].view_for_generation() == []

    def test_decide_custom_descriptions_rendered(self, monkeypatch):
        backend = _stub_backend(_rule_output({"A": -1.0, "B": -1.0}), monkeypatch)
        captured = {}

        def fake_act(intrinsic, context, _backend, *, model_options=None, **_kw):
            captured["kwargs"] = dict(intrinsic.intrinsic_kwargs)
            thunk = MagicMock()
            thunk.is_computed.return_value = True
            thunk.value = _rule_output({"A": -1.0, "B": -1.0})
            thunk.generation.model = "m"
            return thunk, context

        monkeypatch.setattr(_util.mfuncs, "act", fake_act)
        decisions.decide(
            _STATE,
            backend,
            "Is this urgent?",
            true="Same-day human action is required",
            false="Routine handling is fine",
        )
        assert captured["kwargs"]["options_block"] == (
            "A: Routine handling is fine\nB: Same-day human action is required"
        )


# ---------------------------------------------------------------------------
# choose
# ---------------------------------------------------------------------------


class TestChoose:
    OPTIONS = {
        "payments": "Payout failures and payment processing",
        "account": "Login and account access",
        "other": "Something else",
    }

    def test_choose_returns_choice_and_distribution(self, monkeypatch):
        backend = _stub_backend(
            _rule_output({"A": -0.2, "B": -1.7, "C": -4.0}, token="A"), monkeypatch
        )
        answer = decisions.choose(_STATE, backend, "Which team?", self.OPTIONS)
        assert isinstance(answer, ChoiceAnswer)
        assert answer.choice == "payments"
        total = sum(math.exp(lp) for lp in (-0.2, -1.7, -4.0))
        expected = {
            "payments": math.exp(-0.2) / total,
            "account": math.exp(-1.7) / total,
            "other": math.exp(-4.0) / total,
        }
        for key, prob in expected.items():
            assert answer.probabilities[key] == pytest.approx(prob)

    def test_choose_confidence_entropy_formula(self, monkeypatch):
        backend = _stub_backend(
            _rule_output({"A": -0.2, "B": -1.7, "C": -4.0}), monkeypatch
        )
        answer = decisions.choose(_STATE, backend, "Which team?", self.OPTIONS)
        probs = [answer.probabilities[k] for k in self.OPTIONS]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        assert answer.confidence == pytest.approx(1 - entropy / math.log(3))

    def test_choose_tie_breaks_to_first_key(self, monkeypatch):
        backend = _stub_backend(
            _rule_output({"A": -1.0, "B": -1.0, "C": -9.0}), monkeypatch
        )
        answer = decisions.choose(_STATE, backend, "Which team?", self.OPTIONS)
        assert answer.choice == "payments"

    def test_choose_rejects_option_counts_outside_limits(self, monkeypatch):
        backend = _stub_backend(_rule_output({"A": -1.0}), monkeypatch)
        with pytest.raises(ValueError, match="between 2 and 10"):
            decisions.choose(_STATE, backend, "q?", {"only": "one option"})
        with pytest.raises(ValueError, match="between 2 and 10"):
            decisions.choose(
                _STATE,
                backend,
                "q?",
                {f"opt{i}": f"description {i}" for i in range(11)},
            )


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


class TestScore:
    def test_score_returns_expected_level(self, monkeypatch):
        backend = _stub_backend(
            _rule_output({"A": -3.0, "B": -0.5, "C": -1.0}, token="B"), monkeypatch
        )
        answer = decisions.score(
            _STATE, backend, "How urgent?", ["can wait", "this week", "today"]
        )
        assert isinstance(answer, ScoreAnswer)
        total = sum(math.exp(lp) for lp in (-3.0, -0.5, -1.0))
        probs = [math.exp(lp) / total for lp in (-3.0, -0.5, -1.0)]
        assert answer.score == pytest.approx(sum(i * p for i, p in enumerate(probs)))
        assert set(answer.probabilities) == {"0", "1", "2"}
        for i, prob in enumerate(probs):
            assert answer.probabilities[str(i)] == pytest.approx(prob)

    def test_score_confidence_variance_formula(self, monkeypatch):
        """Confidence is 1 - Var(level)/Var_max over the level distribution."""
        backend = _stub_backend(
            _rule_output({"A": -3.0, "B": -0.5, "C": -1.0}), monkeypatch
        )
        answer = decisions.score(_STATE, backend, "How urgent?", ["a", "b", "c"])
        probs = list(answer.probabilities.values())
        k = 3
        expected = sum(i * p for i, p in enumerate(probs))
        variance = sum(p * (i - expected) ** 2 for i, p in enumerate(probs))
        var_max = ((k - 1) / 2.0) ** 2
        assert answer.confidence == pytest.approx(1 - variance / var_max)

    def test_score_confidence_all_mass_on_one_level_is_one(self, monkeypatch):
        """All mass on a single level means zero level variance: confidence 1."""
        backend = _stub_backend(
            _rule_output({"A": -12.0, "B": -12.0, "C": -0.01}), monkeypatch
        )
        answer = decisions.score(_STATE, backend, "q?", ["a", "b", "c"])
        # exp(-12) residue on the other labels leaves a hair under 1.0.
        assert answer.confidence == pytest.approx(1.0, abs=1e-3)

    def test_score_confidence_extremes_split_is_zero(self, monkeypatch):
        """The design doc's named case: a 50/50 split between the lowest
        and highest levels reads as zero confidence, never outside [0, 1]."""
        # K=10, labels A (level 0) and J (level 9) each at 0.5.
        backend = _stub_backend(
            _rule_output({"A": -0.7, "J": -0.7}, min_returned=-8.0), monkeypatch
        )
        answer = decisions.score(
            _STATE, backend, "q?", [f"level {i}" for i in range(10)]
        )
        assert answer.confidence == pytest.approx(0.0)
        assert 0.0 <= answer.confidence <= 1.0

    def test_score_confidence_uniform_is_low(self, monkeypatch):
        """A uniform distribution reads low, not high."""
        backend = _stub_backend(
            _rule_output({"A": -1.0, "B": -1.0, "C": -1.0}), monkeypatch
        )
        answer = decisions.score(_STATE, backend, "q?", ["a", "b", "c"])
        # Uniform K=3: Var(level) = 2/3, Var_max = 1 -> 1/3.
        assert answer.confidence == pytest.approx(1 - (2 / 3))
        assert answer.confidence < 0.5

    def test_score_may_land_between_levels(self, monkeypatch):
        backend = _stub_backend(_rule_output({"A": -1.0, "B": -1.0}), monkeypatch)
        answer = decisions.score(_STATE, backend, "q?", ["low", "high"])
        assert answer.score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Answer-type behaviour
# ---------------------------------------------------------------------------


class TestAnswerTypes:
    """Comparison semantics: ordering delegates, equality stays model equality."""

    @pytest.fixture
    def answers(self, monkeypatch):
        decide_backend = _stub_backend(
            _rule_output({"A": -2.0, "B": -0.5}), monkeypatch
        )
        score_backend = _stub_backend(
            _rule_output({"A": -3.0, "B": -0.5, "C": -1.0}), monkeypatch
        )
        predicate = decisions.decide(_STATE, decide_backend, "q?")
        score = decisions.score(_STATE, score_backend, "q?", ["a", "b", "c"])
        return predicate, score

    def test_bare_if_is_always_true(self, answers):
        predicate, score = answers
        assert bool(predicate)
        assert bool(score)
        assert predicate  # bare truthiness, never the probability
        assert score

    def test_assert_answer(self, answers):
        predicate, _ = answers
        assert predicate

    def test_ans_or_default(self, answers):
        predicate, _ = answers
        assert (predicate or "default") is predicate
        assert (PredicateAnswer(probability=0.0, probabilities={}) or "d") is not None

    def test_ordering_delegates_to_primary_field(self, answers):
        predicate, score = answers
        assert (predicate >= 0.5) == (predicate.probability >= 0.5)
        assert (predicate > 0.0) is True
        assert (predicate < 1.0) is True
        assert (predicate <= 1.0) is True
        assert (score >= 1.0) == (score.score >= 1.0)
        assert (score > 0.0) is True

    def test_ordering_with_non_number_not_implemented(self, answers):
        predicate, _ = answers
        with pytest.raises(TypeError):
            predicate < "not a number"

    def test_model_equality(self, answers):
        predicate, _ = answers
        same = PredicateAnswer(
            probability=predicate.probability,
            probabilities=dict(predicate.probabilities),
        )
        assert predicate == same
        assert predicate != PredicateAnswer(
            probability=0.123, probabilities={"false": 0.0, "true": 1.0}
        )
        # Choice equality is field equality, not .choice == str.
        choose_answer = ChoiceAnswer(
            choice="a", probabilities={"a": 1.0}, confidence=1.0
        )
        assert choose_answer != "a"


# ---------------------------------------------------------------------------
# Label-mass validation
# ---------------------------------------------------------------------------


class TestLabelMassFloor:
    """The single runtime criterion and its two error classes."""

    def test_low_mass_raises_did_not_concentrate(self, monkeypatch):
        # Mass ~0.25 on the two labels: below the 0.5 floor.
        backend = _stub_backend(
            _rule_output({"A": -2.0, "B": -2.2}, token="Q", min_returned=-2.2),
            monkeypatch,
        )
        with pytest.raises(ValueError, match=r"did not.*concentrate"):
            decisions.decide(_STATE, backend, "q?")

    def test_missing_label_beyond_bound_raises_truncation(self, monkeypatch):
        # One label missing; smallest returned logprob -2.0 -> worst-case
        # shift exp(-2.0) = 0.135 > 0.005: the label could not be seen.
        backend = _stub_backend(
            _rule_output({"A": -0.2}, token="A", min_returned=-2.0), monkeypatch
        )
        with pytest.raises(ValueError, match="truncation/server-cap"):
            decisions.decide(_STATE, backend, "q?")

    def test_missing_label_within_bound_treated_as_zero(self, monkeypatch):
        # Missing label bounded by exp(-8.0) = 3e-4 <= 0.005: treated as zero.
        backend = _stub_backend(
            _rule_output({"A": -0.05}, token="A", min_returned=-8.0), monkeypatch
        )
        answer = decisions.decide(_STATE, backend, "q?")
        assert answer.probability == pytest.approx(0.0)
        assert answer.probabilities["true"] == 0.0

    def test_warn_band_logs_warning(self, monkeypatch, caplog):
        # Mass ~0.67 on the labels: above the floor, below the 0.9 warning.
        backend = _stub_backend(
            _rule_output({"A": -1.0, "B": -1.2}, token="B", min_returned=-6.0),
            monkeypatch,
        )
        with caplog.at_level(logging.WARNING):
            decisions.decide(_STATE, backend, "q?")
        assert any("label mass" in r.getMessage().lower() for r in caplog.records)

    def test_high_mass_no_warning(self, monkeypatch, caplog):
        backend = _stub_backend(
            _rule_output({"A": -6.0, "B": -0.02}, token="B", min_returned=-8.0),
            monkeypatch,
        )
        with caplog.at_level(logging.WARNING):
            decisions.decide(_STATE, backend, "q?")
        assert not [r for r in caplog.records if "label mass" in r.getMessage().lower()]

    def test_unexpected_label_alphabet_raises(self, monkeypatch):
        backend = _stub_backend(_rule_output({"a": -0.2}, token="a"), monkeypatch)
        with pytest.raises(ValueError, match=r"outside.*expected pool"):
            decisions.decide(_STATE, backend, "q?")


# ---------------------------------------------------------------------------
# Model option conflicts
# ---------------------------------------------------------------------------


class TestOptionConflicts:
    """Pinned decision-critical settings raise on caller conflicts."""

    @pytest.mark.parametrize(
        "options",
        [
            {ModelOption.STREAM: True},
            {ModelOption.THINKING: False},
            {ModelOption.THINKING: True},
            {ModelOption.MAX_NEW_TOKENS: 1},
            {ModelOption.MAX_NEW_TOKENS: 64},
            {ModelOption.TOOLS: {}},
            {"logprobs": False},
            {"top_logprobs": 5},
            {"n": 2},
            {"max_completion_tokens": 5},
            {"max_tokens": 5},
            {ModelOption.TEMPERATURE: 0.0},
            {ModelOption.TEMPERATURE: 0.7},
            {"top_p": 0.9},
            {"top_k": 5},
            {"logit_bias": {"32": 5}},
            {"frequency_penalty": 0.5},
            {"presence_penalty": 0.5},
            {"repetition_penalty": 1.5},
            {"logits_processor": []},
            {"extra_body": {"logit_bias": {"32": 5}}},
            {"extra_body": {"repetition_penalty": 1.5}},
            {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
        ],
    )
    def test_conflicting_options_raise(self, monkeypatch, options):
        backend = _stub_backend(_rule_output({"A": -1.0, "B": -1.0}), monkeypatch)
        with pytest.raises(ValueError, match="decision"):
            decisions.decide(_STATE, backend, "q?", model_options=options)

    def test_temperature_pin_overrides_backend_level_temperature(self, monkeypatch):
        """A backend built with temperature=0.8 must still decode greedily.

        The helper defaults temperature to 0.0 elsewhere, so asserting the
        default alone would pass without the pin. A non-zero backend value
        is what the pin has to defeat.
        """
        backend = _stub_backend(_rule_output({"A": -1.0, "B": -1.0}), monkeypatch)
        backend.model_options = {ModelOption.TEMPERATURE: 0.8}
        captured = {}

        def fake_act(intrinsic, context, _backend, *, model_options=None, **_kw):
            captured["model_options"] = model_options
            thunk = MagicMock()
            thunk.is_computed.return_value = True
            thunk.value = _rule_output({"A": -1.0, "B": -1.0})
            thunk.generation.model = "m"
            return thunk, context

        monkeypatch.setattr(_util.mfuncs, "act", fake_act)
        decisions.decide(_STATE, backend, "q?")
        assert captured["model_options"][ModelOption.TEMPERATURE] == 0.0

    def test_unrelated_caller_options_are_allowed(self, monkeypatch):
        """The stricter checks do not reject ordinary Mellea options."""
        backend = _stub_backend(_rule_output({"A": -1.0, "B": -1.0}), monkeypatch)
        result = decisions.decide(
            _STATE, backend, "q?", model_options={ModelOption.SEED: 7}
        )
        assert 0.0 <= result.probability <= 1.0

    @pytest.mark.parametrize(
        "backend_options",
        [
            {"logprobs": False},
            {"top_logprobs": 5},
            {"n": 2},
            {"top_p": 0.9},
            {"top_k": 5},
            {ModelOption.TOOLS: {}},
            {ModelOption.THINKING: True},
            {ModelOption.STREAM: True},
            {"logit_bias": {"32": 5}},
            {"repetition_penalty": 1.5},
            {"extra_body": {"frequency_penalty": 0.5}},
        ],
    )
    def test_conflicting_backend_options_raise(self, monkeypatch, backend_options):
        backend = _stub_backend(_rule_output({"A": -1.0, "B": -1.0}), monkeypatch)
        backend.model_options = backend_options
        with pytest.raises(ValueError, match="backend's model_options"):
            decisions.decide(_STATE, backend, "q?")

    @pytest.mark.parametrize(
        "backend_options",
        [{ModelOption.MAX_NEW_TOKENS: 256}, {ModelOption.TEMPERATURE: 0.8}],
    )
    def test_backend_token_and_temperature_defaults_are_overridden(
        self, monkeypatch, backend_options
    ):
        """Per-call pins outrank these backend-level defaults, so they don't raise."""
        backend = _stub_backend(_rule_output({"A": -1.0, "B": -1.0}), monkeypatch)
        backend.model_options = backend_options
        assert decisions.decide(_STATE, backend, "q?").probability is not None


# ---------------------------------------------------------------------------
# Capability resolution and state paths
# ---------------------------------------------------------------------------


class TestCapabilityResolution:
    def test_capability_not_found_clear_error(self, monkeypatch):
        backend = MagicMock()
        backend.resolve_adapter.side_effect = ValueError("not found")
        with pytest.raises(ValueError, match="No decision capability"):
            decisions.decide(_STATE, backend, "q?")


class TestStatePaths:
    def test_non_chat_context_state_is_a_type_error(self, monkeypatch):
        from mellea.stdlib.context import SimpleContext

        backend = _stub_backend(_rule_output({"A": -1.0, "B": -1.0}), monkeypatch)
        with pytest.raises(TypeError, match="ChatContext"):
            decisions.decide(SimpleContext(), backend, "q?")

    def test_chat_context_state_forwards_messages(self, monkeypatch):
        backend = _stub_backend(_rule_output({"A": -1.0, "B": -1.0}), monkeypatch)
        captured = {}

        def fake_act(intrinsic, context, _backend, *, model_options=None, **_kw):
            captured["kwargs"] = dict(intrinsic.intrinsic_kwargs)
            captured["context"] = context
            thunk = MagicMock()
            thunk.is_computed.return_value = True
            thunk.value = _rule_output({"A": -1.0, "B": -1.0})
            thunk.generation.model = "m"
            return thunk, context

        monkeypatch.setattr(_util.mfuncs, "act", fake_act)
        context = (
            ChatContext()
            .add(Message("user", "My payouts failed."))
            .add(Message("assistant", "I can help with that."))
        )
        decisions.decide(context, backend, "q?")
        # A ChatContext state passes no state kwarg; its messages forward.
        assert "state" not in captured["kwargs"]
        assert captured["context"] is context

    def test_empty_chat_context_state_is_an_error(self, monkeypatch):
        backend = _stub_backend(_rule_output({"A": -1.0, "B": -1.0}), monkeypatch)
        with pytest.raises(ValueError, match="forwards no history"):
            decisions.decide(ChatContext(), backend, "q?")

    def test_history_context_plus_state_kwarg_is_an_error(self, monkeypatch):
        """The both-error, through call_intrinsic directly."""
        backend = _stub_backend(_rule_output({"A": -1.0, "B": -1.0}), monkeypatch)
        context = ChatContext().add(Message("user", "history"))
        with pytest.raises(ValueError, match="both"):
            _util.call_intrinsic(
                "decision",
                context,
                backend,
                kwargs={"question": "q?", "state": "some state"},
            )

    def test_state_kwarg_with_empty_context_skips_history_guard(self, monkeypatch):
        """The carve-out: plain state on an empty context reaches the model."""
        backend = _stub_backend(_rule_output({"A": -1.0, "B": -1.0}), monkeypatch)
        result = _util.call_intrinsic(
            "decision",
            ChatContext(),
            backend,
            kwargs={"question": "q?", "state": "some state"},
        )
        assert "label_logprobs" in result

    def test_context_advertising_tools_rejected(self, monkeypatch):
        backend = _stub_backend(_rule_output({"A": -1.0, "B": -1.0}), monkeypatch)
        context = ChatContext().add(Message("user", "hi"))
        monkeypatch.setattr(
            decisions,
            "add_tools_from_context_actions",
            lambda tools, actions: tools.update({"lookup": object()}),
        )
        with pytest.raises(ValueError, match="tools"):
            decisions.decide(context, backend, "q?")


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


class TestBackendDefaultOverrides:
    """Per-call pins must outrank backend-level model-option defaults.

    The conflict check only sees caller-supplied options; a backend
    constructed with its own MAX_NEW_TOKENS merges those as backend
    defaults later, where per-call options take precedence (review
    finding). The helper therefore re-asserts the pin as a per-call
    option and disables tool exposure entirely.
    """

    def test_max_new_tokens_pin_survives_backend_default(self, monkeypatch):
        captured = {}

        def fake_act(intrinsic, context, _backend, *, model_options=None, **kwargs):
            captured["model_options"] = model_options
            captured["tool_calls"] = kwargs.get("tool_calls")
            thunk = MagicMock()
            thunk.is_computed.return_value = True
            thunk.value = _rule_output({"A": -1.0, "B": -1.0})
            thunk.generation.model = "m"
            return thunk, context

        monkeypatch.setattr(_util.mfuncs, "act", fake_act)
        backend = MagicMock()
        backend.resolve_adapter.return_value = MagicMock(
            io_contract=get_io_contract("decision")
        )
        # A backend-level default of 256 must not survive to the call.
        decisions.decide(_STATE, backend, "q?")
        assert captured["model_options"][ModelOption.MAX_NEW_TOKENS] == 1
        assert captured["model_options"][ModelOption.TEMPERATURE] == 0.0
        # Decisions never expose tools to the model.
        assert captured["tool_calls"] is False

    def test_error_order_concentration_before_truncation(self, monkeypatch):
        """A diffuse answer reports concentration, not --max-logprobs advice.

        When both checks would fail (low mass AND missing labels beyond
        the bound), the concentration error is the actionable one — the
        usual cause is a silent activation miss.
        """
        # One label at -2.0 (mass ~0.135 < 0.5), the other missing with a
        # loose bound that would also trip the truncation check.
        backend = _stub_backend(
            _rule_output({"A": -2.0}, min_returned=-2.0), monkeypatch
        )
        with pytest.raises(ValueError, match=r"did not.*concentrate"):
            decisions.decide(_STATE, backend, "q?")


class TestLabelPoolCoverage:
    """The configured pool must cover the requested options."""

    def test_short_configured_label_pool_raises(self, monkeypatch):
        backend = _stub_backend(
            _rule_output({"A": -0.2, "B": -1.0}, labels=["A", "B"]), monkeypatch
        )
        with pytest.raises(ValueError, match="label pool"):
            decisions.choose(
                _STATE, backend, "q?", {"one": "x", "two": "y", "three": "z"}
            )

    def test_full_pool_passes(self, monkeypatch):
        backend = _stub_backend(
            _rule_output({"A": -0.2, "B": -1.0}, labels=list("AB")), monkeypatch
        )
        answer = decisions.decide(_STATE, backend, "q?")
        assert 0.0 <= answer.probability <= 1.0

    def test_shifted_pool_raises(self, monkeypatch):
        """A pool that omits the first used label is also a mismatch."""
        backend = _stub_backend(
            _rule_output({"B": -0.2, "C": -1.0}, labels=["B", "C"]), monkeypatch
        )
        with pytest.raises(ValueError, match="not cover"):
            decisions.decide(_STATE, backend, "q?")


class TestErrorPropagation:
    def test_alora_activation_error_propagates(self, monkeypatch):
        error = AloraActivationError("decision", "base", "qualified", [1, 2, 3])
        backend = _stub_backend(error, monkeypatch)
        with pytest.raises(AloraActivationError):
            decisions.decide(_STATE, backend, "q?")

    def test_contract_mismatch_propagates(self, monkeypatch):
        backend = _stub_backend(json.dumps({"wrong": "shape"}), monkeypatch)
        with pytest.raises(AdapterSchemaMismatchError):
            decisions.decide(_STATE, backend, "q?")

    def test_none_model_output_raises(self, monkeypatch):
        backend = MagicMock()
        backend.resolve_adapter.return_value = MagicMock(
            io_contract=get_io_contract("decision")
        )

        def fake_act(_intrinsic, context, _backend, *, model_options=None, **_kw):
            thunk = MagicMock()
            thunk.is_computed.return_value = True
            thunk.value = None
            thunk.generation.model = "m"
            return thunk, context

        monkeypatch.setattr(_util.mfuncs, "act", fake_act)
        with pytest.raises(ValueError, match="None"):
            decisions.decide(_STATE, backend, "q?")

    def test_per_decision_log_entry_emitted(self, monkeypatch, caplog):
        backend = _stub_backend(
            _rule_output({"A": -2.0, "B": -0.5}, token="B"), monkeypatch
        )
        with caplog.at_level(logging.INFO):
            decisions.decide(_STATE, backend, "Is this urgent?")
        entry = [r for r in caplog.records if r.getMessage().startswith("decision:")]
        assert len(entry) == 1
        message = entry[0].getMessage()
        assert "question='Is this urgent?'" in message
        assert "label_mass=" in message
        assert "state_chars=" in message
        assert "state_sha256_8=" in message
        assert "latency_ms=" in message
        # No state content at info level.
        assert _STATE not in message
