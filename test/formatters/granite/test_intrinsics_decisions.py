# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the one-token decision capability's input and output processing.

Covers the rewriter's trained-layout rendering (system injection, state
turn, marker guard, logprobs sizing, tools/documents rejection) and the
`first_token` transformation rule with the raw-mode gate, against
synthesized configurations shaped like the published decision io.yaml.
The golden-layout parity test against the reference implementation lives
in `test_intrinsics_decisions_golden.py`.
"""

import json
import logging

import pytest

from mellea.backends.adapters._core import AloraActivationError
from mellea.backends.adapters.adapter import _apply_carried_decider_config
from mellea.formatters.granite.base.types import (
    AssistantMessage,
    ChatCompletion,
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    Document,
    SystemMessage,
    ToolDefinition,
    UserMessage,
)
from mellea.formatters.granite.intrinsics.input import IntrinsicsRewriter
from mellea.formatters.granite.intrinsics.output import IntrinsicsResultProcessor
from mellea.stdlib.components.intrinsic import decisions

# ---------------------------------------------------------------------------
# Fixtures: synthesized configs shaped like the published decision io.yaml
# ---------------------------------------------------------------------------


def _published_config(name: str = "decision") -> dict:
    """The published decision io.yaml shape: decider fields present-with-null."""
    return {
        "name": name,
        "model": None,
        "response_format": None,
        "transformations": None,
        "instruction": None,
        "parameters": {"max_completion_tokens": 1, "temperature": 0.0},
        "sentence_boundaries": None,
    }


def _decider_config(name: str = "decision", **overrides: object) -> dict:
    """A merged decision config, as `_discover_embedded_adapters` produces."""
    published = _published_config(name)
    published.update(overrides)
    merged, _reason = _apply_carried_decider_config(name, published)
    assert merged is not None, "test fixture must produce a valid merged config"
    return merged


def _rewriter(name: str = "decision", **overrides: object) -> IntrinsicsRewriter:
    return IntrinsicsRewriter(config_dict=_decider_config(name, **overrides))


def _choose_kwargs(
    question: str = "Which team should handle this ticket?",
    options: dict[str, str] | None = None,
    state: object = "The payouts have failed three times.",
) -> dict:
    """Build the intrinsic kwargs the `choose` helper builds."""
    if options is None:
        options = {
            "payments": "Payout failures and payment processing",
            "account": "Login and account access",
        }
    keys = list(options)
    labels = decisions._LABEL_POOL[: len(keys)]
    descriptions = [options[key] for key in keys]
    return {
        "question": question,
        "options_block": decisions._render_options_block(labels, descriptions),
        "state": state,
    }


def _reference_messages(
    state: object, question: str, labels: list[str], descriptions: list[object]
) -> list[dict]:
    """Reimplement the reference implementation's message construction.

    Mirrors `render_prompt_alora` from the model team's reference script:
    a fixed system prompt, a state turn, and an instruction turn opening
    with the invocation marker.
    """
    system_prompt = (
        "You are a decision engine. Read the state, then answer the question "
        "by replying with exactly one option label from the list. Do not "
        "explain."
    )

    def text_of(x):
        if isinstance(x, (dict, list)):
            return json.dumps(x, ensure_ascii=False)
        return "" if x is None else str(x)

    ctx = text_of(state) if state not in ("", None, {}, []) else "(empty)"
    q_lines = ["Question:", text_of(question), "", "Options:"]
    for lab, d in zip(labels, descriptions, strict=True):
        q_lines.append(f"{lab}: {text_of(d) if d not in (None, '') else lab}")
    q_lines += ["", "Reply with exactly one option label."]
    instr = "### DECISION" + "\n" + "\n".join(q_lines)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "State:\n" + ctx},
        {"role": "user", "content": instr},
    ]


def _as_dicts(messages: list) -> list[dict]:
    return [{"role": m.role, "content": m.content} for m in messages]


def _transform(rewriter: IntrinsicsRewriter, **kwargs) -> ChatCompletion:
    request = ChatCompletion(messages=[])
    return rewriter.transform(request, **kwargs)


# ---------------------------------------------------------------------------
# Layout parity with the reference implementation
# ---------------------------------------------------------------------------


class TestLayoutParity:
    """The rewriter must reproduce the reference layout exactly."""

    def test_choose_layout_matches_reference(self):
        rewriter = _rewriter()
        out = _transform(rewriter, **_choose_kwargs())
        assert _as_dicts(out.messages) == _reference_messages(
            "The payouts have failed three times.",
            "Which team should handle this ticket?",
            ["A", "B"],
            ["Payout failures and payment processing", "Login and account access"],
        )

    def test_decide_layout_matches_reference(self):
        rewriter = _rewriter()
        out = _transform(
            rewriter,
            question="Does this message require urgent human attention?",
            options_block="A: No / false\nB: Yes / true",
            state="The bank says everything is fine.",
        )
        assert _as_dicts(out.messages) == _reference_messages(
            "The bank says everything is fine.",
            "Does this message require urgent human attention?",
            ["A", "B"],
            ["No / false", "Yes / true"],
        )

    def test_score_layout_matches_reference(self):
        rewriter = _rewriter()
        out = _transform(
            rewriter,
            question="How urgent is this?",
            options_block="A: can wait a week\nB: this week\nC: today",
            state="A customer cannot log in.",
        )
        assert _as_dicts(out.messages) == _reference_messages(
            "A customer cannot log in.",
            "How urgent is this?",
            ["A", "B", "C"],
            ["can wait a week", "this week", "today"],
        )

    def test_empty_state_renders_sentinel(self):
        for empty in ("", None, {}, []):
            rewriter = _rewriter()
            out = _transform(rewriter, **_choose_kwargs(state=empty))
            assert out.messages[1].content == "State:\n(empty)"

    def test_dict_state_renders_compact_json(self):
        state = {"severity": 3, "team": "payments"}
        rewriter = _rewriter()
        out = _transform(rewriter, **_choose_kwargs(state=state))
        assert (
            out.messages[1].content
            == f"State:\n{json.dumps(state, ensure_ascii=False)}"
        )

    def test_empty_description_falls_back_to_label(self):
        block = decisions._render_options_block(["A", "B"], ["real", ""])
        assert block == "A: real\nB: B"

    def test_none_description_falls_back_to_label(self):
        block = decisions._render_options_block(["A", "B"], ["real", None])
        assert block == "A: real\nB: B"

    def test_json_description_rendering(self):
        block = decisions._render_options_block(["A", "B"], [{"k": 1}, [1, 2]])
        assert block == 'A: {"k": 1}\nB: [1, 2]'

    def test_options_block_has_no_trailing_newline(self):
        block = decisions._render_options_block(["A", "B"], ["x", "y"])
        assert not block.endswith("\n")

    def test_helper_pool_is_prefix_of_carried_labels(self):
        """D5: the helper's label pool must be a prefix of the rule's labels."""
        config = _decider_config()
        rule_labels = config["transformations"][0]["labels"]
        assert rule_labels[: len(decisions._LABEL_POOL)] == decisions._LABEL_POOL


# ---------------------------------------------------------------------------
# Parameter handling
# ---------------------------------------------------------------------------


class TestParameters:
    """Logprobs sizing and pinned parameters."""

    def test_top_logprobs_sized_to_label_pool(self):
        rewriter = _rewriter()
        assert rewriter.parameters["logprobs"] is True
        # A-Z is 26 labels, capped at the deployment maximum of 20.
        assert rewriter.parameters["top_logprobs"] == 20

    def test_top_logprobs_small_pool(self):
        config = _decider_config()
        config["transformations"][0]["labels"] = ["A", "B", "C"]
        rewriter = IntrinsicsRewriter(config_dict=config)
        assert rewriter.parameters["top_logprobs"] == 3

    def test_configured_top_logprobs_not_clobbered(self):
        rewriter = _rewriter(
            parameters={
                "max_completion_tokens": 1,
                "temperature": 0.0,
                "top_logprobs": 7,
            }
        )
        assert rewriter.parameters["top_logprobs"] == 7

    def test_pinned_parameters_present(self):
        rewriter = _rewriter()
        assert rewriter.parameters["max_completion_tokens"] == 1
        assert rewriter.parameters["temperature"] == 0.0
        assert rewriter.extra_body_parameters == {
            "chat_template_kwargs": {"enable_thinking": False}
        }

    def test_likelihood_still_gets_default_top_logprobs(self):
        """Regression guard: non-decision logprob rules keep the old sizing."""
        config = {
            "model": None,
            "response_format": None,
            "transformations": [{"type": "likelihood", "input_path": ["score"]}],
            "instruction": None,
            "parameters": None,
            "sentence_boundaries": None,
        }
        rewriter = IntrinsicsRewriter(config_dict=config)
        assert rewriter.parameters["top_logprobs"] == 10


# ---------------------------------------------------------------------------
# State kwarg handling
# ---------------------------------------------------------------------------


class TestStateKwargs:
    """The reserved `state` kwarg and its error cases."""

    def test_state_kwarg_without_state_template_is_config_error(self):
        config = _decider_config()
        config["state_template"] = None
        rewriter = IntrinsicsRewriter(config_dict=config)
        with pytest.raises(ValueError, match="state_template"):
            _transform(rewriter, **_choose_kwargs())

    def test_state_placeholder_in_instruction_rejected_at_construction(self):
        config = _decider_config()
        config["instruction"] = "Q: {question} S: {state}"
        with pytest.raises(ValueError, match="reserved"):
            IntrinsicsRewriter(config_dict=config)

    def test_none_state_kwarg_is_valid_empty_state(self):
        rewriter = _rewriter()
        out = _transform(rewriter, **_choose_kwargs(state=None))
        assert out.messages[1].content == "State:\n(empty)"

    def test_state_kwarg_not_forwarded_to_instruction(self):
        """`state` is consumed by the state turn, never by the template."""
        rewriter = _rewriter()
        out = _transform(rewriter, **_choose_kwargs())
        # The instruction turn must contain no `{state}` formatting error;
        # it rendered the question and options only.
        assert "{state}" not in out.messages[-1].content


# ---------------------------------------------------------------------------
# System replacement
# ---------------------------------------------------------------------------


class TestSystemReplacement:
    """Drop all system messages, insert the trained one, warn once."""

    @pytest.fixture(autouse=True)
    def _reset_warn_once(self):
        """Clear the module-global warn-once set between tests."""
        from mellea.formatters.granite.intrinsics import input as intrinsics_input

        intrinsics_input._SYSTEM_REPLACEMENT_WARNED.clear()
        yield
        intrinsics_input._SYSTEM_REPLACEMENT_WARNED.clear()

    def test_incoming_system_replaced(self):
        rewriter = _rewriter()
        request = ChatCompletion(
            messages=[
                SystemMessage(content="You are a pirate."),
                UserMessage(content="hi"),
            ]
        )
        out = rewriter.transform(request, **_choose_kwargs())
        assert out.messages[0].role == "system"
        assert out.messages[0].content.startswith("You are a decision engine")
        assert all(m.role != "system" or m is out.messages[0] for m in out.messages)
        # Non-system messages survive ahead of the state turn.
        assert out.messages[1].content == "hi"

    def test_no_system_messages_still_inserts_trained_prompt(self):
        rewriter = _rewriter()
        out = _transform(rewriter, **_choose_kwargs())
        assert out.messages[0].role == "system"
        assert out.messages[0].content.startswith("You are a decision engine")

    def test_warns_once_per_replaced_prompt(self, caplog):
        rewriter = _rewriter()
        request = ChatCompletion(messages=[SystemMessage(content="You are a pirate.")])
        with caplog.at_level(logging.WARNING):
            rewriter.transform(request, **_choose_kwargs())
            rewriter.transform(
                ChatCompletion(messages=[SystemMessage(content="You are a pirate.")]),
                **_choose_kwargs(),
            )
        warnings = [
            r for r in caplog.records if "replaced a caller-supplied" in r.getMessage()
        ]
        assert len(warnings) == 1

    def test_no_warning_without_dropped_system(self, caplog):
        rewriter = _rewriter()
        with caplog.at_level(logging.WARNING):
            _transform(rewriter, **_choose_kwargs())
        assert not [
            r for r in caplog.records if "replaced a caller-supplied" in r.getMessage()
        ]


# ---------------------------------------------------------------------------
# Marker guard
# ---------------------------------------------------------------------------


class TestMarkerGuard:
    """The invocation marker must occur exactly once across final messages."""

    def test_marker_in_state_raises(self):
        rewriter = _rewriter()
        with pytest.raises(AloraActivationError, match="2 times"):
            _transform(
                rewriter,
                **_choose_kwargs(state="We follow ### DECISION conventions here."),
            )

    def test_marker_in_question_raises(self):
        rewriter = _rewriter()
        with pytest.raises(AloraActivationError, match="2 times"):
            _transform(
                rewriter, **_choose_kwargs(question="What does ### DECISION mean?")
            )

    def test_marker_in_options_raises(self):
        rewriter = _rewriter()
        kwargs = _choose_kwargs()
        kwargs["options_block"] = "A: uses ### DECISION markers"
        with pytest.raises(AloraActivationError, match="2 times"):
            _transform(rewriter, **kwargs)

    def test_missing_instruction_marker_raises(self):
        """A config whose instruction drops the marker entirely fails closed."""
        config = _decider_config()
        config["instruction"] = "Question:\n{question}\n\nOptions:\n{options_block}"
        rewriter = IntrinsicsRewriter(config_dict=config)
        with pytest.raises(AloraActivationError, match="0 times"):
            _transform(rewriter, **_choose_kwargs())

    def test_marker_error_is_value_error(self):
        rewriter = _rewriter()
        with pytest.raises(ValueError):
            _transform(
                rewriter, **_choose_kwargs(question="What does ### DECISION mean?")
            )


# ---------------------------------------------------------------------------
# Tools and documents rejection
# ---------------------------------------------------------------------------


class TestDecisionIncompatibilities:
    """Decision calls send no tools and reject documents."""

    def test_tools_rejected(self):
        rewriter = _rewriter()
        request = ChatCompletion(
            messages=[],
            tools=[ToolDefinition(name="lookup", description="Look things up.")],
        )
        with pytest.raises(ValueError, match="tools"):
            rewriter.transform(request, **_choose_kwargs())

    def test_documents_rejected(self):
        rewriter = _rewriter()
        request = ChatCompletion.model_validate(
            {
                "messages": [],
                "extra_body": {"documents": [Document(text="A document.", doc_id="0")]},
            }
        )
        with pytest.raises(ValueError, match="documents"):
            rewriter.transform(request, **_choose_kwargs())

    def test_empty_documents_allowed(self):
        rewriter = _rewriter()
        request = ChatCompletion.model_validate(
            {"messages": [], "extra_body": {"documents": []}}
        )
        out = rewriter.transform(request, **_choose_kwargs())
        assert len(out.messages) == 3


# ---------------------------------------------------------------------------
# Output processing: FirstTokenChoice and the raw-mode gate
# ---------------------------------------------------------------------------


def _logprobs_response(
    token: str = "A",
    logprob: float = -0.2,
    top_logprobs: list[tuple[str, float]] | None = None,
    n_choices: int = 1,
) -> ChatCompletionResponse:
    """Build a chat completion response carrying one-token logprobs."""
    content = ChatCompletionLogProbsContent(
        token=token,
        logprob=logprob,
        top_logprobs=[
            ChatCompletionLogProbsContent(token=t, logprob=lp)
            for t, lp in (top_logprobs or [])
        ],
    )
    choices = [
        ChatCompletionResponseChoice(
            index=i,
            message=AssistantMessage(content=token),
            logprobs=ChatCompletionLogProbs(content=[content]),
            finish_reason="length",
        )
        for i in range(n_choices)
    ]
    return ChatCompletionResponse(choices=choices)


def _processor(labels: list[str] | None = None) -> IntrinsicsResultProcessor:
    config = _decider_config()
    if labels is not None:
        config["transformations"][0]["labels"] = labels
    return IntrinsicsResultProcessor(config_dict=config)


class TestFirstTokenRule:
    """The `first_token` rule's readout mechanics."""

    def test_exact_token_match_excludes_prefixed_variant(self):
        processor = _processor(["A", "B"])
        resp = _logprobs_response("A", -0.2, [("A", -0.2), ("B", -1.7), (" A", -0.5)])
        out = json.loads(processor.transform(resp).choices[0].message.content)
        assert out["label_logprobs"] == {"A": -0.2, "B": -1.7}

    def test_top1_union_dedup(self):
        """The sampled top-1 also appearing in top_logprobs counts once."""
        processor = _processor(["A", "B"])
        resp = _logprobs_response("A", -0.3, [("A", -0.3), ("B", -1.9)])
        out = json.loads(processor.transform(resp).choices[0].message.content)
        assert out["label_logprobs"] == {"A": -0.3, "B": -1.9}

    def test_min_returned_logprob_over_all_returned(self):
        processor = _processor(["A", "B"])
        resp = _logprobs_response("A", -0.2, [("A", -0.2), ("B", -1.7), ("C", -6.4)])
        out = json.loads(processor.transform(resp).choices[0].message.content)
        assert out["min_returned_logprob"] == -6.4

    def test_zero_token_response_raises(self):
        processor = _processor(["A", "B"])
        resp = _logprobs_response("A")
        resp.choices[0].logprobs = ChatCompletionLogProbs(content=[])
        with pytest.raises(ValueError, match="no per-token content"):
            processor.transform(resp)

    def test_no_logprobs_raises(self):
        processor = _processor(["A", "B"])
        resp = _logprobs_response("A")
        resp.choices[0].logprobs = None
        with pytest.raises(ValueError, match="requires logprobs"):
            processor.transform(resp)

    def test_n_greater_than_one_rejected(self):
        processor = _processor(["A", "B"])
        resp = _logprobs_response("A", n_choices=2)
        with pytest.raises(ValueError, match="exactly one choice"):
            processor.transform(resp)

    def test_raw_mode_skips_json_parsing(self):
        """A one-token answer is not JSON; the gate must not parse it."""
        processor = _processor(["A", "B"])
        resp = _logprobs_response("B", -0.1, [("B", -0.1), ("A", -2.5)])
        out = processor.transform(resp)
        assert json.loads(out.choices[0].message.content)["token"] == "B"

    def test_first_token_must_be_first_rule(self):
        config = _decider_config()
        config["transformations"] = [
            {"type": "project", "input_path": ["$"], "retained_fields": ["x"]},
            {"type": "first_token", "labels": ["A", "B"]},
        ]
        with pytest.raises(ValueError, match="must be the first"):
            IntrinsicsResultProcessor(config_dict=config)

    def test_other_rules_still_require_input_path(self):
        config = _decider_config()
        config["transformations"] = [{"type": "explode", "target_field": "x"}]
        with pytest.raises(ValueError, match="requires an 'input_path'"):
            IntrinsicsResultProcessor(config_dict=config)

    def test_empty_label_match_is_not_a_rule_error(self):
        """Zero matched labels flows to the caller's label-mass check."""
        processor = _processor(["A", "B"])
        resp = _logprobs_response("Q", -0.1, [("Q", -0.1), ("the", -2.0)])
        out = json.loads(processor.transform(resp).choices[0].message.content)
        assert out["label_logprobs"] == {}
        assert out["token"] == "Q"

    def test_bad_labels_config_rejected(self):
        config = _decider_config()
        config["transformations"][0]["labels"] = "AB"
        with pytest.raises(TypeError, match="labels"):
            IntrinsicsResultProcessor(config_dict=config)
