# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden-layout test for the decision capability: byte-level parity with the reference.

Renders the rewriter's messages through the real Granite Switch chat
template (staged, not vendored: the model repos are private) and
byte-compares with the reference implementation's render for fixed cases.
This is the standing drift detector for the mellea-carried decider config:
any divergence in the trained layout (system prompt, state turn, marker,
question/options block) fails here with no GPU and no served model.

The staged artefacts are located via the `MELLEA_DECISIONS_TEST_ARTEFACTS`
environment variable, pointing at a directory containing the switch chat
template (`8b-chat_template.jinja`) and the published decision io.yamls
(`8b-decision-io.yaml`). The test
skips cleanly when the variable is unset or the files are missing.
"""

import json
import os
import pathlib

import jinja2
import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from mellea.backends.adapters.adapter import _apply_carried_decider_config
from mellea.formatters.granite.base.types import ChatCompletion
from mellea.formatters.granite.intrinsics.input import IntrinsicsRewriter
from mellea.stdlib.components.intrinsic import decisions

_ARTEFACTS_DIR: str = os.environ.get("MELLEA_DECISIONS_TEST_ARTEFACTS", "")
_TEMPLATE_NAME = "8b-chat_template.jinja"
_PUBLISHED_IO_YAMLS = ("8b-decision-io.yaml",)

pytestmark = pytest.mark.skipif(
    not _ARTEFACTS_DIR or not (pathlib.Path(_ARTEFACTS_DIR) / _TEMPLATE_NAME).is_file(),
    reason=(
        "Golden-layout decision artefacts not staged: set "
        "MELLEA_DECISIONS_TEST_ARTEFACTS to a directory containing the "
        f"staged switch chat template ({_TEMPLATE_NAME}) and the published "
        "decision io.yamls to run this drift detector."
    ),
)


def _load_template() -> jinja2.Template:
    """Load the staged switch chat template with transformers' render settings."""
    template_text = (pathlib.Path(_ARTEFACTS_DIR) / _TEMPLATE_NAME).read_text(
        encoding="utf8"
    )
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    return env.from_string(template_text)


def _load_published_config(filename: str) -> dict:
    """Parse a staged published io.yaml."""
    # Third Party
    import yaml

    path = pathlib.Path(_ARTEFACTS_DIR) / filename
    with open(path, encoding="utf8") as f:
        return yaml.safe_load(f)


def _reference_messages(
    state: object, question: str, labels: list[str], descriptions: list[object]
) -> list[dict]:
    """Reimplement the reference implementation's message construction.

    Mirrors `render_prompt_alora` from the model team's reference script
    (the model team's reference decider script): a fixed system prompt, a first user turn with the
    state, then a second user turn opening with the invocation marker plus
    question and options.
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


def _mellea_messages(
    published_config: dict,
    question: str,
    labels: list[str],
    descriptions: list[object],
    state: object,
) -> list[dict]:
    """Build the rewriter's final messages for a decision call.

    Runs the full client-side pipeline the way a backend does: the carried
    decider config is merged into the published io.yaml, the rewriter is
    constructed from the merged config, and the helper's kwargs are
    transformed.
    """
    name = published_config.get("name") or "decision"
    merged, _reason = _apply_carried_decider_config(name, published_config)
    assert merged is not None, "golden fixture must produce a valid merged config"
    rewriter = IntrinsicsRewriter(config_dict=merged, model_name=name)
    kwargs = {
        "question": question,
        "options_block": decisions._render_options_block(labels, descriptions),
        "state": state,
    }
    out = rewriter.transform(ChatCompletion(messages=[]), **kwargs)
    return [{"role": m.role, "content": m.content} for m in out.messages]


# Fixed cases: all three types, empty state, dict state, empty description.
_GOLDEN_CASES = [
    pytest.param(
        "choose",
        "My payouts have failed three times. The bank says everything is "
        "fine. Can someone please fix this?",
        "Which team should handle this ticket?",
        ["A", "B", "C"],
        [
            "Payout failures and payment processing",
            "Login and account access",
            "Something else",
        ],
        id="choose-reference-example",
    ),
    pytest.param(
        "decide",
        "My payouts have failed three times.",
        "Does this message require urgent human attention?",
        ["A", "B"],
        ["No / false", "Yes / true"],
        id="decide-default-descriptions",
    ),
    pytest.param(
        "decide",
        "My payouts have failed three times.",
        "Does this message require urgent human attention?",
        ["A", "B"],
        ["Routine handling is fine", "Same-day human action is required"],
        id="decide-custom-descriptions",
    ),
    pytest.param(
        "score",
        "A customer cannot log in.",
        "How urgent is this?",
        ["A", "B", "C"],
        ["can wait a week", "this week", "today"],
        id="score-three-levels",
    ),
    pytest.param(
        "choose",
        "",
        "Which team handles this?",
        ["A", "B"],
        ["Payments", "Account"],
        id="empty-state-sentinel",
    ),
    pytest.param(
        "choose",
        {"severity": 3, "channel": "chat"},
        "Which team handles this?",
        ["A", "B"],
        ["Payments", "Account"],
        id="dict-state-json",
    ),
    pytest.param(
        "choose",
        "A ticket arrived.",
        "Which team handles this?",
        ["A", "B", "C"],
        ["Payments", "", "Something else"],
        id="empty-description-fallback",
    ),
]


class TestGoldenLayout:
    """Byte-compare mellea's rendered prompt with the reference's."""

    @pytest.mark.parametrize(
        ("kind", "state", "question", "labels", "descriptions"), _GOLDEN_CASES
    )
    def test_render_matches_reference(
        self, kind, state, question, labels, descriptions
    ):
        template = _load_template()
        for io_yaml in _PUBLISHED_IO_YAMLS:
            published = _load_published_config(io_yaml)
            reference = _reference_messages(state, question, labels, descriptions)
            mellea = _mellea_messages(published, question, labels, descriptions, state)
            # The message lists must be identical before any rendering.
            assert mellea == reference, f"message lists differ for {io_yaml}"
            rendered_reference = template.render(
                messages=reference, add_generation_prompt=True, enable_thinking=False
            )
            rendered_mellea = template.render(
                messages=mellea, add_generation_prompt=True, enable_thinking=False
            )
            assert rendered_mellea == rendered_reference, (
                f"rendered prompts differ for {io_yaml}:\n"
                f"reference: {rendered_reference!r}\n"
                f"mellea:    {rendered_mellea!r}"
            )

    def test_control_token_appears_exactly_once(self):
        """The serving-side activation path swaps the marker for the control token."""
        template = _load_template()
        published = _load_published_config(_PUBLISHED_IO_YAMLS[0])
        state = "My payouts have failed three times."
        labels = ["A", "B"]
        descriptions = ["Payout failures", "Account access"]
        messages = _mellea_messages(
            published, "Which team?", labels, descriptions, state
        )
        rendered = template.render(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=False,
            adapter_name="decision",
        )
        # Pre-swap marker is replaced by the control token + shortened marker.
        assert rendered.count("<|decision|>") == 1
        assert "### DECISION" not in rendered
        # The rendered prompt still opens the assistant turn for the answer.
        assert rendered.endswith("assistant<|end_of_role|>")
