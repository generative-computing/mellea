"""Regression tests for the Guardian generic-path fix (issue #1017).

Before the overlay fix, the four Guardian helpers in
``mellea/stdlib/components/intrinsic/guardian.py`` constructed a
``<guardian>``-prefixed envelope in Python and appended it to the
``ChatContext`` as a new user message before calling ``call_intrinsic``. That
meant callers who skipped the helpers and used the generic ``Intrinsic(...)``
path sent a bare last turn to the adapter — with no system prompt, criteria,
or scoring schema — so scoring silently broke.

The fix moves the envelope into each intrinsic's ``io.yaml`` ``instruction``
field (shipped in-repo via the ``_overlays/`` mechanism) and the helpers just
forward resolved kwargs. These tests assert that, for each Guardian intrinsic:

* The overlay's ``instruction`` template, formatted with the kwargs the helper
  now passes, matches the exact string the *old* helper built.
* Feeding the overlay through :class:`IntrinsicsRewriter` produces the same
  message bytes as the old helper's ``context.add(Message("user", envelope))``.

Both tests are mock-based — no model is loaded. The point is wire format,
not scoring quality.
"""

import pathlib

import pytest
import yaml

from mellea.backends.adapters.adapter import _resolve_catalog_overlay
from mellea.backends.adapters.catalog import fetch_intrinsic_metadata
from mellea.formatters.granite.base.types import (
    AssistantMessage,
    ChatCompletion,
    UserMessage,
)
from mellea.formatters.granite.intrinsics.input import IntrinsicsRewriter


def _overlay_path(intrinsic_name: str) -> pathlib.Path | None:
    """Resolve the granite-4.1-3b lora overlay via the catalog."""
    metadata = fetch_intrinsic_metadata(intrinsic_name)
    return _resolve_catalog_overlay(metadata, "granite-4.1-3b", alora=False)


# --- expected envelope strings (copied verbatim from the pre-refactor helpers) ---

_JUDGE_SYSTEM_PROMPT = (
    "As a judge agent, your role is to help assess whether the provided text "
    "meets the given judging criteria, utilizing all available information, "
    "including conversations, documents, and tools."
)

_COMPLIANCE_SYSTEM_PROMPT = (
    "You are a compliance agent trying to help determine whether a scenario "
    "is compliant with a given policy."
)


def _expected_guardian_core_envelope(criteria: str, target_role: str) -> str:
    """Exactly what the pre-refactor ``guardian_check`` helper emitted."""
    scoring = (
        f"If the last {target_role}'s text meets the criteria, "
        f"return 'yes'; otherwise, return 'no'."
    )
    return (
        f"<guardian>{_JUDGE_SYSTEM_PROMPT}\n\n"
        f"### Criteria: {criteria}\n\n"
        f"### Scoring Schema: {scoring}"
    )


def _expected_policy_guardrails_envelope(policy_text: str) -> str:
    """Exactly what the pre-refactor ``policy_guardrails`` helper emitted."""
    judge_criteria = "Policy: " + policy_text
    scoring_schema = (
        "Does the scenario described in the previous message comply with "
        "the policy? Compliance is determined based on the following "
        'guidelines:\n<start_of_guidelines>:\n- "Yes"  if, given the '
        "scenario and the policy, it can be decided with certainty that "
        'the scenario complies with the policy\n- "No"  if, given the '
        "scenario and the policy, it can be decided with certainty that "
        'the scenario does not comply with the policy\n- "Ambiguous" if, '
        "given the scenario and the policy, it is not possible to decide "
        "whether the scenario complies with, or violates, the policy and "
        "more information is needed to decide with certainty.\n"
        '<end_of_guidelines>\n\nYour answer must be either "Yes", "No", '
        'or "Ambiguous". You MUST return your answer as a valid JSON '
        'object with the key "label". For example, if your answer is '
        '"Yes", respond as "{"label":"Yes"}".'
    )
    return (
        f"<guardian> {_COMPLIANCE_SYSTEM_PROMPT}\n\n"
        f"### Criteria: {judge_criteria}\n\n"
        f"### Scoring Schema: {scoring_schema}"
    )


# The factuality helpers were fully static; these are the verbatim strings
# they used to push into the context (leading newline preserved).
_EXPECTED_FACTUALITY_DETECTION_ENVELOPE = """
<guardian>As a judge agent, your role is to help assess whether the provided text meets the given judging criteria, utilizing all available information, including conversations, documents, and tools.

### Criteria: A factually incorrect response occurs when the assistant's message contains one or more factual claims that are unsupported by, inconsistent with, or directly contradicted by the information provided in the documents or context. This includes situations where the assistant: introduces details not grounded in the context, misstates or distorts facts contained within the context, misinterprets the meaning or implications of the context, supplies erroneous or conflicting information relative to the context. Even if only a small portion of the response contains such inaccuracies, the overall message is considered factually incorrect.

### Scoring Schema: If the last assistant's text meets the criteria, return 'yes'; otherwise, return 'no'.
"""

_EXPECTED_FACTUALITY_CORRECTION_ENVELOPE = """
<guardian>As a judge agent, your role is to help assess whether the provided text meets the given judging criteria, utilizing all available information, including conversations, documents, and tools.

### Criteria: A factually incorrect response occurs when the assistant's message contains one or more factual claims that are unsupported by, inconsistent with, or directly contradicted by the information provided in the documents or context. This includes situations where the assistant: introduces details not grounded in the context, misstates or distorts facts contained within the context, misinterprets the meaning or implications of the context, supplies erroneous or conflicting information relative to the context. Even if only a small portion of the response contains such inaccuracies, the overall message is considered factually incorrect.

### Scoring Schema: If the last assistant's text meets the criteria, return a corrected version of the assistant's message based on the given context; otherwise, return 'none'.
"""

# --- helpers ---


def _load_overlay_instruction(intrinsic_name: str) -> str:
    path = _overlay_path(intrinsic_name)
    assert path is not None, f"no overlay for {intrinsic_name}"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    instruction = cfg.get("instruction")
    assert instruction is not None, (
        f"{intrinsic_name}: overlay has no instruction field"
    )
    return instruction


def _base_context() -> ChatCompletion:
    """A minimal user/assistant exchange for driving the rewriter."""
    return ChatCompletion(
        messages=[
            UserMessage(content="Can you help me with something?"),
            AssistantMessage(content="Of course — what do you need?"),
        ]
    )


# --- template-level tests: the overlay substitutes to the old helper output ---


def test_guardian_core_template_matches_old_helper():
    template = _load_overlay_instruction("guardian-core")
    # Kwargs the new helper passes to call_intrinsic
    criteria = "example criteria text"
    target_role = "assistant"
    produced = template.format(criteria=criteria, target_role=target_role)
    assert produced == _expected_guardian_core_envelope(criteria, target_role)


def test_policy_guardrails_template_matches_old_helper():
    template = _load_overlay_instruction("policy-guardrails")
    policy_text = "example policy text"
    produced = template.format(policy_text=policy_text)
    assert produced == _expected_policy_guardrails_envelope(policy_text)


def test_factuality_detection_template_matches_old_helper():
    template = _load_overlay_instruction("factuality-detection")
    assert template == _EXPECTED_FACTUALITY_DETECTION_ENVELOPE


def test_factuality_correction_template_matches_old_helper():
    template = _load_overlay_instruction("factuality-correction")
    assert template == _EXPECTED_FACTUALITY_CORRECTION_ENVELOPE


# --- rewriter-level tests: drive IntrinsicsRewriter with the overlay yaml ---


def _rewriter_last_message(intrinsic_name: str, **kwargs: str) -> str:
    """Build a rewriter from the overlay yaml and return the content of the
    user message the rewriter appends to a minimal two-turn context.
    """
    yaml_path = _overlay_path(intrinsic_name)
    assert yaml_path is not None
    rewriter = IntrinsicsRewriter(config_file=yaml_path)
    before = _base_context()
    after = rewriter.transform(before, **kwargs)
    return after.messages[-1].content


def test_rewriter_guardian_core_appends_expected_message():
    criteria = "example criteria text"
    target_role = "assistant"
    content = _rewriter_last_message(
        "guardian-core", criteria=criteria, target_role=target_role
    )
    assert content == _expected_guardian_core_envelope(criteria, target_role)


def test_rewriter_policy_guardrails_appends_expected_message():
    policy_text = "example policy text"
    content = _rewriter_last_message("policy-guardrails", policy_text=policy_text)
    assert content == _expected_policy_guardrails_envelope(policy_text)


def test_rewriter_factuality_detection_appends_expected_message():
    content = _rewriter_last_message("factuality-detection")
    assert content == _EXPECTED_FACTUALITY_DETECTION_ENVELOPE


def test_rewriter_factuality_correction_appends_expected_message():
    content = _rewriter_last_message("factuality-correction")
    assert content == _EXPECTED_FACTUALITY_CORRECTION_ENVELOPE


# --- invariant: the rewriter appends (never replaces) the last message ---


def test_rewriter_preserves_preexisting_conversation():
    """The envelope is appended as a new user message; history is untouched."""
    yaml_path = _overlay_path("guardian-core")
    rewriter = IntrinsicsRewriter(config_file=yaml_path)
    before = _base_context()
    after = rewriter.transform(before, criteria="c", target_role="assistant")
    assert len(after.messages) == len(before.messages) + 1
    for i in range(len(before.messages)):
        assert after.messages[i].content == before.messages[i].content
        assert after.messages[i].role == before.messages[i].role
    assert after.messages[-1].role == "user"


# --- generic Intrinsic(...) path: prove the wire format for the path the
# issue actually asks to fix (callers who skip the helpers and go direct
# through ``mfuncs.act(Intrinsic(name, intrinsic_kwargs=...), ...)``).
#
# The backends (HF and OpenAI) both build a dict-shaped ``request_json``
# (``messages`` / ``extra_body`` / ``tools``) and call
# ``rewriter.transform(request_json, **action.intrinsic_kwargs)``. We
# replicate that shape here, drive the overlay-loaded rewriter with the
# kwargs a caller would pass to ``Intrinsic``, and assert the final message
# bytes match the old helper's envelope. No model is loaded.


from mellea.backends.adapters.catalog import AdapterType
from mellea.stdlib.components.intrinsic.intrinsic import Intrinsic


def _dict_request() -> dict:
    """The dict shape the backends build before calling ``rewriter.transform``."""
    return {
        "messages": [
            {"role": "user", "content": "Can you help me with something?"},
            {"role": "assistant", "content": "Of course — what do you need?"},
        ],
        "extra_body": {"documents": []},
        "tools": None,
    }


def _generic_path_last_message(intrinsic_name: str, **intrinsic_kwargs: str) -> str:
    """Drive the generic ``Intrinsic(...)`` path and return the appended user message content.

    Mirrors what ``LocalHFBackend._generate_from_context_with_adapter`` /
    ``OpenAIBackend`` do: build the rewriter from the adapter's ``io.yaml``
    (here, the overlay), then call ``transform`` with ``action.intrinsic_kwargs``.
    """
    yaml_path = _overlay_path(intrinsic_name)
    assert yaml_path is not None
    rewriter = IntrinsicsRewriter(config_file=yaml_path)

    intrinsic = Intrinsic(
        intrinsic_name,
        intrinsic_kwargs=intrinsic_kwargs or None,
        adapter_types=(AdapterType.LORA,),
    )

    rewritten = rewriter.transform(_dict_request(), **intrinsic.intrinsic_kwargs)
    return rewritten.messages[-1].content


def test_generic_path_guardian_core_matches_old_helper_envelope():
    """``Intrinsic("guardian-core", intrinsic_kwargs=...)`` produces the pre-refactor envelope."""
    criteria = "example criteria text"
    target_role = "assistant"
    content = _generic_path_last_message(
        "guardian-core", criteria=criteria, target_role=target_role
    )
    assert content == _expected_guardian_core_envelope(criteria, target_role)


def test_generic_path_policy_guardrails_matches_old_helper_envelope():
    """``Intrinsic("policy-guardrails", intrinsic_kwargs=...)`` produces the pre-refactor envelope."""
    policy_text = "example policy text"
    content = _generic_path_last_message("policy-guardrails", policy_text=policy_text)
    assert content == _expected_policy_guardrails_envelope(policy_text)


def test_generic_path_factuality_detection_matches_old_helper_envelope():
    """``Intrinsic("factuality-detection")`` produces the pre-refactor envelope — no kwargs."""
    content = _generic_path_last_message("factuality-detection")
    assert content == _EXPECTED_FACTUALITY_DETECTION_ENVELOPE


def test_generic_path_factuality_correction_matches_old_helper_envelope():
    """``Intrinsic("factuality-correction")`` produces the pre-refactor envelope — no kwargs."""
    content = _generic_path_last_message("factuality-correction")
    assert content == _EXPECTED_FACTUALITY_CORRECTION_ENVELOPE
