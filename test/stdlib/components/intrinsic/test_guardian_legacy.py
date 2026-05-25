"""Unit tests for the ``legacy_envelope`` backwards-compat path of guardian helpers.

The pre-#1037 helpers built a ``<guardian>``-prefixed envelope locally and
appended it as a user message before dispatch. The new helpers rely on the
``instruction:`` template in the adapter's ``io.yaml``. For users pinned to
older HuggingFace adapter checkouts (whose ``io.yaml`` has ``instruction: ~``),
the new path silently dispatches with no envelope at all — the model gets
garbage. ``legacy_envelope=True`` (or ``MELLEA_GUARDIAN_LEGACY_ENVELOPE=1``)
restores the legacy behavior.

These tests assert:

* The frozen envelope strings match the pre-#1037 byte-for-byte expectation.
* When ``legacy_envelope=True``, the helpers append the envelope as a user
  message and dispatch with empty kwargs.
* The env var fallback works.
* ``scoring_schema`` is rejected when combined with ``legacy_envelope=True``.

Tests are mock-based; no model is loaded.
"""

import pytest

from mellea.stdlib.components.chat import Message
from mellea.stdlib.components.intrinsic import guardian
from mellea.stdlib.components.intrinsic._guardian_legacy import (
    _legacy_factuality_correction_envelope,
    _legacy_factuality_detection_envelope,
    _legacy_guardian_check_envelope,
    _legacy_policy_guardrails_envelope,
)
from mellea.stdlib.context import ChatContext

# --- Frozen envelope strings (verbatim from pre-#1037 commit 4d372b0) ---

_JUDGE_SYSTEM_PROMPT = (
    "As a judge agent, your role is to help assess whether the provided text "
    "meets the given judging criteria, utilizing all available information, "
    "including conversations, documents, and tools."
)

_EXPECTED_GUARDIAN_CHECK = (
    f"<guardian>{_JUDGE_SYSTEM_PROMPT}\n\n"
    f"### Criteria: {guardian.CRITERIA_BANK['harm']}\n\n"
    f"### Scoring Schema: If the last assistant's text meets the criteria, "
    f"return 'yes'; otherwise, return 'no'."
)

_EXPECTED_GUARDIAN_CHECK_USER = (
    f"<guardian>{_JUDGE_SYSTEM_PROMPT}\n\n"
    f"### Criteria: {guardian.CRITERIA_BANK['harm']}\n\n"
    f"### Scoring Schema: If the last user's text meets the criteria, "
    f"return 'yes'; otherwise, return 'no'."
)

_EXPECTED_POLICY_GUARDRAILS = (
    "<guardian> You are a compliance agent trying to help determine whether "
    "a scenario is compliant with a given policy.\n\n"
    "### Criteria: Policy: no PII\n\n"
    "### Scoring Schema: Does the scenario described in the previous message "
    "comply with the policy? Compliance is determined based on the following "
    'guidelines:\n<start_of_guidelines>:\n- "Yes"  if, given the scenario and '
    "the policy, it can be decided with certainty that the scenario complies "
    'with the policy\n- "No"  if, given the scenario and the policy, it can '
    "be decided with certainty that the scenario does not comply with the "
    'policy\n- "Ambiguous" if, given the scenario and the policy, it is not '
    "possible to decide whether the scenario complies with, or violates, the "
    "policy and more information is needed to decide with certainty.\n"
    '<end_of_guidelines>\n\nYour answer must be either "Yes", "No", or '
    '"Ambiguous". You MUST return your answer as a valid JSON object with '
    'the key "label". For example, if your answer is "Yes", respond as '
    '"{"label":"Yes"}".'
)

_EXPECTED_FACTUALITY_DETECTION = """
<guardian>As a judge agent, your role is to help assess whether the provided text meets the given judging criteria, utilizing all available information, including conversations, documents, and tools.

### Criteria: A factually incorrect response occurs when the assistant's message contains one or more factual claims that are unsupported by, inconsistent with, or directly contradicted by the information provided in the documents or context. This includes situations where the assistant: introduces details not grounded in the context, misstates or distorts facts contained within the context, misinterprets the meaning or implications of the context, supplies erroneous or conflicting information relative to the context. Even if only a small portion of the response contains such inaccuracies, the overall message is considered factually incorrect.

### Scoring Schema: If the last assistant's text meets the criteria, return 'yes'; otherwise, return 'no'.
"""

_EXPECTED_FACTUALITY_CORRECTION = """
<guardian>As a judge agent, your role is to help assess whether the provided text meets the given judging criteria, utilizing all available information, including conversations, documents, and tools.

### Criteria: A factually incorrect response occurs when the assistant's message contains one or more factual claims that are unsupported by, inconsistent with, or directly contradicted by the information provided in the documents or context. This includes situations where the assistant: introduces details not grounded in the context, misstates or distorts facts contained within the context, misinterprets the meaning or implications of the context, supplies erroneous or conflicting information relative to the context. Even if only a small portion of the response contains such inaccuracies, the overall message is considered factually incorrect.

### Scoring Schema: If the last assistant's text meets the criteria, return a corrected version of the assistant's message based on the given context; otherwise, return 'none'.
"""


# --- Frozen-string contract guards ------------------------------------------


def test_legacy_guardian_check_envelope_assistant_matches_pre_1037():
    actual = _legacy_guardian_check_envelope(
        guardian.CRITERIA_BANK["harm"], "assistant"
    )
    assert actual == _EXPECTED_GUARDIAN_CHECK


def test_legacy_guardian_check_envelope_user_matches_pre_1037():
    actual = _legacy_guardian_check_envelope(guardian.CRITERIA_BANK["harm"], "user")
    assert actual == _EXPECTED_GUARDIAN_CHECK_USER


def test_legacy_policy_guardrails_envelope_matches_pre_1037():
    assert _legacy_policy_guardrails_envelope("no PII") == _EXPECTED_POLICY_GUARDRAILS


def test_legacy_factuality_detection_envelope_matches_pre_1037():
    assert _legacy_factuality_detection_envelope() == _EXPECTED_FACTUALITY_DETECTION


def test_legacy_factuality_correction_envelope_matches_pre_1037():
    assert _legacy_factuality_correction_envelope() == _EXPECTED_FACTUALITY_CORRECTION


# --- Routing tests with mock backend ----------------------------------------


@pytest.fixture
def capture_call(monkeypatch):
    """Spy ``call_intrinsic`` and capture (name, last context message, kwargs).

    Returns stub responses shaped like the real Guardian outputs so the helpers'
    return-value handling stays exercised.
    """
    captured: dict = {}

    def fake_call_intrinsic(name, context, backend, /, kwargs=None, model_options=None):
        captured["name"] = name
        captured["kwargs"] = kwargs
        # Pull the last message off the context, if any, so tests can assert
        # what envelope (if any) the helper appended.
        last_turn = context.last_turn()
        if last_turn is not None and isinstance(last_turn.model_input, Message):
            captured["last_message_content"] = last_turn.model_input.content
        else:
            captured["last_message_content"] = None
        # Return a shape compatible with whichever helper called us. Unwrap
        # to one of {label, score} for policy_guardrails (it forbids both),
        # while still satisfying guardian_check ("guardian" record),
        # factuality_detection ("score"), and factuality_correction
        # ("correction").
        if name == "policy-guardrails":
            return {"label": "Yes"}
        return {"guardian": {"score": 0.0}, "score": "no", "correction": "ok"}

    monkeypatch.setattr(guardian, "call_intrinsic", fake_call_intrinsic)
    return captured


def test_guardian_check_legacy_envelope_appends_message_and_empties_kwargs(
    capture_call,
):
    guardian.guardian_check(
        ChatContext(), object(), criteria="harm", legacy_envelope=True
    )
    assert capture_call["name"] == "guardian-core"
    assert capture_call["kwargs"] is None
    assert capture_call["last_message_content"] == _EXPECTED_GUARDIAN_CHECK


def test_guardian_check_legacy_envelope_with_user_target_role(capture_call):
    guardian.guardian_check(
        ChatContext(),
        object(),
        criteria="harm",
        target_role="user",
        legacy_envelope=True,
    )
    assert capture_call["last_message_content"] == _EXPECTED_GUARDIAN_CHECK_USER


def test_guardian_check_legacy_rejects_scoring_schema(capture_call):
    with pytest.raises(TypeError, match="scoring_schema is not supported"):
        guardian.guardian_check(
            ChatContext(),
            object(),
            criteria="harm",
            scoring_schema="user_prompt",
            legacy_envelope=True,
        )


def test_guardian_check_legacy_rejects_invalid_target_role(capture_call):
    with pytest.raises(ValueError, match="target_role must be"):
        guardian.guardian_check(
            ChatContext(),
            object(),
            criteria="harm",
            target_role="system",
            legacy_envelope=True,
        )


def test_policy_guardrails_legacy_envelope_appends_message_and_empties_kwargs(
    capture_call,
):
    guardian.policy_guardrails(
        ChatContext(), object(), policy_text="no PII", legacy_envelope=True
    )
    assert capture_call["name"] == "policy-guardrails"
    assert capture_call["kwargs"] is None
    assert capture_call["last_message_content"] == _EXPECTED_POLICY_GUARDRAILS


def test_factuality_detection_legacy_envelope_appends_message(capture_call):
    guardian.factuality_detection(ChatContext(), object(), legacy_envelope=True)
    assert capture_call["name"] == "factuality-detection"
    assert capture_call["kwargs"] is None
    assert capture_call["last_message_content"] == _EXPECTED_FACTUALITY_DETECTION


def test_factuality_correction_legacy_envelope_appends_message(capture_call):
    guardian.factuality_correction(ChatContext(), object(), legacy_envelope=True)
    assert capture_call["name"] == "factuality-correction"
    assert capture_call["kwargs"] is None
    assert capture_call["last_message_content"] == _EXPECTED_FACTUALITY_CORRECTION


# --- Default-path preservation ---------------------------------------------


def test_default_path_unchanged_for_guardian_check(capture_call):
    """Without legacy_envelope, helpers should still pass kwargs through."""
    guardian.guardian_check(ChatContext(), object(), criteria="harm")
    # New path passes criteria + scoring_schema as kwargs and does NOT add a
    # user message itself.
    assert capture_call["kwargs"] is not None
    assert "criteria" in capture_call["kwargs"]
    assert "scoring_schema" in capture_call["kwargs"]
    assert capture_call["last_message_content"] is None


def test_default_path_unchanged_for_policy_guardrails(capture_call):
    guardian.policy_guardrails(ChatContext(), object(), policy_text="no PII")
    assert capture_call["kwargs"] == {"policy_text": "no PII"}
    assert capture_call["last_message_content"] is None


def test_default_path_unchanged_for_factuality_detection(capture_call):
    guardian.factuality_detection(ChatContext(), object())
    assert capture_call["kwargs"] is None
    assert capture_call["last_message_content"] is None


# --- Env-var fallback -------------------------------------------------------


def test_env_var_enables_legacy_envelope(monkeypatch, capture_call):
    monkeypatch.setenv("MELLEA_GUARDIAN_LEGACY_ENVELOPE", "1")
    guardian.guardian_check(ChatContext(), object(), criteria="harm")
    assert capture_call["last_message_content"] == _EXPECTED_GUARDIAN_CHECK
    assert capture_call["kwargs"] is None


def test_env_var_unset_means_default_path(monkeypatch, capture_call):
    monkeypatch.delenv("MELLEA_GUARDIAN_LEGACY_ENVELOPE", raising=False)
    guardian.guardian_check(ChatContext(), object(), criteria="harm")
    assert capture_call["kwargs"] is not None
    assert capture_call["last_message_content"] is None


def test_explicit_kwarg_overrides_env_var(monkeypatch, capture_call):
    """legacy_envelope=False wins even when env var is set."""
    monkeypatch.setenv("MELLEA_GUARDIAN_LEGACY_ENVELOPE", "1")
    guardian.guardian_check(
        ChatContext(), object(), criteria="harm", legacy_envelope=False
    )
    assert capture_call["kwargs"] is not None
    assert capture_call["last_message_content"] is None


@pytest.mark.parametrize("value", ["true", "yes", "on", "TRUE", "On"])
def test_env_var_truthy_variants(monkeypatch, capture_call, value):
    monkeypatch.setenv("MELLEA_GUARDIAN_LEGACY_ENVELOPE", value)
    guardian.guardian_check(ChatContext(), object(), criteria="harm")
    assert capture_call["last_message_content"] == _EXPECTED_GUARDIAN_CHECK


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_env_var_falsy_variants(monkeypatch, capture_call, value):
    monkeypatch.setenv("MELLEA_GUARDIAN_LEGACY_ENVELOPE", value)
    guardian.guardian_check(ChatContext(), object(), criteria="harm")
    assert capture_call["last_message_content"] is None
