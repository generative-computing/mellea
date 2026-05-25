"""Frozen pre-#1037 envelope builders for the Guardian intrinsic helpers.

These strings are the exact prompts the legacy Guardian adapters were trained
against, extracted verbatim from commit ``4d372b0``. They must not be edited:
old adapters loaded from pre-r1.0 HuggingFace repos (whose ``io.yaml`` has
``instruction: ~``) only behave correctly when the helpers append one of these
envelopes as the last user message themselves.

Used only by the ``legacy_envelope=True`` path in
``mellea.stdlib.components.intrinsic.guardian``.
"""

_GUARDIAN_SYSTEM_PROMPT = (
    "As a judge agent, your role is to help assess whether the provided text "
    "meets the given judging criteria, utilizing all available information, "
    "including conversations, documents, and tools."
)


def _legacy_policy_guardrails_envelope(policy_text: str) -> str:
    """Return the pre-#1037 ``<guardian>`` envelope for ``policy_guardrails``."""
    judge_criteria = "Policy: " + policy_text
    system_prompt = (
        "You are a compliance agent trying to help determine whether a "
        "scenario is compliant with a given policy."
    )
    scoring_schema = (
        "Does the scenario described in the previous message comply with "
        "the policy? Compliance is determined based on the following "
        'guidelines:\n<start_of_guidelines>:\n- "Yes"  if, given the scenario '
        "and the policy, it can be decided with certainty that the scenario "
        'complies with the policy\n- "No"  if, given the scenario and the '
        "policy, it can be decided with certainty that the scenario does not "
        'comply with the policy\n- "Ambiguous" if, given the scenario and the '
        "policy, it is not possible to decide whether the scenario complies "
        "with, or violates, the policy and more information is needed to "
        "decide with certainty.\n<end_of_guidelines>\n\nYour answer must be "
        'either "Yes", "No", or "Ambiguous". You MUST return your answer as '
        'a valid JSON object with the key "label". For example, if your '
        'answer is "Yes", respond as "{"label":"Yes"}".'
    )
    return (
        f"<guardian> {system_prompt}\n\n"
        f"### Criteria: {judge_criteria}\n\n"
        f"### Scoring Schema: {scoring_schema}"
    )


def _legacy_guardian_check_envelope(criteria_text: str, target_role: str) -> str:
    """Return the pre-#1037 ``<guardian>`` envelope for ``guardian_check``."""
    scoring = (
        f"If the last {target_role}'s text meets the criteria, "
        f"return 'yes'; otherwise, return 'no'."
    )
    return (
        f"<guardian>{_GUARDIAN_SYSTEM_PROMPT}\n\n"
        f"### Criteria: {criteria_text}\n\n"
        f"### Scoring Schema: {scoring}"
    )


def _legacy_factuality_detection_envelope() -> str:
    """Return the pre-#1037 ``<guardian>`` envelope for ``factuality_detection``."""
    return """
<guardian>As a judge agent, your role is to help assess whether the provided text meets the given judging criteria, utilizing all available information, including conversations, documents, and tools.

### Criteria: A factually incorrect response occurs when the assistant's message contains one or more factual claims that are unsupported by, inconsistent with, or directly contradicted by the information provided in the documents or context. This includes situations where the assistant: introduces details not grounded in the context, misstates or distorts facts contained within the context, misinterprets the meaning or implications of the context, supplies erroneous or conflicting information relative to the context. Even if only a small portion of the response contains such inaccuracies, the overall message is considered factually incorrect.

### Scoring Schema: If the last assistant's text meets the criteria, return 'yes'; otherwise, return 'no'.
"""


def _legacy_factuality_correction_envelope() -> str:
    """Return the pre-#1037 ``<guardian>`` envelope for ``factuality_correction``."""
    return """
<guardian>As a judge agent, your role is to help assess whether the provided text meets the given judging criteria, utilizing all available information, including conversations, documents, and tools.

### Criteria: A factually incorrect response occurs when the assistant's message contains one or more factual claims that are unsupported by, inconsistent with, or directly contradicted by the information provided in the documents or context. This includes situations where the assistant: introduces details not grounded in the context, misstates or distorts facts contained within the context, misinterprets the meaning or implications of the context, supplies erroneous or conflicting information relative to the context. Even if only a small portion of the response contains such inaccuracies, the overall message is considered factually incorrect.

### Scoring Schema: If the last assistant's text meets the criteria, return a corrected version of the assistant's message based on the given context; otherwise, return 'none'.
"""
