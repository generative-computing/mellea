"""Unit tests for the ``transformations`` pipelines in shipped overlays.

The companion file ``test_adapter_overlays.py`` proves each overlay file is
discoverable, ships in the wheel, and loads through ``IntrinsicAdapter``. This
file goes one step further and exercises the *behaviour* declared in each
overlay's ``transformations`` block: it loads the overlay yaml, instantiates
``IntrinsicsResultProcessor`` directly, feeds it a synthetic
``ChatCompletionResponse`` with logprobs that match a known model output, and
asserts the post-transform JSON matches what the helper functions in
``mellea.stdlib.components.intrinsic`` are documented to return.

Three overlays today have non-trivial transformation chains:

* ``uncertainty``      — likelihood (digit → float) + project to ``certainty``
* ``requirement-check`` — likelihood (yes/no → 1/0) + nest under ``requirement_check``
* ``guardian-core``    — likelihood (yes/no → 1/0) + nest under ``guardian``

The other Guardian overlays declare ``transformations: ~`` and have no output
pipeline to exercise — they are covered by ``test_guardian_generic_path.py``
on the input side.
"""

import json

import pytest
import yaml

from mellea.backends.adapters.adapter import _resolve_catalog_overlay
from mellea.backends.adapters.catalog import fetch_intrinsic_metadata
from mellea.formatters.granite import IntrinsicsResultProcessor
from mellea.formatters.granite.base.types import (
    AssistantMessage,
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
)


def _overlay_config(intrinsic_name: str) -> dict:
    """Load the granite-4.1-3b lora overlay yaml for the given intrinsic."""
    metadata = fetch_intrinsic_metadata(intrinsic_name)
    path = _resolve_catalog_overlay(metadata, "granite-4.1-3b", alora=False)
    assert path is not None, f"no granite-4.1-3b lora overlay for {intrinsic_name}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _score_response(score_token: str) -> ChatCompletionResponse:
    """Build a synthetic response shaped like ``{"score": "<token>"}``.

    The ``likelihood`` rule reads logprobs to compute an expected value, so
    each token is emitted with logprob ~0 (probability ~1) and the picked
    token dominates the weighted sum.
    """
    content = f'{{"score": "{score_token}"}}'
    # Tokenize as: '{"score": "' | '<score_token>' | '"}'
    tokens = ['{"score": "', score_token, '"}']
    logprob_content = [
        ChatCompletionLogProbsContent(
            token=tok,
            logprob=-0.001,
            top_logprobs=[ChatCompletionLogProb(token=tok, logprob=-0.001)],
        )
        for tok in tokens
    ]
    return ChatCompletionResponse(
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=AssistantMessage(content=content),
                logprobs=ChatCompletionLogProbs(content=logprob_content),
                finish_reason="stop",
            )
        ]
    )


def _run_overlay(intrinsic_name: str, score_token: str) -> dict:
    """Run the overlay's full pipeline against ``{"score": "<token>"}`` and return parsed JSON."""
    config = _overlay_config(intrinsic_name)
    processor = IntrinsicsResultProcessor(config_dict=config)
    response = _score_response(score_token)
    transformed = processor.transform(response)
    transformed_content = transformed.choices[0].message.content
    assert transformed_content is not None
    return json.loads(transformed_content)


# ---------------------------------------------------------------------------
# uncertainty: digits → 0.05/0.15/.../0.95 → projected to {"certainty": <float>}
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("digit,expected", [("0", 0.05), ("5", 0.55), ("9", 0.95)])
def test_uncertainty_overlay_maps_digit_to_certainty_float(digit, expected):
    """The uncertainty overlay maps digit N to 0.1*N + 0.05 and projects to ``certainty``."""
    parsed = _run_overlay("uncertainty", digit)
    assert "certainty" in parsed
    assert parsed["certainty"] == pytest.approx(expected, abs=1e-3)
    # The original ``score`` field is dropped by the project transform.
    assert "score" not in parsed


# ---------------------------------------------------------------------------
# requirement-check: yes/no → 1.0/0.0 → nested under "requirement_check"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label,expected", [("yes", 1.0), ("no", 0.0)])
def test_requirement_check_overlay_maps_label_to_score(label, expected):
    """The requirement-check overlay maps yes/no to 1.0/0.0 nested under ``requirement_check``."""
    parsed = _run_overlay("requirement-check", label)
    assert "requirement_check" in parsed
    inner = parsed["requirement_check"]
    # ``nest`` wraps the existing record (``{"score": <float>}``) under the field.
    assert isinstance(inner, dict)
    assert inner["score"] == pytest.approx(expected, abs=1e-3)


# ---------------------------------------------------------------------------
# guardian-core: yes/no → 1.0/0.0 → nested under "guardian"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label,expected", [("yes", 1.0), ("no", 0.0)])
def test_guardian_core_overlay_maps_label_to_score(label, expected):
    """The guardian-core overlay maps yes/no to 1.0/0.0 nested under ``guardian``."""
    parsed = _run_overlay("guardian-core", label)
    assert "guardian" in parsed
    inner = parsed["guardian"]
    assert isinstance(inner, dict)
    assert inner["score"] == pytest.approx(expected, abs=1e-3)


# ---------------------------------------------------------------------------
# Input side: requirement-check renders {requirement} into its instruction
# (mirrors test_guardian_generic_path's coverage of the Guardian overlays).
# ---------------------------------------------------------------------------


def test_requirement_check_overlay_instruction_substitutes_kwarg():
    """The requirement-check overlay's instruction template formats the ``requirement`` kwarg."""
    config = _overlay_config("requirement-check")
    instruction = config["instruction"]
    rendered = instruction.format(requirement="response must be polite")
    assert "<requirements>: response must be polite" in rendered
    # Sanity: the canonical scoring schema sentence is preserved.
    assert '{"score": "yes"}' in rendered
    assert '{"score": "no"}' in rendered
