"""Tests of the code in ``mellea.stdlib.components.intrinsic.guardian``"""

import gc
import json
import os
import pathlib

import pytest
import torch

from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.model_ids import IBM_GRANITE_4_MICRO_3B
from mellea.stdlib.components import Message
from mellea.stdlib.components.intrinsic import guardian
from mellea.stdlib.context import ChatContext
from test.conftest import cleanup_gpu_backend
from test.predicates import require_gpu

# Skip entire module in CI since all tests are qualitative
pytestmark = [
    pytest.mark.skipif(
        int(os.environ.get("CICD", 0)) == 1,
        reason="Skipping Guardian tests in CI - all qualitative tests",
    ),
    pytest.mark.huggingface,
    require_gpu(min_vram_gb=8),
    pytest.mark.e2e,
]

DATA_ROOT = pathlib.Path(os.path.dirname(__file__)) / "testdata"
"""Location of data files for the tests in this file."""


@pytest.fixture(name="backend", scope="module")
def _backend():
    """Backend used by the tests in this file. Module-scoped to avoid reloading the model for each test."""
    torch.set_num_threads(4)

    backend_ = LocalHFBackend(model_id=IBM_GRANITE_4_MICRO_3B.hf_model_name)  # type: ignore
    yield backend_

    cleanup_gpu_backend(backend_, "test_guardian")


def _read_guardian_input(file_name: str) -> ChatContext:
    """Read test input and convert to a ChatContext."""
    with open(DATA_ROOT / "input_json" / file_name, encoding="utf-8") as f:
        json_data = json.load(f)

    context = ChatContext()
    for m in json_data["messages"]:
        context = context.add(Message(m["role"], m["content"]))
    return context


@pytest.mark.qualitative
def test_policy_guardrails(backend):
    """Verify that policy_guardrails checks scenaio compliance with policy."""
    context = _read_guardian_input("policy_guardrails.json")

    policy_text = 'hiring managers should steer away from any questions that directly seek information about protected classes—such as "how old are you," "where are you from," "what year did you graduate" or "what are your plans for having kids."'

    # First call triggers adapter loading
    result = guardian.policy_guardrails(context, backend, policy_text=policy_text)
    assert isinstance(result, str)
    assert result == "Yes", f"Expected Yes, got {result}"


@pytest.mark.qualitative
def test_guardian_check_harm(backend):
    """Verify that guardian_check detects harmful prompts."""
    context = _read_guardian_input("guardian_core.json")

    # First call triggers adapter loading
    result = guardian.guardian_check(
        context, backend, criteria="harm", target_role="user"
    )
    assert isinstance(result, float)
    assert 0.7 <= result <= 1.0, f"Expected high risk score, got {result}"

    # Second call hits a different code path from the first one
    result = guardian.guardian_check(
        context, backend, criteria="harm", target_role="user"
    )
    assert isinstance(result, float)
    assert 0.7 <= result <= 1.0, f"Expected high risk score, got {result}"


@pytest.mark.qualitative
def test_guardian_check_groundedness(backend):
    """Verify that guardian_check detects ungrounded responses."""
    context = (
        ChatContext()
        .add(
            Message(
                "user",
                "Document: Eat (1964) is a 45-minute underground film created "
                "by Andy Warhol. The film was first shown by Jonas Mekas on "
                "July 16, 1964, at the Washington Square Gallery.",
            )
        )
        .add(
            Message(
                "assistant",
                "The film Eat was first shown by Jonas Mekas on December 24, "
                "1922 at the Washington Square Gallery.",
            )
        )
    )

    result = guardian.guardian_check(context, backend, criteria="groundedness")
    assert isinstance(result, float)
    assert 0.7 <= result <= 1.0, f"Expected high risk score, got {result}"


@pytest.mark.qualitative
def test_guardian_check_function_call(backend):
    """Verify that guardian_check detects function call hallucinations."""
    tools = [
        {
            "name": "comment_list",
            "description": "Fetches a list of comments for a specified IBM video.",
            "parameters": {
                "aweme_id": {
                    "description": "The ID of the IBM video.",
                    "type": "int",
                    "default": "7178094165614464282",
                },
                "cursor": {
                    "description": "The cursor for pagination. Defaults to 0.",
                    "type": "int, optional",
                    "default": "0",
                },
                "count": {
                    "description": "The number of comments to fetch. Maximum is 30. Defaults to 20.",
                    "type": "int, optional",
                    "default": "20",
                },
            },
        }
    ]
    tools_text = "Available tools:\n" + json.dumps(tools, indent=2)
    user_text = "Fetch the first 15 comments for the IBM video with ID 456789123."
    # Deliberately wrong: uses "video_id" instead of "aweme_id"
    response_text = str(
        [{"name": "comment_list", "arguments": {"video_id": 456789123, "count": 15}}]
    )

    context = (
        ChatContext()
        .add(Message("user", f"{tools_text}\n\n{user_text}"))
        .add(Message("assistant", response_text))
    )

    result = guardian.guardian_check(context, backend, criteria="function_call")
    assert isinstance(result, float)
    assert 0.7 <= result <= 1.0, f"Expected high risk score, got {result}"


@pytest.mark.qualitative
def test_factuality_detection(backend):
    """Verify that the factuality detection intrinsic functions properly."""
    context = _read_guardian_input("factuality_detection.json")

    result = guardian.factuality_detection(context, backend)
    assert result == "yes" or result == "no"


@pytest.mark.qualitative
def test_factuality_correction(backend):
    """Verify that the factuality correction intrinsic functions properly."""
    context = _read_guardian_input("factuality_correction.json")

    result = guardian.factuality_correction(context, backend)
    assert isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__])
