"""Testing functions for budget forcing generation."""

import pytest

from mellea import MelleaSession, start_session
from mellea.backends import ModelOption
from mellea.backends.model_ids import OPENAI_GPT_OSS_20B
from mellea.core import CBlock
from mellea.stdlib.sampling.budget_forcing import BudgetForcingSamplingStrategy

MODEL_ID = OPENAI_GPT_OSS_20B

# Module-level markers: gpt-oss:20b is a 20B model requiring heavy resources
pytestmark = [
    pytest.mark.ollama,
    pytest.mark.requires_heavy_ram,
    pytest.mark.llm,
    pytest.mark.qualitative,
]


@pytest.fixture(scope="module")
def m_session(gh_run):
    """Start default Mellea's session."""
    if gh_run == 1:  # on github
        m = start_session(
            "ollama", model_id=MODEL_ID, model_options={ModelOption.MAX_NEW_TOKENS: 5}
        )
    else:
        m = start_session("ollama", model_id=MODEL_ID)
    yield m
    del m


def test_think_big(m_session: MelleaSession, gh_run: int):
    """Tests big thinking budget."""
    # if on github we can run big thinking mode
    if gh_run == 1:
        pytest.skip("Skipping big_thinking runs in gh workflows.")

    prompt = "What is the smallest positive integer $n$ such that all the roots of $z^4 + z^2 + 1 = 0$ are $n^{\\text{th}}$ roots of unity?"
    prompt_suffix = "\nPlease reason step by step, use \n\n to end each step, and put your final answer within \\boxed{}."
    action = CBlock(value=prompt + prompt_suffix)
    THINK_MAX_TOKENS = 2048
    ANSWER_MAX_TOKENS = 512

    strategy = BudgetForcingSamplingStrategy(
        think_max_tokens=THINK_MAX_TOKENS,
        answer_max_tokens=ANSWER_MAX_TOKENS,
        start_think_token="<think>",
        end_think_token="</think>",
        think_more_suffix="\nWait, let's think more carefully",
        answer_suffix="The final answer is:",
        requirements=None,
    )
    result = m_session.instruct(action, strategy=strategy)  # type: ignore

    print("\n******\nThink big:")
    print(str(result))


def test_think_little(m_session: MelleaSession, gh_run: int):
    """Tests little thinking budget."""
    prompt = "Compute 1+1?"
    prompt_suffix = "\nPlease reason step by step, use \n\n to end each step, and put your final answer within \\boxed{}."
    action = CBlock(value=prompt + prompt_suffix)
    THINK_MAX_TOKENS = 16
    ANSWER_MAX_TOKENS = 8
    if gh_run == 1:  # on github
        THINK_MAX_TOKENS = 0
        ANSWER_MAX_TOKENS = 5

    strategy = BudgetForcingSamplingStrategy(
        think_max_tokens=THINK_MAX_TOKENS,
        answer_max_tokens=ANSWER_MAX_TOKENS,
        start_think_token="<think>",
        end_think_token="</think>",
        answer_suffix="The final answer is: \\boxed{",
        requirements=None,
    )
    result = m_session.instruct(action, strategy=strategy)  # type: ignore

    print("\n******\nThink little:")
    print(str(result))


if __name__ == "__main__":
    pytest.main(["-s", __file__])
