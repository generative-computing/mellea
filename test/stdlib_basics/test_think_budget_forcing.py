"""Testing functions for budget forcing generation."""

import pytest
from transformers import AutoTokenizer

from mellea import MelleaSession, start_session
from mellea.backends import ModelOption
from mellea.backends.model_ids import OPENAI_GPT_OSS_20B
from mellea.stdlib.base import CBlock
from mellea.stdlib.sampling_algos.budget_forcing_alg import think_budget_forcing

MODEL_ID = OPENAI_GPT_OSS_20B


@pytest.fixture(scope="module")
def m_session(gh_run):
    """Start default Mellea's session."""
    if gh_run == 1:  # on github
        m = start_session(
            "ollama",
            model_id=MODEL_ID,
            model_options={ModelOption.MAX_NEW_TOKENS: 5},
        )
    else:
        m = start_session(
            "ollama",
            model_id=MODEL_ID,
        )
    yield m
    del m


def prepare_prmpt_for_math(query):
    """Formats query for math task."""
    # Preparing prompt for math reasoning tasks
    system_prompt = None  # Use default of chat template
    prompt_suffix = "\nPlease reason step by step, use \n\n to end each step, and put your final answer within \\boxed{}."

    if prompt_suffix:
        query += prompt_suffix

    msg = []
    if system_prompt is not None:
        msg.append({"role": "system", "content": system_prompt})

    msg.append({"role": "user", "content": query})
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID.hf_model_name, trust_remote_code=True)
    if tokenizer.chat_template is None:
        raise RuntimeError("No explicit chat template is defined for model-id: ")

    else:
        prompt = tokenizer.apply_chat_template(
            msg,
            tokenize=False,
            thinking=True,
            add_generation_prompt=True,
        )

    return prompt


def test_think_big(m_session: MelleaSession, gh_run: int):
    """Tests big thinking budget."""
    # if on github we can run big thinking mode
    if gh_run == 1:
        return
    if gh_run == 1:
        pytest.skip("Skipping big_thinking runs in gh workflows.")

    prompt = "What is the smallest positive integer $n$ such that all the roots of $z^4 + z^2 + 1 = 0$ are $n^{\\text{th}}$ roots of unity?"
    prompt = prepare_prmpt_for_math(prompt)
    action = CBlock(value=prompt)
    THINK_MAX_TOKENS = 2048
    ANSWER_MAX_TOKENS = 512

    result, gen_tok_cnt = think_budget_forcing(
        m_session.backend,
        action,
        think_max_tokens=THINK_MAX_TOKENS,
        answer_max_tokens=ANSWER_MAX_TOKENS,
        start_think_token="<think>",
        end_think_token="</think>",
        think_wait_suffix="\nWait, let's think more carefully",
        answer_suffix="The final answer is:",
        answer_regex=r"\\boxed{.*?}"
    )

    print("\n******\nThink big:")
    print(str(result))
    assert gen_tok_cnt >= 0.5 * THINK_MAX_TOKENS


def test_think_little(m_session: MelleaSession, gh_run: int):
    """Tests small thinking budget."""
    prompt = "what is 1+1?"
    prompt = prepare_prmpt_for_math(prompt)
    action = CBlock(value=prompt)
    THINK_MAX_TOKENS = 512
    ANSWER_MAX_TOKENS = 256
    if gh_run == 1:  # on github
        THINK_MAX_TOKENS = 5
        ANSWER_MAX_TOKENS = 5

    result, gen_tok_cnt = think_budget_forcing(
        m_session.backend,
        action,
        think_max_tokens=THINK_MAX_TOKENS,
        answer_max_tokens=ANSWER_MAX_TOKENS,
        start_think_token="<think>",
        end_think_token="</think>",
        think_wait_suffix="\nWait, let's think more carefully",
        answer_suffix="The final answer is:",
        answer_regex=r"\\boxed{.*?}"
    )

    print("\n******\nThink little:")
    print(str(result))
    assert gen_tok_cnt <= 2 * THINK_MAX_TOKENS


def test_dont_think(m_session: MelleaSession, gh_run: int):
    """Tests no thinking budget."""
    prompt = "what is 1+1?"
    prompt = prepare_prmpt_for_math(prompt)
    action = CBlock(value=prompt)
    THINK_MAX_TOKENS = 0
    ANSWER_MAX_TOKENS = 512
    if gh_run == 1:
        ANSWER_MAX_TOKENS = 5

    result, gen_tok_cnt = think_budget_forcing(
        m_session.backend,
        action,
        think_max_tokens=THINK_MAX_TOKENS,
        answer_max_tokens=ANSWER_MAX_TOKENS,
        start_think_token="",
        end_think_token="<think> Okay, I think I have finished thinking. </think>",
        think_wait_suffix="",
        answer_suffix="The final answer is:",
    )

    print("\n******\nDon't think:")
    print(str(result))
    assert gen_tok_cnt >= 0.5 * THINK_MAX_TOKENS


if __name__ == "__main__":
    pytest.main(["-s", __file__])
