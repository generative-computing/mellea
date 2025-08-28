from mellea import MelleaSession
from mellea.stdlib.base import CBlock, SimpleContext
from mellea.backends.openai import OpenAIBackend
from transformers import AutoTokenizer
import pytest
import os

class TestOpenAIBackend:
    model_id = "ibm-granite/granite-4.0-tiny-preview"
    backend = OpenAIBackend(
        model_id=model_id,
        base_url="http://0.0.0.0:8000/v1",
        api_key="EMPTY",
    )
    m = MelleaSession(backend, ctx=SimpleContext())
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    def prepare_prmpt_for_math(self, query):
        # Preparing prompt for math reasoning tasks
        system_prompt = None  # Use default of chat template
        prompt_suffix = "\nPlease reason step by step, use \n\n to end each step, and put your final answer within \\boxed{}."

        if prompt_suffix:
            query += prompt_suffix

        msg = []
        if system_prompt is not None:
            msg.append({"role": "system", "content": system_prompt})

        msg.append({"role": "user", "content": query})
        prompt = self.tokenizer.apply_chat_template(
            msg,
            tokenize=False,
            thinking=True,
            add_generation_prompt=True,
        )

        return prompt

    def test_generate_from_raw_small(self):
        prompt = "what is 1+1?"
        prompt = self.prepare_prmpt_for_math(prompt)
        action = CBlock(value=prompt)
        results = []
        THINK_MAX_TOKENS = 64
        ANSWER_MAX_TOKENS = 16
        result, gen_tok_cnt = self.m.backend.generate_with_budget_forcing(
            action=action,
            think_max_tokens=THINK_MAX_TOKENS,
            answer_max_tokens=ANSWER_MAX_TOKENS,
            start_think_token = "<think>",
            end_think_token="</think>",
            think_wait_suffix="Wait",
            answer_suffix="The final answer is:",
            # answer_suffix="",
            answer_token="boxed",
        )

        assert gen_tok_cnt <= 2 * THINK_MAX_TOKENS


    def test_generate_from_raw_large(self):
        prompt = "what is 1+1?"
        prompt = self.prepare_prmpt_for_math(prompt)
        action = CBlock(value=prompt)
        results = []
        THINK_MAX_TOKENS = 1024
        ANSWER_MAX_TOKENS = 256
        result, gen_tok_cnt = self.m.backend.generate_with_budget_forcing(
            action=action,
            think_max_tokens=THINK_MAX_TOKENS,
            answer_max_tokens=ANSWER_MAX_TOKENS,
            start_think_token = "<think>",
            end_think_token="</think>",
            think_wait_suffix="Wait",
            answer_suffix="The final answer is:",
            answer_token="boxed",
        )

        assert gen_tok_cnt >= 0.5 * THINK_MAX_TOKENS


if __name__ == "__main__":
    pytest.main(["-s", __file__])
