import mellea
from mellea import MelleaSession
from mellea.backends.model_ids import OPENAI_GPT_OSS_20B, META_LLAMA_3_2_1B, IBM_GRANITE_4_TINY_PREVIEW_7B
from mellea.stdlib.base import CBlock, SimpleContext
from mellea.stdlib.requirement import check, req, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy, MajorityVotingStrategyForMath
from mellea.backends.openai import OpenAIBackend
from mellea.backends.formatter import TemplateFormatter
from transformers import AutoTokenizer
import pytest
import os


class TestMajorityVoting:
    MODEL_ID = os.environ.get("LOCAL_TEST_MODEL", None)
    if MODEL_ID is None:
        raise RuntimeError(f"Must set environment variable `LOCAL_TEST_MODEL` to a HF model id")

    # Local testing mode
    if MODEL_ID == "ibm-granite/granite-4.0-tiny-preview":
        MODEL_ID = IBM_GRANITE_4_TINY_PREVIEW_7B

    elif MODEL_ID == "unsloth/Llama-3.2-1B":
        MODEL_ID = META_LLAMA_3_2_1B

    else:
        raise RuntimeError(f"Unsupported model-id:{MODEL_ID}")

    model_id = "ibm-granite/granite-4.0-tiny-preview"
    backend = OpenAIBackend(
        model_id=MODEL_ID,
        formatter=TemplateFormatter(model_id=MODEL_ID),
        base_url=f"http://{os.environ.get('OLLAMA_HOST', 'localhost:8000')}/v1",
        api_key="ollama",
    )

    m = MelleaSession(backend, ctx=SimpleContext())
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID.hf_model_name, trust_remote_code=True)


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
        if self.tokenizer.chat_template is None:
            raise RuntimeError(f"No explicit chat template is defined for model-id: ")

        else:
            prompt = self.tokenizer.apply_chat_template(
                msg,
                tokenize=False,
                thinking=True,
                add_generation_prompt=True,
            )

        return prompt

    def test_majority_voting(self):

        query = "Compute 1+1"
        prompt = self.prepare_prmpt_for_math(query)

        # requirements = [
        #     req("The email should have a salutation"),  # == r1
        #     req(
        #         "Use only lower-case letters",
        #         validation_fn=simple_validate(lambda x: x.lower() == x),
        #     ),  # == r2
        #     check("Do not mention purple elephants."),  # == r3
        # ]

        # def write_email(m: mellea.MelleaSession, name: str, notes: str) -> str:
        #     breakpoint()
        #     email_candidate = m.instruct(
        #         "Write an email to {{name}} using the notes following: {{notes}}.",
        #         # requirements=requirements,
        #         strategy=RejectionSamplingStrategy(loop_budget=5),
        #         user_variables={"name": name, "notes": notes},
        #         return_sampling_results=True,
        #     )
        result = self.m.instruct(
            prompt,
            # requirements=requirements,
            strategy=MajorityVotingStrategyForMath(number_of_samples=8, loop_budget=1),
            # user_variables={"name": name, "notes": notes},
            return_sampling_results=True,
        )
        if result.success:
            output =  str(result.result)
        else:
            output =  result.sample_generations[0].value
        print(output)


        # assert gen_tok_cnt <= 2 * THINK_MAX_TOKENS

if __name__ == "__main__":
    pytest.main(["-s", __file__])
