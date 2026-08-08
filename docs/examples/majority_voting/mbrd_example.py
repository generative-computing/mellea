"""
Example usage of minimum Bayes risk decoding (MBRD).

To run this script from the root of the Mellea source tree, use the command:
```
uv run python docs/examples/majority_voting/mbrd_example.py
```
"""

from mellea import MelleaSession
from mellea.backends.vllm import LocalVLLMBackend
from mellea.backends.types import ModelOption
from mellea.backends.model_ids import QWEN3_0_6B
from mellea.stdlib.sampling.majority_voting import MBRDRougeLStrategy

import os

os.environ["VLLM_USE_V1"] = "0"


def solve_using_mbrd(
    m_session: MelleaSession, prompt: str, num_samples: int = 8
) -> str:
    """Solves the problem in `prompt` by generating `num_samples` solutions and
    selecting the one with the highest average RougeL with the rest"""
    # Generate and select the MBR solution
    result = m_session.instruct(
        prompt,
        strategy=MBRDRougeLStrategy(number_of_samples=num_samples, loop_budget=1),
        model_options={
            ModelOption.MAX_NEW_TOKENS: 1024,
            ModelOption.SYSTEM_PROMPT: "Answer in English.",
        },
        return_sampling_results=True,
    )
    raw_output = str(result.result).strip()

    # Do any required post-processing (can be model-specific) and extract the final response
    def postprocess(raw_output: str) -> str:
        # If the raw output contains a thinking section in the beginning, remove it so that
        # the user only sees the actual response that follows the closing `</think>` token
        if "</think>" in raw_output:
            return raw_output.split("</think>")[1].strip()
        return raw_output

    output = postprocess(raw_output)
    return output


# Create a Mellea session for the target use case
max_samples = 8  # indicates that we might want to do MBRD with as many as 8 samples
backend = LocalVLLMBackend(
    model_id=QWEN3_0_6B,
    model_options={
        "gpu_memory_utilization": 0.8,
        "trust_remote_code": True,
        "max_model_len": 2048,
        "max_num_seqs": max_samples,
    },
)
m_session = MelleaSession(backend)

# A few example prompts to test
a_science_prompt = "Why does metal feel colder to the touch than wood?"
a_psycholing_prompt = (
    "Three reasons why children are better at learning languages than adults."
)
a_history_prompt = "Why was the great wall built?"
an_email_prompt = "We have an applicant for an intern position named Olivia Smith. I want to schedule a phone interview with her. Please draft a short email asking her about her availability."

# Let's use the email prompt in this demo
prompt = an_email_prompt

# Demonstrate how to use the MBRD feature
output = solve_using_mbrd(m_session, prompt, num_samples=8)
print(f"\n\nOutput:\n{output}")

# # Cleanup to avoid torch warning unrelated to MBRD (if needed)
# def torch_destroy_process_group():
#     import torch.distributed as dist
#     if dist.is_initialized():
#         dist.destroy_process_group()
# torch_destroy_process_group()
