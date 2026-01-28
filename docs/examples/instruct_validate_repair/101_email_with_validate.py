from docs.examples.helper import req_print, w
from mellea import start_session
from mellea.backends import ModelOption
from mellea.backends.model_ids import IBM_GRANITE_3_3_8B
from mellea.stdlib.sampling import RejectionSamplingStrategy

# create a session using Granite 4 Micro (3B) on Ollama and a simple context [see below]
m = start_session(model_options={ModelOption.MAX_NEW_TOKENS: 200})

email_v2_samples = m.instruct(
    "Write a very short email to invite all interns to the office party.",
    requirements=["Use formal language.", "Use 'Dear Interns' as greeting."],
    strategy=RejectionSamplingStrategy(loop_budget=3),
    return_sampling_results=True,
)

if email_v2_samples.success:
    print(f"Success: \n{w(email_v2_samples.result)}")
    print(
        f"===> Requirement for this sample: \n{req_print(email_v2_samples.sample_validations[-1])}"
    )
else:
    print(f"Failure: \n{w(email_v2_samples.result)}")
    selected_index = email_v2_samples.sample_generations.index(email_v2_samples.result)
    print(
        f"===> Requirement for this sample: \n{req_print(email_v2_samples.sample_validations[selected_index])}"
    )

# # [optional] get logs for all loops:
# from mellea.stdlib.base import GenerateLog
# _,logs = m.ctx.last_output_and_logs(all_intermediate_results=True)
# assert isinstance(logs, list) and isinstance(logs[0], GenerateLog)
# for i, log in enumerate(logs):
#     print(f"*** Prompt {i} ****\n{w(log.prompt)}\n\n-- RES--- \n{w(log.result)}")
