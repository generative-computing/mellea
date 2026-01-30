# pytest: ollama, llm

from docs.examples.helper import w
from mellea import start_session
from mellea.backends import ModelOption
from mellea.stdlib.requirements import simple_validate
from mellea.core import Requirement
from mellea.stdlib.sampling import RejectionSamplingStrategy

# create a session using Granite 4 Micro 3B on Ollama and a simple context [see below]
m = start_session(model_options={ModelOption.MAX_NEW_TOKENS: 200})

# Define a requirement which checks that the output is less than 100 words
# using a custom validate function
word_limit_req = Requirement(
    "Use less than 100 words.",
    validation_fn=simple_validate(lambda output: len(output.split(" ")) < 100),
)

email_v1 = m.instruct(
    "Write an email to invite all interns to the office party.",
    requirements=["be formal", word_limit_req],
    strategy=RejectionSamplingStrategy(loop_budget=3),
)

# print result
print(f"***** email ****\n{w(email_v1)}\n*******")

# # [optional] get logs for all loops:
# from mellea.stdlib.base import GenerateLog
# _,logs = m.ctx.last_output_and_logs(all_intermediate_results=True)
# assert isinstance(logs, list) and isinstance(logs[0], GenerateLog)
# for i, log in enumerate(logs):
#     print(f"*** Prompt {i} ****\n{w(log.prompt)}\n\n-- RES--- \n{w(log.result)}")
