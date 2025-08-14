# This is the 101 example for using `session` and `instruct`.
# helper function to wrap text
from docs.examples.helper import w
from mellea import session, instruct
from mellea.backends.types import ModelOption

# create a session using Granite 3.3 8B on Ollama and a simple context [see below]
with session(model_options={ModelOption.MAX_NEW_TOKENS: 200}):
# write an email
    email_v1 = instruct("Write an email to invite all interns to the office party.")

with session(model_options={ModelOption.MAX_NEW_TOKENS: 200}) as m:
# write an email
    email_v1 = m.instruct("Write an email to invite all interns to the office party.")

# print result
print(f"***** email ****\n{w(email_v1)}\n*******")

# ************** END *************


# # optionally: print the debug log for the last instruction on the context
# from mellea.stdlib.base import GenerateLog
# _, log = m.ctx.last_output_and_logs()
# if isinstance(log, GenerateLog): # should be
#     print(f"Prompt:\n{w(log.prompt)}") # print prompt

# # start_session() is equivalent to:
# from mellea.backends import model_ids
# from mellea.backends.ollama import OllamaModelBackend
# from mellea import MelleaSession, SimpleContext
# m = MelleaSession(
#     backend=OllamaModelBackend(
#         model_id=model_ids.IBM_GRANITE_3_3_8B,
#         model_options={ModelOption.MAX_NEW_TOKENS: 200},
#     ),
#     ctx=SimpleContext()
# )
