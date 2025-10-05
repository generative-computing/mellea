# This is the 101 example for using `session` and `instruct`.
# helper function to wrap text
from docs.examples.helper import w
from mellea import start_session
from mellea.backends.types import ModelOption

with start_session(model_options={ModelOption.MAX_NEW_TOKENS: 200}) as m:
    # write an email
    email_v1 = m.instruct("Write an email to invite all interns to the office party.")
    print(m.last_prompt())

# print result
print(f"***** email ****\n{w(email_v1)}\n*******")

# ************** END *************

# # start_session() is equivalent to:
# from mellea.backends import model_ids
# from mellea.backends.ollama import OllamaModelBackend
# from mellea import MelleaSession
# from mellea.stdlib.base import SimpleContext
# m = MelleaSession(
#     backend=OllamaModelBackend(
#         model_id=model_ids.IBM_GRANITE_4_MICRO_3B,
#         model_options={ModelOption.MAX_NEW_TOKENS: 200},
#     ),
#     ctx=SimpleContext()
# )
