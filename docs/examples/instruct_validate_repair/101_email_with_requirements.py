from docs.examples.helper import w
from mellea import start_session
from mellea.backends.types import ModelOption

# create a session using Granite 3.3 8B on Ollama and a simple context [see below]
m = start_session(model_options={ModelOption.MAX_NEW_TOKENS: 200})

# write an email with automatic requirement checking.
email_v1 = m.instruct(
    "Write an email to invite all interns to the office party.",
    requirements=["be formal", "Use 'Dear Interns' as greeting."],
)

# print result
print(f"***** email ****\n{w(email_v1)}\n*******")

# # optionally, for debugging.. print last prompt
# print(f"Prompt:\n{w(m.last_prompt())}")
