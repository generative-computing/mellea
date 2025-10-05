from docs.examples.helper import w
from mellea import start_session
from mellea.backends.types import ModelOption

# create a session using Granite 3.3 8B on Ollama and a simple context [see below]
m = start_session(model_options={ModelOption.MAX_NEW_TOKENS: 200})

# Write a more formal and a more funny email
email_v1 = m.instruct("Write an email to invite all interns to the office party.")
email_v2 = m.instruct(
    "Write a very funny email to invite all interns to the office party."
)
print(
    f"***** email 1 ****\n{w(email_v1)}\n*******email 2 ******\n{w(email_v2)}\n\n*******"
)

# Use the emails as grounding context to evaluate which one is quirkier
email_checking = m.instruct(
    "Which email is more quirky?",
    grounding_context={"email_1": email_v1, "email_2": email_v2},
)
print(w(email_checking))
