# pytest: ollama, e2e
"""Email generation constrained by natural language requirements.

Demonstrates constraining outputs using string-based requirements in
`m.instruct()`. Mellea automatically checks requirements against the output
and repairs the result if needed.

Usage:
    uv run python docs/examples/instruct_validate_repair/101_email_with_requirements.py
"""

from docs.examples.helper import w
from mellea import start_session
from mellea.backends import ModelOption

m = start_session(model_options={ModelOption.MAX_NEW_TOKENS: 200})

email_v1 = m.instruct(
    "Write an email to invite all interns to the office party.",
    requirements=["be formal", "Use 'Dear Interns' as greeting."],
)

print(f"***** email ****\n{w(email_v1)}\n*******")
