# pytest: ollama, e2e
"""Basic email generation using instruct.

Demonstrates creating a basic Mellea session using `start_session()` and
generating text with `m.instruct()` without constraints, as well as inspecting
the underlying prompt with `m.last_prompt()`.

Usage:
    uv run python docs/examples/instruct_validate_repair/101_email.py
"""

from docs.examples.helper import w
from mellea import start_session
from mellea.backends import ModelOption

with start_session(model_options={ModelOption.MAX_NEW_TOKENS: 200}) as m:
    email_v1 = m.instruct("Write an email to invite all interns to the office party.")
    print(m.last_prompt())

print(f"***** email ****\n{w(email_v1)}\n*******")
