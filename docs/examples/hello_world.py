# pytest: ollama, e2e
"""Minimal single-file starting point demonstrating email generation.

Prerequisites:
    ollama serve
    ollama pull granite4.2:3b

Usage:
    uv run python docs/examples/hello_world.py
"""

import mellea

m = mellea.start_session()

email = m.instruct("Write an email inviting the interns to a lunch party.")

print(email)
