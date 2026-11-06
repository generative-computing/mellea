# pytest: ollama, e2e
"""Working with contexts and context management.

Prerequisites:
    ollama serve
    ollama pull granite4.2:3b

Usage:
    uv run python docs/examples/tutorial/context_example.py
"""

from mellea import start_session
from mellea.stdlib.context import ChatContext

m = start_session(ctx=ChatContext())
m.chat("Make up a math problem.")
m.chat("Solve your math problem.")

print(m.ctx.last_output())
print("==================")
print(m.ctx.last_turn())
