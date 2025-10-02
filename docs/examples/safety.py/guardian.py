"""Example of using the Guardian Requirement."""

from mellea import MelleaSession
from mellea.backends import model_ids
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.base import ContextTurn, ModelOutputThunk
from mellea.stdlib.chat import Message
from mellea.stdlib.safety.guardian import GuardianCheck, GuardianRisk

# Start with instantiating a GuardianCheck.
guardian = GuardianCheck(risk=GuardianRisk.HARM)

print("\n Test 1\n")
# Run a query against an Ollama model
m = MelleaSession(OllamaModelBackend(model_ids.DEEPSEEK_R1_8B))
m.chat("Write an  email to your colleague. Use less than 100 words.")
print(f"Full context: {m.ctx}")
# Validate on the last turn of the conversation
valid = m.validate([guardian])
# Should return [True] meaning that input and output are free of harm and it's safe to return result
print(f"Guardian result: {valid}")

# Reproducing a harm scenario and run the validation again
print("\n Test 2\n")

# create a mean conversation and add to context
m.ctx = m.ctx.add(Message("user", "Hello. ")).add(
    ModelOutputThunk("You are very ugly.")
)
# show last turn in chat
print(f"Context: {m.ctx.last_turn()}")

check_results = m.validate([guardian])
# Should return [False] meaning that input and output contain harm and it's NOT safe to return result
print(f"Guardian check results: {check_results}")
