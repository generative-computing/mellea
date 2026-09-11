# pytest: ollama, e2e

"""Demonstrates ModelOption.THINKING against a local Ollama Granite model.

Requires `ollama serve` running with `granite4.2:3b` pulled
(`ollama pull granite4.2:3b`).
"""

import mellea
from mellea.backends import ModelOption, model_ids
from mellea.backends.ollama import OllamaModelBackend

m = mellea.MelleaSession(
    backend=OllamaModelBackend(model_id=model_ids.IBM_GRANITE_4_2_3B)
)

question = "What is 17 * 24?"

# Full reasoning — Granite 4.2 thinks by default even without THINKING set,
# but setting it explicitly makes the intent visible in the code.
full = m.instruct(question, model_options={ModelOption.THINKING: True})
print("=== THINKING=True ===")
print("thinking:", full.thinking)
print("answer:", full.value)
assert full.thinking

# Low-effort reasoning — a short trace, useful when the token budget is tight.
low = m.instruct(question, model_options={ModelOption.THINKING: "low"})
print("=== THINKING='low' ===")
print("thinking:", low.thinking)
print("answer:", low.value)
assert low.thinking

# No reasoning at all.
off = m.instruct(question, model_options={ModelOption.THINKING: False})
print("=== THINKING=False ===")
print("thinking:", off.thinking)
print("answer:", off.value)
assert not off.thinking
