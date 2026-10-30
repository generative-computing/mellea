# pytest: ollama, e2e
"""Configuring model options and parameters.

Prerequisites:
    ollama serve
    ollama pull granite4.2:3b

Usage:
    uv run python docs/examples/tutorial/model_options_example.py
"""

import mellea
from mellea.backends import ModelOption, model_ids
from mellea.backends.ollama import OllamaModelBackend

m = mellea.MelleaSession(
    backend=OllamaModelBackend(model_options={ModelOption.SEED: 42})
)

answer = m.instruct(
    "What is 2x2?",
    model_options={ModelOption.TEMPERATURE: 0.5, ModelOption.MAX_NEW_TOKENS: 15},
)

print(str(answer))
