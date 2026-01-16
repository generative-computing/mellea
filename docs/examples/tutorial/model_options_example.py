import mellea
from mellea.backends import model_ids
from mellea.backends.ollama import OllamaModelBackend
from mellea.backends import ModelOption

m = mellea.MelleaSession(
    backend=OllamaModelBackend(model_options={ModelOption.SEED: 42})
)

answer = m.instruct(
    "What is 2x2?",
    model_options={ModelOption.TEMPERATURE: 0.5, ModelOption.MAX_NEW_TOKENS: 15},
)

print(str(answer))
