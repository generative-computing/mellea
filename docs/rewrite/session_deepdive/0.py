from mellea import MelleaSession
from mellea.stdlib.base import SimpleContext
from mellea.backends.ollama import OllamaModelBackend


m = MelleaSession(
    backend=OllamaModelBackend("granite4:latest"), context=SimpleContext()
)
response = m.chat("What is 1+1?")
print(response.content)
