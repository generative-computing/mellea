from mellea import MelleaSession
from mellea.stdlib.context import SimpleContext
from mellea.backends.ollama import OllamaModelBackend


m = MelleaSession(
    backend=OllamaModelBackend("granite4:latest"), ctx=SimpleContext()
)
response = m.chat("What is 1+1?")
print(response.content)
