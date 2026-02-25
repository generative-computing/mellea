from mellea import MelleaSession
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.context import SimpleContext

m = MelleaSession(backend=OllamaModelBackend("granite4:latest"), ctx=SimpleContext())
response = m.chat("What is 1+1?")
print(response.content)
