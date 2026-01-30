import mellea.stdlib.functional as mfuncs
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.context import SimpleContext

response, next_context = mfuncs.chat(
    "What is 1+1?",
    context=SimpleContext(),
    backend=OllamaModelBackend("granite4:latest"),
)

print(response.content)
