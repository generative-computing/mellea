import mellea.stdlib.functional as mfuncs
from mellea.stdlib.context import SimpleContext
from mellea.core import CBlock
from mellea.backends.ollama import OllamaModelBackend

response, next_context = mfuncs.act(
    CBlock("What is 1+1?"),
    context=SimpleContext(),
    backend=OllamaModelBackend("granite4:latest"),
)

print(response.value)
