import mellea.stdlib.functional as mfuncs
from mellea.backends.ollama import OllamaModelBackend
from mellea.core import CBlock
from mellea.stdlib.context import SimpleContext

response, next_context = mfuncs.act(
    action=CBlock("What is 1+1?"),
    context=SimpleContext(),
    backend=OllamaModelBackend("granite4:latest"),
)

print(response.value)
