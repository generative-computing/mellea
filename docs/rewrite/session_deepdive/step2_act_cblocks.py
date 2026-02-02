import mellea.stdlib.functional as mfuncs
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.components import Instruction
from mellea.stdlib.context import SimpleContext

response, next_context = mfuncs.act(
    action=Instruction("What is 1+1?"),
    context=SimpleContext(),
    backend=OllamaModelBackend("granite4:latest"),
)

print(response.value)
