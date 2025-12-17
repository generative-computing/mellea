import mellea.stdlib.functional as mfuncs
from mellea.stdlib.base import SimpleContext, CBlock
from mellea.backends.ollama import OllamaModelBackend

response, next_context = mfuncs.act(CBlock("What is 1+1?"), 
                                     context=SimpleContext(), 
                                     backend=OllamaModelBackend("granite4:latest"))

print(response.value)