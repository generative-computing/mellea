import mellea.stdlib.functional as mfuncs
from mellea.stdlib.base import SimpleContext
from mellea.backends.ollama import OllamaModelBackend

response, next_context = mfuncs.chat("What is 1+1?", 
                                     context=SimpleContext(), 
                                     backend=OllamaModelBackend("granite4:latest"))

print(response.content)