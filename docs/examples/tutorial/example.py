# pytest: ollama, llm

import mellea

m = mellea.start_session()
print(m.chat("What is the etymology of mellea?").content)
