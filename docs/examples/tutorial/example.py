# pytest: ollama, e2e

import mellea

m = mellea.start_session()
print(m.chat("What is the etymology of mellea?").content)
