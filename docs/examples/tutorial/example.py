# pytest: ollama, e2e
"""General introduction to Mellea with basic examples.

Prerequisites:
    ollama serve
    ollama pull granite4.2:3b

Usage:
    uv run python docs/examples/tutorial/example.py
"""

import mellea

m = mellea.start_session()
print(m.chat("What is the etymology of mellea?").content)
