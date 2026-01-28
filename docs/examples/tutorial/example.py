import pytest

pytestmark = [pytest.mark.ollama, pytest.mark.llm]
import mellea

m = mellea.start_session()
print(m.chat("What is the etymology of mellea?").content)
