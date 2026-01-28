try:
    import pytest

    pytestmark = [pytest.mark.ollama, pytest.mark.llm]
except ImportError:
    pass  # Running standalone, pytest not available
import mellea

m = mellea.start_session()
print(m.chat("What is the etymology of mellea?").content)
