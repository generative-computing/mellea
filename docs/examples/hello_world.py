try:
    import pytest

    pytestmark = [pytest.mark.ollama, pytest.mark.llm]
except ImportError:
    pass  # Running standalone, pytest not available

import mellea

m = mellea.start_session()

email = m.instruct("Write an email inviting the interns to a lunch party.")

print(email)
