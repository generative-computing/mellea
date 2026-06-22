"""Unit tests for GuardianCheck requirement behavior."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from mellea.core import ModelOutputThunk
from mellea.stdlib.components import Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements.safety.guardian import GuardianCheck


@pytest.mark.asyncio
async def test_guardian_validate_uses_thinking_trace_in_reason() -> None:
    """validate() should include explicit mot.thinking content in the reason."""
    mot = ModelOutputThunk(value="<score>no</score>")
    mot.thinking = "grounded in provided content"

    backend = MagicMock()
    backend.generate_from_context = AsyncMock(return_value=(mot, ChatContext()))

    with pytest.warns(DeprecationWarning, match="GuardianCheck is deprecated"):
        req = GuardianCheck(risk="harm", backend=backend, backend_type="ollama")

    ctx = ChatContext().add(Message("user", "Is this safe?")).add(
        Message("assistant", "Yes.")
    )
    result = await req.validate(backend, ctx)

    assert result.as_bool() is True
    assert result.reason is not None
    assert "Reasoning: grounded in provided content" in result.reason
