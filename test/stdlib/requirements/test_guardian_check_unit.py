# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

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

    ctx = (
        ChatContext()
        .add(Message("user", "Is this safe?"))
        .add(Message("assistant", "Yes."))
    )
    result = await req.validate(backend, ctx)

    assert result.as_bool() is True
    assert result.reason is not None
    assert "Reasoning: grounded in provided content" in result.reason


@pytest.mark.asyncio
async def test_guardian_function_call_risk_with_assistant_message() -> None:
    """function_call risk should extract content from Message(role='assistant') in last_turn.output."""
    # Mock backend that returns a successful validation response
    mot = ModelOutputThunk(value="<score>yes</score>")
    backend = MagicMock()
    backend.generate_from_context = AsyncMock(return_value=(mot, ChatContext()))

    with pytest.warns(DeprecationWarning, match="GuardianCheck is deprecated"):
        req = GuardianCheck(
            risk="function_call", backend=backend, backend_type="ollama"
        )

    # Build context with manually-added assistant Message (simulating extracted content)
    ctx = (
        ChatContext()
        .add(Message("user", "Execute this function"))
        .add(Message("assistant", "Executing function X"))
    )
    await req.validate(backend, ctx)

    # Verify validation succeeded and backend.generate_from_context was called
    # The key assertion is that no exception was raised when extracting content from Message
    assert backend.generate_from_context.called
