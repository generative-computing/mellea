"""Concrete ``Context`` implementations and the ``Compactor`` protocol.

Provides:

- :class:`ChatContext` — accumulates all turns in a chat history (with an
  optional sliding window).
- :class:`SimpleContext` — stateless, single-turn exchange (no prior history is
  passed to the model).
- :class:`Compactor` — generic protocol for shrinking any ``Context`` subtype.

The names :class:`Context`, :class:`ContextTurn`, :class:`CBlock`, and
:class:`Component` are re-exported from :mod:`mellea.core` for the convenience
of callers that import them via ``mellea.stdlib.context``.
"""

from mellea.core import CBlock, Component, Context, ContextTurn
from mellea.stdlib.context.chat import ChatContext
from mellea.stdlib.context.compactor import (
    Compactor,
    InlineCompactor,
    LLMSummarizeCompactor,
    PinPredicate,
    ThresholdCompactor,
    WindowCompactor,
    pin_nothing,
    pin_system,
    pin_system_and_initial_user,
)
from mellea.stdlib.context.simple import SimpleContext

__all__ = [
    "CBlock",
    "ChatContext",
    "Compactor",
    "Component",
    "Context",
    "ContextTurn",
    "InlineCompactor",
    "LLMSummarizeCompactor",
    "PinPredicate",
    "SimpleContext",
    "ThresholdCompactor",
    "WindowCompactor",
    "pin_nothing",
    "pin_system",
    "pin_system_and_initial_user",
]
