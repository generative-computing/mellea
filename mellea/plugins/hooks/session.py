"""Session lifecycle hook payloads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mellea.plugins.base import MelleaBasePayload

if TYPE_CHECKING:
    pass


class SessionPreInitPayload(MelleaBasePayload):
    """Payload for ``session_pre_init`` — before backend initialization."""

    backend_name: str
    model_id: str
    model_options: dict[str, Any] | None = None
    backend_kwargs: dict[str, Any] = {}
    context_type: str = "SimpleContext"


class SessionPostInitPayload(MelleaBasePayload):
    """Payload for ``session_post_init`` — after session is fully initialized."""

    session: Any = None  # MelleaSession (Any to avoid import issues with frozen model)


class SessionResetPayload(MelleaBasePayload):
    """Payload for ``session_reset`` — when session context is reset."""

    previous_context: Any = None  # Context


class SessionCleanupPayload(MelleaBasePayload):
    """Payload for ``session_cleanup`` — before session cleanup/teardown."""

    context: Any = None  # Context
    interaction_count: int = 0
