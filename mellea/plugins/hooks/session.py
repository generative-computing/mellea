"""Session lifecycle hook payloads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mellea.plugins.base import MelleaBasePayload

if TYPE_CHECKING:
    pass


class SessionPreInitPayload(MelleaBasePayload):
    """Payload for ``session_pre_init`` — before backend initialization.

    Attributes:
        backend_name: Name of the backend (e.g. ``"ollama"``, ``"openai"``) (writable).
        model_id: Model identifier string (writable).
        model_options: Optional dict of model options like temperature, max_tokens (writable).
        context_type: Class name of the context being used (e.g. ``"SimpleContext"``).
    """

    backend_name: str
    model_id: str
    model_options: dict[str, Any] | None = None
    context_type: str = "SimpleContext"


class SessionPostInitPayload(MelleaBasePayload):
    """Payload for ``session_post_init`` — after session is fully initialized.

    Attributes:
        session_id: UUID string identifying this session.
        model_id: Model identifier used by the backend (e.g. ``"granite4:micro"``).
        context: The initial ``Context`` instance for this session.
    """

    session_id: str = ""
    model_id: str = ""
    context: Any = None


class SessionResetPayload(MelleaBasePayload):
    """Payload for ``session_reset`` — when session context is reset.

    Attributes:
        previous_context: The ``Context`` that is about to be discarded (observe-only).

    """

    previous_context: Any = None


class SessionCleanupPayload(MelleaBasePayload):
    """Payload for ``session_cleanup`` — before session cleanup/teardown.

    Attributes:
        context: The ``Context`` at the time of cleanup (observe-only).

        interaction_count: Number of items in the context at cleanup time.
    """

    context: Any = None
    interaction_count: int = 0
