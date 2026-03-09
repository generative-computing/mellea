"""Session lifecycle hook payloads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mellea.plugins.base import MelleaBasePayload, WeakProxy

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
        session: The fully initialized ``MelleaSession`` instance (observe-only).
            Held as a weak reference — do not cache this payload.
    """

    session: WeakProxy = None


class SessionResetPayload(MelleaBasePayload):
    """Payload for ``session_reset`` — when session context is reset.

    Attributes:
        previous_context: The ``Context`` that is about to be discarded (observe-only).
            Held as a weak reference — do not cache this payload.
    """

    previous_context: WeakProxy = None


class SessionCleanupPayload(MelleaBasePayload):
    """Payload for ``session_cleanup`` — before session cleanup/teardown.

    Attributes:
        context: The ``Context`` at the time of cleanup (observe-only).
            Held as a weak reference — do not cache this payload.
        interaction_count: Number of items in the context at cleanup time.
    """

    context: WeakProxy = None
    interaction_count: int = 0
