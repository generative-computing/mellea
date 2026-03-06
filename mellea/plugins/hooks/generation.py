"""Generation pipeline hook payloads."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mellea.plugins.base import MelleaBasePayload


class GenerationPreCallPayload(MelleaBasePayload):
    """Payload for ``generation_pre_call`` — before LLM backend call.

    Attributes:
        action: The ``Component`` or ``CBlock`` about to be sent to the backend.
        context: The ``Context`` being used for this generation call.
        model_options: Dict of model options (writable — plugins may adjust temperature, etc.).
        tools: Optional dict mapping tool names to callables (writable).
        format: Optional ``BaseModel`` subclass for constrained decoding (writable).
    """

    action: Any = None
    context: Any = None
    model_options: dict[str, Any] = {}
    tools: dict[str, Callable] | None = None
    format: Any = None


class GenerationPostCallPayload(MelleaBasePayload):
    """Payload for ``generation_post_call`` — after LLM response received.

    .. note::
        This hook fires immediately after ``generate_from_context`` returns.
        The ``ModelOutputThunk`` is typically **uncomputed** (lazy) at this
        point, so ``model_output.value`` may be ``None``.  ``latency_ms``
        is always ``0`` here; accurate timing requires the hook to move into
        ``ModelOutputThunk.astream`` (see backend.py TODO).

    Attributes:
        prompt: The formatted prompt sent to the backend (str or list of message dicts).
        model_output: The ``ModelOutputThunk`` returned by the backend (writable).
        latency_ms: Always ``0`` — see note above.
    """

    prompt: str | list[dict[str, Any]] = ""
    model_output: Any = None
    latency_ms: int = 0


class GenerationStreamChunkPayload(MelleaBasePayload):
    """Payload for ``generation_stream_chunk`` — for each streaming chunk.

    Attributes:
        chunk: The new text chunk received in this streaming event (writable).
        accumulated: All text received so far, including this chunk (writable).
        chunk_index: 0-based index of this chunk in the stream.
        is_final: ``True`` if this is the last chunk in the stream.
    """

    chunk: str = ""
    accumulated: str = ""
    chunk_index: int = 0
    is_final: bool = False
