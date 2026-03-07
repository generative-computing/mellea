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
        format: Optional ``BaseModel`` subclass for constrained decoding (writable).
        tool_calls: Whether tool calls are enabled for this generation (writable).
    """

    action: Any = None
    context: Any = None
    model_options: dict[str, Any] = {}
    format: Any = None
    tool_calls: bool = False


class GenerationPostCallPayload(MelleaBasePayload):
    """Payload for ``generation_post_call`` — after LLM response received.

    Attributes:
        prompt: The formatted prompt sent to the backend (str or list of message dicts).
        model_output: The ``ModelOutputThunk`` returned by the backend (writable).
        latency_ms: Always ``0`` at this call site — the ``ModelOutputThunk`` is
            uncomputed (lazy) when the hook fires. TODO: move hook into
            ``ModelOutputThunk.astream`` (after ``post_process``) where real
            latency can be measured.
    """

    prompt: str | list[dict[str, Any]] = ""
    model_output: Any = None
    latency_ms: float = 0.0


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
