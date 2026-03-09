"""Generation pipeline hook payloads."""

from __future__ import annotations

from typing import Any

from mellea.plugins.base import MelleaBasePayload, WeakProxy


class GenerationPreCallPayload(MelleaBasePayload):
    """Payload for ``generation_pre_call`` — before LLM backend call.

    Attributes:
        action: The ``Component`` or ``CBlock`` about to be sent to the backend.
            Held as a weak reference — do not cache this payload.
        context: The ``Context`` being used for this generation call.
            Held as a weak reference — do not cache this payload.
        model_options: Dict of model options (writable — plugins may adjust temperature, etc.).
        format: Optional ``BaseModel`` subclass for constrained decoding (writable).
        tool_calls: Whether tool calls are enabled for this generation (writable).
    """

    action: WeakProxy = None
    context: WeakProxy = None
    model_options: dict[str, Any] = {}
    format: Any = None
    tool_calls: bool = False


class GenerationPostCallPayload(MelleaBasePayload):
    """Payload for ``generation_post_call`` — fires once the model output is fully computed.

    For lazy ``ModelOutputThunk`` objects this hook fires inside
    ``ModelOutputThunk.astream`` after ``post_process`` completes, so
    ``model_output.value`` is guaranteed to be available. For already-computed
    thunks (e.g. cached responses) it fires before ``generate_from_context``
    returns.

    Attributes:
        prompt: The formatted prompt sent to the backend (str or list of message dicts).
        model_output: The fully-computed ``ModelOutputThunk`` (writable — replacing
            it is supported on both lazy and already-computed paths. On the lazy
            path the original MOT's output fields are updated in-place via
            ``_copy_from``).
        latency_ms: Elapsed milliseconds from the ``generate_from_context`` call
            to when the value was fully materialized.
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
